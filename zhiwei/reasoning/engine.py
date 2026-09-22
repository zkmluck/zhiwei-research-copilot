"""科研推理引擎：问答、细粒度抽取、跨文献对比。

三个能力共用同一条流水线，这也是本项目的骨架：

    改写检索式 -> 混合检索 -> 组材料 -> 让模型产出「主张 + 逐字原文片段」
    -> 把片段反查回精确锚点 -> 逐句过闸门 -> 只让通过的主张出口

关键点在于**出口是闸门决定的，不是模型决定的**。
模型只负责「提议」，闸门负责「放行」。所以幻觉不是被"减少"了，
而是从结构上到不了用户面前 —— 到不了的证据，会在回答末尾如实列出被拦了几条。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from ..config import settings
from ..contracts import (
    Answer,
    Claim,
    Evidence,
    GateDecision,
    Verdict,
    dedupe_evidence,
    normalize_text,
)
from ..llm import prompts
from ..llm.client import Gateway, GatewayError, get_gateway
from .claim_gate import ClaimGate, GateConfig
from .evidence import GateContext, attach_anchors, split_sentences


@dataclass
class Material:
    """给模型看的一条材料。"""

    index: int
    doc_id: str
    doc_title: str
    page: int
    text: str
    section_path: list[str] = field(default_factory=list)
    year: Optional[int] = None

    def to_block(self) -> dict:
        return {
            "index": self.index,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "page": self.page,
            "text": self.text,
            "section_path": self.section_path,
            "year": self.year,
        }


class ScholarEngine:
    """把「检索 + 生成 + 闸门」串起来。"""

    def __init__(
        self,
        library: Any,
        index: Any,
        gate: Optional[ClaimGate] = None,
        gateway: Optional[Gateway] = None,
    ) -> None:
        self.library = library
        self.index = index
        self.gateway = gateway or get_gateway()
        self.gate = gate or ClaimGate(gateway=self.gateway)

    # ==================================================================
    # 公共：检索
    # ==================================================================
    def _rewrite(self, question: str, trace: list[dict]) -> list[str]:
        """把问题改写成多条检索式，提升召回。失败就原样检索。"""
        queries = [question]
        if not self.gateway.available:
            return queries
        started = time.time()
        try:
            data = self.gateway.json(
                [
                    {"role": "system", "content": prompts.REWRITE_SYSTEM},
                    {"role": "user", "content": prompts.build_rewrite_user(question)},
                ],
                model=settings.fast_model,
                max_tokens=300,
                kind="rag.rewrite",
            )
            extra = [str(q).strip() for q in (data.get("queries") or []) if str(q).strip()]
            queries.extend(q for q in extra[:3] if q not in queries)
            trace.append(
                {
                    "step": "rewrite",
                    "detail": f"把问题改写成 {len(queries)} 条检索式：" + "；".join(queries[1:4]),
                    "ms": int((time.time() - started) * 1000),
                }
            )
        except GatewayError as exc:
            trace.append(
                {
                    "step": "rewrite",
                    "detail": f"检索式改写不可用（{exc.kind}），使用原问题检索",
                    "ms": int((time.time() - started) * 1000),
                }
            )
        return queries

    def retrieve(
        self,
        question: str,
        *,
        doc_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        trace: Optional[list[dict]] = None,
    ) -> list[Evidence]:
        trace = trace if trace is not None else []
        top_k = top_k or settings.top_k
        started = time.time()
        queries = self._rewrite(question, trace)
        collected: list[Evidence] = []
        for q in queries:
            try:
                collected.extend(self.index.search(q, doc_ids=doc_ids, top_k=top_k))
            except Exception:
                # 检索层异常不能让整个问答挂掉
                continue
        merged = dedupe_evidence(collected)
        merged = self._boost_section_match(question, merged)
        merged = merged[: max(top_k, 6)]
        trace.append(
            {
                "step": "retrieve",
                "detail": f"混合检索到 {len(merged)} 段候选材料（来自 {len(set(e.doc_id for e in merged))} 篇文献）",
                "ms": int((time.time() - started) * 1000),
            }
        )
        return merged

    @staticmethod
    def _boost_section_match(question: str, evidences: list[Evidence]) -> list[Evidence]:
        """问题里点名了某个章节/术语时，把同章节的证据往前排（很小的排序修正）。"""
        terms = {t for t in normalize_text(question).lower().split() if len(t) >= 3}
        if not terms:
            return evidences

        def key(ev: Evidence) -> tuple:
            hit = sum(1 for t in terms if t in (ev.quote or "").lower())
            return (-hit, -ev.score)

        return sorted(evidences, key=key)

    # ==================================================================
    # 公共：材料与闸门上下文
    # ==================================================================
    def _material(self, evidences: list[Evidence]) -> list[Material]:
        materials: list[Material] = []
        for i, ev in enumerate(evidences, start=1):
            title = ev.doc_title or ev.doc_id
            year = None
            try:
                paper = self.library.get_paper(ev.doc_id)
                if paper is not None:
                    title = paper.title or title
                    year = paper.year
            except Exception:
                pass
            materials.append(
                Material(
                    index=i,
                    doc_id=ev.doc_id,
                    doc_title=title,
                    page=ev.anchor.page,
                    text=ev.quote,
                    year=year,
                )
            )
        return materials

    def _gate_context(self, doc_ids: list[str]) -> GateContext:
        """从档案库取每页纯文本，供闸门把引文反查回位置。"""
        page_texts: dict[str, dict[int, str]] = {}
        titles: dict[str, str] = {}
        for doc_id in doc_ids:
            try:
                page_texts[doc_id] = self.library.load_page_texts(doc_id) or {}
            except Exception:
                page_texts[doc_id] = {}
            try:
                paper = self.library.get_paper(doc_id)
                if paper is not None:
                    titles[doc_id] = paper.title or doc_id
            except Exception:
                titles[doc_id] = doc_id
        return GateContext(page_texts=page_texts, doc_titles=titles)

    def _vet(
        self,
        raw_claims: list[dict],
        *,
        doc_ids: list[str],
        fallback_doc_id: str = "",
        trace: Optional[list[dict]] = None,
    ) -> list[Claim]:
        """把模型产出的原始主张走完「反查锚点 -> 逐句过闸」全流程。"""
        started = time.time()
        context = self._gate_context(doc_ids)
        claims = attach_anchors(raw_claims, context, fallback_doc_id=fallback_doc_id)
        self.gate.vet_all(claims, context)
        summary = ClaimGate.summarize(claims)
        if trace is not None:
            trace.append(
                {
                    "step": "gate",
                    "detail": (
                        f"闸门逐句裁决 {summary['total']} 条主张："
                        f"放行 {summary['by_verdict']['allow']}、"
                        f"待核实 {summary['by_verdict']['downgrade']}、"
                        f"拦下 {summary['by_verdict']['block']}（裁判：{getattr(self.gate.judge, 'name', '?')}）"
                    ),
                    "ms": int((time.time() - started) * 1000),
                }
            )
        return claims

    # ==================================================================
    # 能力一：抗幻觉问答
    # ==================================================================
    def answer(
        self,
        question: str,
        *,
        doc_ids: Optional[list[str]] = None,
        mode: str = "auto",
        top_k: Optional[int] = None,
    ) -> Answer:
        trace: list[dict] = []
        evidences = self.retrieve(question, doc_ids=doc_ids, top_k=top_k, trace=trace)
        materials = self._material(evidences)

        if not materials:
            return Answer(
                question=question,
                text="文献库里没有找到与这个问题相关的材料，无法作答。请先上传相关论文，或换一种问法。",
                refused=True,
                refusal_reason="检索结果为空",
                trace=trace,
            )

        if not self.gateway.available:
            # 没有模型时不做生成，只给可核验的原句 —— 宁可少说，也不瞎说
            claims = [
                Claim(
                    text=f"《{m.doc_title}》第 {m.page} 页原文：" + m.text[:200],
                    kind="fact",
                    evidence=[
                        Evidence(
                            doc_id=m.doc_id,
                            anchor=evidences[m.index - 1].anchor,
                            score=evidences[m.index - 1].score,
                            doc_title=m.doc_title,
                            chunk_id=evidences[m.index - 1].chunk_id,
                        )
                    ],
                    claim_id=f"c{m.index}",
                )
                for m in materials
            ]
            context = self._gate_context(sorted({m.doc_id for m in materials}))
            self.gate.vet_all(claims, context)
            trace.append({"step": "answer", "detail": "未配置模型，改为直接返回可核验的原文片段", "ms": 0})
            return Answer(
                question=question,
                text=self.compose(claims),
                claims=claims,
                refused=not ClaimGate.collect(claims),
                refusal_reason="" if ClaimGate.collect(claims) else "没有可放行的证据片段",
                trace=trace,
            )

        started = time.time()
        blocks = [m.to_block() for m in materials]
        try:
            data = self.gateway.json(
                [
                    {"role": "system", "content": prompts.QA_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_qa_user(question, blocks, mode=mode),
                    },
                ],
                model=settings.model,
                max_tokens=2500,
                kind="rag.answer",
            )
        except GatewayError as exc:
            trace.append(
                {
                    "step": "answer",
                    "detail": f"生成失败（{exc.kind}），已转为只返回可核验原文",
                    "ms": int((time.time() - started) * 1000),
                }
            )
            return self._fallback_answer(question, evidences, materials, trace)

        raw_claims = data.get("claims") or []
        claims = self._vet(
            raw_claims,
            doc_ids=sorted({m.doc_id for m in materials}),
            fallback_doc_id=materials[0].doc_id,
            trace=trace,
        )
        trace.append(
            {
                "step": "answer",
                "detail": f"生成 {len(raw_claims)} 条候选主张，用时 {int((time.time() - started) * 1000)}ms",
                "ms": int((time.time() - started) * 1000),
            }
        )

        if data.get("refused") and not claims:
            return Answer(
                question=question,
                text="根据现有材料无法回答这个问题。"
                + str(data.get("refusal_reason") or ""),
                claims=[],
                refused=True,
                refusal_reason=str(data.get("refusal_reason") or "材料不足"),
                trace=trace,
            )

        allowed = ClaimGate.collect(claims)
        if not allowed:
            return Answer(
                question=question,
                text=(
                    "生成的内容没能通过溯源闸门，因此不予输出。"
                    f"被拦下的 {len(claims)} 条主张都缺少可核验的原文出处。"
                ),
                claims=claims,
                refused=True,
                refusal_reason="全部主张未通过溯源闸门",
                trace=trace,
            )

        return Answer(question=question, text=self.compose(claims), claims=claims, trace=trace)

    def _fallback_answer(
        self,
        question: str,
        evidences: list[Evidence],
        materials: list[Material],
        trace: list[dict],
    ) -> Answer:
        claims = [
            Claim(
                text=f"《{m.doc_title}》第 {m.page} 页原文：" + m.text[:200],
                kind="fact",
                evidence=[evidences[m.index - 1]],
                claim_id=f"c{m.index}",
            )
            for m in materials
        ]
        self.gate.vet_all(claims, self._gate_context(sorted({m.doc_id for m in materials})))
        return Answer(
            question=question,
            text=self.compose(claims),
            claims=claims,
            refused=not ClaimGate.collect(claims),
            refusal_reason="" if ClaimGate.collect(claims) else "没有可放行的证据片段",
            trace=trace,
        )

    # ------------------------------------------------------------------ 组装
    @staticmethod
    def compose(claims: list[Claim], *, include_downgrade: bool = True) -> str:
        """把通过闸门的主张拼成最终答复，并在末尾如实交代被拦下的部分。"""
        lines: list[str] = []
        for c in claims:
            if not c.decision:
                continue
            if c.decision.verdict is Verdict.ALLOW:
                lines.append(c.text)
            elif c.decision.verdict is Verdict.DOWNGRADE and include_downgrade:
                lines.append(c.text + "（口径待核实）")

        blocked = [c for c in claims if c.decision and c.decision.verdict is Verdict.BLOCK]
        if blocked:
            lines.append("")
            lines.append(
                f"另有 {len(blocked)} 条内容因无法溯源到原文而被闸门拦下，未纳入以上回答："
            )
            for c in blocked[:5]:
                reason = c.decision.reasons[0] if c.decision.reasons else "缺少可核验出处"
                lines.append(f"- 「{c.text[:60]}」——{reason}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 流式
    def stream(
        self,
        question: str,
        *,
        doc_ids: Optional[list[str]] = None,
        mode: str = "auto",
        top_k: Optional[int] = None,
    ) -> Iterator[tuple[str, dict]]:
        """产出 SSE 事件序列，供前端渲染「思考过程 + 逐句溯源」。"""
        trace: list[dict] = []
        yield "trace", {"step": "plan", "detail": "开始规划检索策略", "ms": 0}

        evidences = self.retrieve(question, doc_ids=doc_ids, top_k=top_k, trace=trace)
        for item in trace:
            yield "trace", item
        yield "evidence", {"evidence": [e.to_dict() for e in evidences]}

        if not evidences:
            yield "error", {"kind": "no_material", "message": "文献库里没有找到相关材料"}
            yield "done", {"answer_id": "", "support_rate": 0.0, "refused": True, "usage": {}}
            return

        answer = self.answer(question, doc_ids=doc_ids, mode=mode, top_k=top_k)
        for item in answer.trace:
            if item.get("step") in {"answer"}:
                yield "trace", item
        for claim in answer.claims:
            yield "claim", claim.to_dict()
        for sentence in split_sentences(answer.text) or [answer.text]:
            yield "token", {"text": sentence + ("\n" if not sentence.endswith("\n") else "")}
        yield "done", {
            "answer_id": f"a{int(time.time())}",
            "support_rate": answer.support_rate,
            "refused": answer.refused,
            "usage": self.gateway.ledger.summary(),
        }

    # ==================================================================
    # 能力二：细粒度抽取
    # ==================================================================
    def extract(
        self,
        fields: list[str],
        *,
        doc_id: str,
        question_hint: str = "",
        top_k: int = 12,
    ) -> dict:
        trace: list[dict] = []
        query = question_hint or " ".join(fields)
        evidences = self.retrieve(query, doc_ids=[doc_id], top_k=top_k, trace=trace)
        materials = self._material(evidences)
        if not materials:
            return {"items": [], "trace": trace}
        if not self.gateway.available:
            return {"items": [], "trace": trace, "error": "未配置模型，无法抽取"}

        try:
            data = self.gateway.json(
                [
                    {"role": "system", "content": prompts.EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_extract_user(fields, [m.to_block() for m in materials]),
                    },
                ],
                model=settings.model,
                max_tokens=1800,
                kind="extract",
            )
        except GatewayError as exc:
            return {"items": [], "trace": trace, "error": f"{exc.kind}: {exc}"}

        items: list[dict] = []
        raw_claims: list[dict] = []
        for i, item in enumerate(data.get("items") or []):
            quote = str(item.get("quote") or "").strip()
            field = str(item.get("field") or "").strip()
            value = item.get("value")
            if value is None or not quote:
                items.append(
                    {
                        "field": field,
                        "value": None,
                        "unit": item.get("unit") or "",
                        "evidence": [],
                        "decision": None,
                        "note": "材料中未找到该字段",
                    }
                )
                continue
            unit = str(item.get("unit") or "").strip()
            text = f"{field}：{value}{(' ' + unit) if unit else ''}"
            raw_claims.append(
                {"text": text, "kind": "number" if _looks_numeric(value) else "fact", "quotes": [quote], "doc_id": doc_id}
            )
            items.append({"field": field, "value": value, "unit": unit, "_idx": len(raw_claims) - 1})

        claims = self._vet(raw_claims, doc_ids=[doc_id], fallback_doc_id=doc_id, trace=trace)
        for item in items:
            idx = item.pop("_idx", None)
            if idx is None:
                continue
            claim = claims[idx]
            item["evidence"] = [e.to_dict() for e in claim.evidence]
            item["decision"] = claim.decision.to_dict() if claim.decision else None
            # 闸门没放行的字段，值要标为不可信，而不是悄悄给出去
            if claim.decision and claim.decision.verdict is Verdict.BLOCK:
                item["value"] = None
                item["note"] = "抽取结果未通过溯源闸门：" + (
                    claim.decision.reasons[0] if claim.decision.reasons else ""
                )
        return {"items": items, "trace": trace, "usage": self.gateway.ledger.summary()}

    # ==================================================================
    # 能力三：跨文献对比
    # ==================================================================
    def compare(
        self,
        aspects: list[str],
        *,
        doc_ids: list[str],
        per_aspect_k: int = 4,
    ) -> dict:
        trace: list[dict] = []
        evidences: list[Evidence] = []
        for aspect in aspects:
            evidences.extend(self.retrieve(aspect, doc_ids=doc_ids, top_k=per_aspect_k, trace=[]))
        evidences = dedupe_evidence(evidences)
        materials = self._material(evidences)
        if not materials or not self.gateway.available:
            return {"aspects": aspects, "table": [], "summary_claims": [], "trace": trace}

        try:
            data = self.gateway.json(
                [
                    {"role": "system", "content": prompts.COMPARE_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_compare_user(aspects, [m.to_block() for m in materials]),
                    },
                ],
                model=settings.model,
                max_tokens=3000,
                kind="compare",
            )
        except GatewayError as exc:
            return {"aspects": aspects, "table": [], "summary_claims": [], "trace": trace,
                    "error": f"{exc.kind}: {exc}"}

        doc_seq = sorted({m.doc_id for m in materials})
        title_by_index = {str(m.index): m.doc_title for m in materials}

        raw_claims: list[dict] = []
        table: list[dict] = []
        for row in data.get("table") or []:
            cells = []
            for cell in row.get("cells") or []:
                quote = str(cell.get("quote") or "").strip()
                value = cell.get("value")
                doc = str(cell.get("doc") or "")
                if not quote or value is None:
                    continue
                doc_id = doc_seq[int(doc) - 1] if doc.isdigit() and 1 <= int(doc) <= len(doc_seq) else ""
                raw_claims.append(
                    {
                        "text": f"{row.get('aspect','')}：{value}",
                        "kind": "comparison",
                        "quotes": [quote],
                        "doc_id": doc_id,
                    }
                )
                cells.append(
                    {
                        "doc_id": doc_id,
                        "doc_label": title_by_index.get(doc, doc),
                        "value": value,
                        "page": cell.get("page"),
                        "_idx": len(raw_claims) - 1,
                    }
                )
            table.append({"aspect": row.get("aspect", ""), "cells": cells})

        summary_raw = [
            {
                "text": s.get("text", ""),
                "kind": "comparison",
                "quotes": s.get("quotes") or [],
            }
            for s in (data.get("summary_claims") or [])
            if s.get("text")
        ]

        all_claims = self._vet(
            raw_claims + summary_raw, doc_ids=doc_seq, fallback_doc_id=doc_seq[0] if doc_seq else "", trace=trace
        )
        summary_claims = all_claims[len(raw_claims):]

        for row in table:
            for cell in row["cells"]:
                idx = cell.pop("_idx", None)
                if idx is None:
                    continue
                claim = all_claims[idx]
                cell["evidence"] = [e.to_dict() for e in claim.evidence]
                cell["decision"] = claim.decision.to_dict() if claim.decision else None
                if claim.decision and claim.decision.verdict is Verdict.BLOCK:
                    cell["value"] = None
                    cell["note"] = "未通过溯源闸门"

        return {
            "aspects": aspects,
            "table": table,
            "summary_claims": [c.to_dict() for c in summary_claims],
            "summary_text": self.compose(summary_claims),
            "trace": trace,
        }


def _looks_numeric(value: Any) -> bool:
    text = str(value)
    return any(ch.isdigit() for ch in text)
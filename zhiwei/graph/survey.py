"""综述生成：从图谱直接长出一篇「每句话都能点回原文」的领域综述。

和市面上「让模型写一段综述」最大的区别是：这里的综述是**图谱的下游产物**。
先有引用网络和基石节点判定，再按图上的重要性挑材料，最后让模型写、
由闸门逐句放行。所以读者点任何一句话，都能跳回某篇论文的某一页。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import settings
from ..contracts import Claim, Evidence, Verdict
from ..llm import prompts
from ..llm.client import Gateway, GatewayError, get_gateway
from ..reasoning.claim_gate import ClaimGate
from ..reasoning.engine import ScholarEngine
from ..reasoning.evidence import GateContext, attach_anchors
from .centrality import classify_roles

_ROLE_LABEL = {
    "cornerstone": "基石工作",
    "bridge": "桥接工作",
    "derivative": "边缘衍生",
    "peripheral": "外围工作",
    "isolated": "孤岛节点",
}

_ROLE_ORDER = {"cornerstone": 0, "bridge": 1, "derivative": 2, "peripheral": 3, "isolated": 4}


def build_materials(
    doc_ids: list[str],
    *,
    library: Any,
    insights: list,
    data_dir: Path | str,
    per_doc: int = 3,
    total: int = 24,
) -> list[dict]:
    """按图谱角色挑材料：基石工作优先，每篇取摘要 + 开篇正文。"""
    from .citation import load_parsed_doc

    data_dir = Path(data_dir or settings.data_dir)
    by_doc = {i.doc_id: i for i in insights}
    ordered = sorted(
        doc_ids,
        key=lambda d: (
            _ROLE_ORDER.get(getattr(by_doc.get(d), "role", "peripheral"), 9),
            -getattr(by_doc.get(d), "pagerank", 0.0),
        ),
    )

    materials: list[dict] = []
    index = 0
    for doc_id in ordered:
        parsed = load_parsed_doc(data_dir, doc_id)
        if parsed is None and library is not None:
            try:
                parsed = library.load_parsed(doc_id)
            except Exception:
                parsed = None
        title, year = doc_id, None
        try:
            paper = library.get_paper(doc_id)
            if paper is not None:
                title, year = paper.title or doc_id, paper.year
        except Exception:
            pass

        texts: list[tuple[int, str]] = []
        if parsed is not None:
            abstract = str((parsed.get("record") or {}).get("abstract") or "").strip()
            if abstract:
                texts.append((1, abstract))
            taken = 0
            for block in parsed.get("blocks") or []:
                if block.get("kind") != "paragraph":
                    continue
                text = str(block.get("text") or "").strip()
                if len(text) < 80:
                    continue
                texts.append((int(block.get("page") or 1), text))
                taken += 1
                if taken >= per_doc:
                    break

        role = getattr(by_doc.get(doc_id), "role", "peripheral")
        for page, text in texts[: per_doc + 1]:
            index += 1
            materials.append(
                {
                    "index": index,
                    "doc_id": doc_id,
                    "doc_title": title,
                    "year": year,
                    "page": page,
                    "text": text[:1200],
                    "role": _ROLE_LABEL.get(role, role),
                }
            )
            if len(materials) >= total:
                return materials
    return materials


def _markdown_table(rows: list[list[str]], header: list[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
    return "\n".join(out)


def generate_survey(
    topic: str,
    doc_ids: list[str],
    *,
    library: Any,
    insights: list,
    data_dir: Path | str,
    gateway: Optional[Gateway] = None,
    engine: Optional[ScholarEngine] = None,
    gate: Optional[ClaimGate] = None,
) -> dict:
    """生成领域综述。返回 `{"markdown", "claims", "timeline", "roles", "stats"}`。"""
    engine = engine or ScholarEngine(library=library, index=getattr(library, "index", None), gateway=gateway)
    gate = gate or engine.gate
    # 没显式给网关就沿用引擎的、再退回全局网关 —— 否则综述会静默变成空壳
    gateway = gateway or getattr(engine, "gateway", None) or get_gateway()
    materials = build_materials(doc_ids, library=library, insights=insights, data_dir=data_dir)

    title_by_doc = {m["doc_id"]: m["doc_title"] for m in materials}
    order: list[str] = []
    for m in materials:
        if m["doc_id"] not in order:
            order.append(m["doc_id"])
    cite_no = {doc_id: i + 1 for i, doc_id in enumerate(order)}

    timeline_rows = []
    for doc_id in order:
        ins = next((i for i in insights if i.doc_id == doc_id), None)
        timeline_rows.append(
            [
                str(getattr(ins, "year", "") or "?"),
                title_by_doc.get(doc_id, doc_id),
                _ROLE_LABEL.get(getattr(ins, "role", ""), getattr(ins, "role", "")),
            ]
        )
    timeline_rows.sort(key=lambda r: r[0])

    claims: list[Claim] = []
    if materials and gateway is not None and gateway.available:
        try:
            data = gateway.json(
                [
                    {"role": "system", "content": prompts.SURVEY_SYSTEM},
                    {"role": "user", "content": prompts.build_survey_user(topic, materials)},
                ],
                model=settings.model,
                max_tokens=2600,
                kind="survey",
            )
            raw = []
            for row in data.get("claims") or []:
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                from_doc = str(row.get("from_doc") or "").strip()
                doc_id = ""
                if from_doc.isdigit() and 1 <= int(from_doc) <= len(materials):
                    doc_id = materials[int(from_doc) - 1]["doc_id"]
                raw.append(
                    {
                        "text": text,
                        "kind": "summary",
                        "quotes": list(row.get("quotes") or []),
                        "doc_id": doc_id,
                    }
                )
            page_texts: dict[str, dict[int, str]] = {}
            titles: dict[str, str] = {}
            for doc_id in order:
                try:
                    page_texts[doc_id] = library.load_page_texts(doc_id) or {}
                except Exception:
                    page_texts[doc_id] = {}
                titles[doc_id] = title_by_doc.get(doc_id, doc_id)
            context = GateContext(page_texts=page_texts, doc_titles=titles)
            claims = attach_anchors(raw, context, fallback_doc_id=order[0] if order else "")
            gate.vet_all(claims, context)
        except (GatewayError, ValueError, TypeError) as exc:
            claims = []

    allowed = ClaimGate.collect(claims)
    downgraded = [c for c in claims if c.decision and c.decision.verdict is Verdict.DOWNGRADE]
    blocked = [c for c in claims if c.decision and c.decision.verdict is Verdict.BLOCK]

    body_lines: list[str] = []
    for claim in allowed:
        doc_id = claim.evidence[0].doc_id if claim.evidence else ""
        marker = f" [{cite_no[doc_id]}]" if doc_id in cite_no else ""
        body_lines.append(claim.text.rstrip("。.") + marker + "。")
    if downgraded:
        body_lines.append("")
        body_lines.append("**以下内容证据口径待核实：**")
        for claim in downgraded:
            doc_id = claim.evidence[0].doc_id if claim.evidence else ""
            marker = f" [{cite_no[doc_id]}]" if doc_id in cite_no else ""
            body_lines.append("- " + claim.text + marker + f"（{claim.decision.reasons[0] if claim.decision.reasons else ''}）")

    cornerstone = [i for i in insights if getattr(i, "role", "") == "cornerstone"]
    derivative = [i for i in insights if getattr(i, "role", "") == "derivative"]

    parts = [
        f"# {topic} · 领域综述（自动生成，逐句可溯源）",
        "",
        f"> 材料范围：{len(order)} 篇文献，{len(claims)} 条候选结论，"
        f"经溯源闸门放行 {len(allowed)} 条、待核实 {len(downgraded)} 条、拦下 {len(blocked)} 条。",
        "",
        "## 一、时间线",
        "",
        _markdown_table(timeline_rows, ["年份", "文献", "图谱角色"]),
        "",
        "## 二、演进脉络",
        "",
        "\n\n".join(body_lines) if body_lines else "（材料不足，未能形成可溯源的脉络陈述）",
        "",
        "## 三、基石工作",
        "",
    ]
    if cornerstone:
        for ins in sorted(cornerstone, key=lambda i: -i.pagerank):
            parts.append(
                f"- **{ins.title or ins.doc_id}**（{ins.year or '?'}）：{ins.reason}"
            )
    else:
        parts.append("（本次材料中未识别出达到基石阈值的节点）")

    parts += ["", "## 四、边缘衍生与新近工作", ""]
    if derivative:
        for ins in sorted(derivative, key=lambda i: -(i.year or 0)):
            parts.append(f"- {ins.title or ins.doc_id}（{ins.year or '?'}）：{ins.reason}")
    else:
        parts.append("（无）")

    parts += ["", "## 五、仍未解决", "", "见「前沿探索」页签：那里对每篇的 Future Work 做了时效判定。", ""]

    if blocked:
        parts += ["", "## 附：被溯源闸门拦下的内容（不计入综述）", ""]
        for claim in blocked[:8]:
            reason = claim.decision.reasons[0] if claim.decision.reasons else "缺少可核验出处"
            parts.append(f"- 「{claim.text[:80]}」——{reason}")

    return {
        "topic": topic,
        "markdown": "\n".join(parts),
        "claims": [c.to_dict() for c in claims],
        "timeline": [{"year": r[0], "title": r[1], "role": r[2]} for r in timeline_rows],
        "roles": classify_roles(insights),
        "stats": {
            "materials": len(materials),
            "documents": len(order),
            "allowed": len(allowed),
            "downgraded": len(downgraded),
            "blocked": len(blocked),
            "cornerstone": len(cornerstone),
        },
    }

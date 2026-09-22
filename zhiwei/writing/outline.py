"""写作框架生成：从 Idea 到 Abstract / Introduction / Related Work。

赛题写的是「真实无幻觉 References」。我们的做法比"让模型少编引用"更强一层：

  **参考文献不由模型写。** 模型只负责「在哪句话里引用库里的第几篇」，
  真正的引用串（作者、标题、年份、arXiv 号）由我们根据入库元数据**拼装**出来。
  库里没有的文献，物理上不可能出现在参考文献列表里 —— 因为它根本没有对应的元数据记录。
  这比事后校验更彻底：不是"检查出幻觉"，而是"没地方产生幻觉"。

正文里的每一句仍然要过溯源闸门，未通过的不进正文。
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import settings
from ..contracts import Claim, Evidence, Verdict
from ..llm import prompts
from ..llm.client import Gateway, GatewayError
from ..reasoning.claim_gate import ClaimGate
from ..reasoning.evidence import GateContext, attach_anchors


def format_reference(paper: Any) -> str:
    """按学术规范拼装引用串。完全由元数据决定，不经过模型。"""
    authors = list(getattr(paper, "authors", []) or [])
    if len(authors) > 6:
        author_str = ", ".join(authors[:6]) + ", et al."
    elif authors:
        author_str = ", ".join(authors)
    else:
        author_str = "Anonymous"

    title = (getattr(paper, "title", "") or "").strip().rstrip(".")
    venue = (getattr(paper, "venue", "") or "").strip()
    year = getattr(paper, "year", None)
    arxiv_id = (getattr(paper, "arxiv_id", "") or "").strip()
    version = getattr(paper, "version", 1) or 1

    parts = [f"{author_str}."]
    if title:
        parts.append(f"{title}.")
    if venue:
        parts.append(f"In {venue}.")
    if arxiv_id:
        parts.append(f"arXiv:{arxiv_id}v{version}.")
    if year:
        parts.append(f"{year}.")
    return " ".join(parts)


def build_outline(
    idea: str,
    *,
    doc_ids: list[str],
    library: Any,
    gateway: Optional[Gateway] = None,
    gate: Optional[ClaimGate] = None,
    engine: Any = None,
    venue: str = "generic",
) -> dict:
    """生成论文框架。返回 `{"abstract","introduction","related_work","references",...}`。"""
    from ..reasoning.engine import ScholarEngine

    engine = engine or ScholarEngine(library=library, index=getattr(library, "index", None), gateway=gateway)
    gateway = gateway or engine.gateway
    gate = gate or engine.gate

    # 1) 用 Idea 在库内检索材料
    evidences: list[Evidence] = []
    for doc_id in doc_ids or []:
        try:
            evidences.extend(engine.index.search(idea, doc_ids=[doc_id], top_k=4))
        except Exception:
            continue
    if not evidences:
        evidences = engine.retrieve(idea, doc_ids=doc_ids or None, top_k=10)
    evidences = evidences[:20]
    materials = engine._material(evidences)

    references: list[dict] = []
    cite_no: dict[str, int] = {}
    for doc_id in {m.doc_id for m in materials}:
        try:
            paper = library.get_paper(doc_id)
        except Exception:
            paper = None
        if paper is None:
            continue
        cite_no[doc_id] = len(references) + 1
        references.append(
            {
                "index": cite_no[doc_id],
                "doc_id": doc_id,
                "raw": format_reference(paper),
                "title": getattr(paper, "title", ""),
                "year": getattr(paper, "year", None),
                "source": "library",
            }
        )

    if not materials or not gateway.available:
        return {
            "idea": idea,
            "abstract": None,
            "introduction": [],
            "related_work": [],
            "references": references,
            "unsupported": [],
            "note": "材料不足或未配置模型，仅返回可由元数据确定的参考文献列表",
        }

    try:
        data = gateway.json(
            [
                {"role": "system", "content": prompts.OUTLINE_SYSTEM},
                {
                    "role": "user",
                    "content": prompts.build_outline_user(idea, [m.to_block() for m in materials]),
                },
            ],
            model=settings.model,
            max_tokens=3000,
            kind="writing.outline",
        )
    except GatewayError as exc:
        return {
            "idea": idea,
            "abstract": None,
            "introduction": [],
            "related_work": [],
            "references": references,
            "unsupported": [],
            "error": f"{exc.kind}: {exc}",
        }

    # 2) 把每个部分的主张转成带锚点的 Claim 并过闸
    page_texts: dict[str, dict[int, str]] = {}
    titles: dict[str, str] = {}
    for doc_id in cite_no:
        try:
            page_texts[doc_id] = library.load_page_texts(doc_id) or {}
        except Exception:
            page_texts[doc_id] = {}
        try:
            paper = library.get_paper(doc_id)
            titles[doc_id] = getattr(paper, "title", doc_id) if paper else doc_id
        except Exception:
            titles[doc_id] = doc_id
    context = GateContext(page_texts=page_texts, doc_titles=titles)

    def _to_claims(raw_items: list[dict]) -> list[Claim]:
        prepared = []
        for item in raw_items or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            prepared.append(
                {
                    "text": text,
                    "kind": "summary",
                    "quotes": list(item.get("quotes") or []),
                    "doc_id": (
                        next(iter(cite_no))
                        if not cite_no
                        else list(cite_no.keys())[0]
                    ),
                }
            )
        claims = attach_anchors(prepared, context, fallback_doc_id=next(iter(cite_no), ""))
        gate.vet_all(claims, context)
        return claims

    abstract_claims = _to_claims([data.get("abstract") or {}] if data.get("abstract") else [])
    intro_claims = _to_claims(data.get("introduction") or [])
    related_claims = _to_claims(data.get("related_work") or [])

    all_claims = abstract_claims + intro_claims + related_claims
    allowed = ClaimGate.collect(all_claims)
    blocked = [c for c in all_claims if c.decision and c.decision.verdict is Verdict.BLOCK]

    def _render(claims: list[Claim]) -> list[dict]:
        out = []
        for c in claims:
            doc_id = c.evidence[0].doc_id if c.evidence else ""
            out.append(
                {
                    "text": c.text,
                    "cite_index": cite_no.get(doc_id),
                    "decision": c.decision.to_dict() if c.decision else None,
                    "evidence": [e.to_dict() for e in c.evidence],
                }
            )
        return out

    # 3) 只把库内真实文献标为被引用；模型声称引用但库里没有的，一律剔除
    used = {
        c.evidence[0].doc_id for c in allowed if c.evidence and c.evidence[0].doc_id in cite_no
    }
    final_refs = [r for r in references if r["doc_id"] in used] or references[:1]

    return {
        "idea": idea,
        "venue": venue,
        "abstract": _render(ClaimGate.collect(abstract_claims)) or None,
        "introduction": _render(ClaimGate.collect(intro_claims)),
        "related_work": _render(ClaimGate.collect(related_claims)),
        "references": final_refs,
        "unsupported": [
            {"text": c.text, "reason": (c.decision.reasons[0] if c.decision and c.decision.reasons else "")}
            for c in blocked
        ],
        "stats": {
            "candidate_claims": len(all_claims),
            "allowed": len(allowed),
            "blocked": len(blocked),
            "references": len(final_refs),
        },
    }
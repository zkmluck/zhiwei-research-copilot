"""前沿探索：从 Future Work 里挑出「今天还值得做」的方向。

赛题原话：「提取多篇论文的 Future Work 部分，结合时间线与最新进展，
自动排除已被解决的旧 idea，提炼出真正具有探索价值的未来方向」。

实现分三步，每步都可单独失败而不影响其它步骤：

  1. 定位：在解析产物里找 Future Work / Conclusion / Discussion 章节；
  2. 抽取：把作者明确提出的未来方向抽成条目（带原文锚点，可回溯）；
  3. 时效判定：拿这条方向的关键词去 OpenAlex 找**该年之后**的相关工作，
     再用裁判判断它是否已经被解决。判不了就老实标 `unknown`，不硬给结论。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from ..config import settings
from ..contracts import Anchor, FutureWorkItem, normalize_text
from ..llm import prompts
from ..llm.client import Gateway, GatewayError

OPENALEX_WORKS = "https://api.openalex.org/works"

_FUTURE_SECTION_PAT = re.compile(
    r"(future\s*work|future\s*direction|conclusion|discussion|"
    r"未来工作|未来方向|结论与展望|讨论)",
    re.I,
)
_FUTURE_SENT_PAT = re.compile(
    r"(future work|in the future|we leave|left for future|remains? (an )?open|"
    r"future research|open problem|有待|未来工作|后续工作|尚待)",
    re.I,
)

_ROLE_ORDER = {"cornerstone": 0, "bridge": 1, "derivative": 2, "peripheral": 3, "isolated": 4}


# --------------------------------------------------------------------------
# 1. 定位 + 2. 抽取
# --------------------------------------------------------------------------


def find_future_sections(parsed: dict) -> list[dict]:
    """在章节树与块里找出候选的 Future Work 区域。"""
    hits: list[dict] = []
    for section in parsed.get("sections") or []:
        title = str(section.get("title") or "")
        if _FUTURE_SECTION_PAT.search(title):
            hits.append(
                {
                    "title": title,
                    "page": section.get("page", 0),
                    "path": [title],
                    "section_id": section.get("section_id", ""),
                }
            )
    if hits:
        return hits

    # 章节树没识别出来时，退一步：直接在块文本里找带 Future Work 的段落
    for block in parsed.get("blocks") or []:
        text = str(block.get("text") or "")
        if _FUTURE_SENT_PAT.search(text[:400]):
            hits.append(
                {
                    "title": "（未分级标题，按正文定位）",
                    "page": block.get("page", 0),
                    "path": block.get("section_path") or [],
                    "section_id": "",
                }
            )
    return hits[:6]


def collect_section_blocks(parsed: dict, section_titles: list[str], *, limit: int = 40) -> list[dict]:
    """取出这些章节下的正文块。"""
    wanted = {t.strip().lower() for t in section_titles if t}
    out: list[dict] = []
    for block in parsed.get("blocks") or []:
        if block.get("kind") not in {"paragraph", "heading", "caption"}:
            continue
        path = [str(p).strip().lower() for p in (block.get("section_path") or [])]
        if wanted and not any(p in wanted for p in path):
            continue
        text = str(block.get("text") or "").strip()
        if len(text) < 20:
            continue
        out.append(block)
        if len(out) >= limit:
            break
    return out


def extract_future_work(
    doc_id: str,
    parsed: dict,
    *,
    gateway: Optional[Gateway] = None,
    page_texts: Optional[dict[int, str]] = None,
    max_items: int = 6,
) -> list[FutureWorkItem]:
    """抽取一篇论文里的 Future Work 条目。"""
    sections = find_future_sections(parsed)
    if not sections:
        return []
    blocks = collect_section_blocks(parsed, [s["title"] for s in sections])
    if not blocks:
        return []

    items: list[FutureWorkItem] = []
    if gateway is not None and gateway.available:
        try:
            data = gateway.json(
                [
                    {"role": "system", "content": prompts.FUTURE_WORK_EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_future_work_extract_user(
                            [{"page": b.get("page"), "text": b.get("text")} for b in blocks[:20]]
                        ),
                    },
                ],
                model=settings.fast_model,
                max_tokens=1200,
                kind="future.extract",
            )
            for row in (data.get("items") or [])[:max_items]:
                text = str(row.get("text") or "").strip()
                quote = str(row.get("quote") or "").strip()
                if not text:
                    continue
                items.append(
                    FutureWorkItem(
                        doc_id=doc_id,
                        text=text,
                        anchor=_locate(quote, page_texts or {}),
                        status="unknown",
                        checked_at=_now(),
                    )
                )
        except (GatewayError, ValueError, TypeError):
            items = []

    if not items:
        # 规则兜底：抽含"未来/尚待"的句子
        for block in blocks:
            for sentence in re.split(r"(?<=[.。！？])\s+", str(block.get("text") or "")):
                if _FUTURE_SENT_PAT.search(sentence) and 30 <= len(sentence) <= 400:
                    items.append(
                        FutureWorkItem(
                            doc_id=doc_id,
                            text=normalize_text(sentence)[:300],
                            anchor=_locate(sentence, page_texts or {}, block.get("page", 0)),
                            status="unknown",
                            checked_at=_now(),
                        )
                    )
                if len(items) >= max_items:
                    break
            if len(items) >= max_items:
                break
    return items[:max_items]


def _locate(quote: str, page_texts: dict[int, str], page_hint: int = 0) -> Optional[Anchor]:
    if not quote:
        return None
    try:
        from ..store.index import locate_quote

        anchor = locate_quote(page_texts, quote, None)
        if anchor is not None:
            return anchor
    except Exception:
        pass
    try:
        from ..reasoning.evidence import _simple_locate

        return _simple_locate(page_texts, quote, page_hint)
    except Exception:
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 3. 时效判定
# --------------------------------------------------------------------------


@dataclass
class TimelinessResult:
    status: str
    confidence: float = 0.0
    evidence: str = ""
    resolved_by: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.resolved_by is None:
            self.resolved_by = []


def search_followups(
    query: str,
    *,
    since_year: Optional[int] = None,
    limit: int = 5,
    timeout: int = 25,
) -> list[dict]:
    """在 OpenAlex 里找该方向在指定年份之后的工作。"""
    params: dict[str, Any] = {
        "search": normalize_text(query)[:220],
        "per-page": str(limit),
        "sort": "cited_by_count:desc",
        "select": "id,title,publication_year,doi,cited_by_count,abstract_inverted_index",
        "mailto": settings.contact_email,
    }
    if since_year:
        params["filter"] = f"from_publication_date:{since_year + 1}-01-01"
    try:
        resp = requests.get(OPENALEX_WORKS, params=params, timeout=timeout,
                            headers={"User-Agent": f"zhiwei-research-copilot ({settings.contact_email})"})
        if resp.status_code != 200:
            return []
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    out: list[dict] = []
    for work in data.get("results") or []:
        out.append(
            {
                "title": work.get("title") or "",
                "year": work.get("publication_year"),
                "doi": work.get("doi") or "",
                "cited_by_count": work.get("cited_by_count") or 0,
                "abstract": _deinvert(work.get("abstract_inverted_index")),
            }
        )
    return out


def _deinvert(inverted: Optional[dict]) -> str:
    """OpenAlex 的摘要用倒排索引存，这里还原成文本。"""
    if not inverted:
        return ""
    slots: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions or []:
            slots.append((pos, word))
    slots.sort()
    return " ".join(w for _, w in slots)[:2000]


def judge_timeliness(
    item: FutureWorkItem,
    followups: list[dict],
    gateway: Optional[Gateway] = None,
) -> TimelinessResult:
    """判断一条 Future Work 是否还活着。"""
    if gateway is not None and gateway.available and followups:
        try:
            data = gateway.json(
                [
                    {"role": "system", "content": prompts.FUTURE_WORK_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_future_work_user(item.text, followups),
                    },
                ],
                model=settings.model,
                max_tokens=600,
                kind="future.judge",
            )
            status = str(data.get("status") or "unknown").strip().lower()
            if status not in {"resolved", "partially", "open", "stale"}:
                status = "unknown"
            return TimelinessResult(
                status=status,
                confidence=float(data.get("confidence") or 0.0),
                evidence=str(data.get("evidence") or "")[:300],
                resolved_by=[str(t) for t in (data.get("resolved_by") or [])][:5],
            )
        except (GatewayError, ValueError, TypeError):
            pass

    if not followups:
        return TimelinessResult("unknown", 0.0, "未检索到后续相关工作", [])
    return TimelinessResult(
        "unknown",
        0.0,
        f"检索到 {len(followups)} 篇后续工作，但未配置判定模型",
        [f.get("title", "") for f in followups[:3]],
    )


def scan_future_work(
    doc_ids: list[str],
    *,
    library: Any = None,
    data_dir: Optional[Path | str] = None,
    gateway: Optional[Gateway] = None,
    max_items_per_doc: int = 4,
    check_timeliness: bool = True,
) -> dict:
    """对一批文献做前沿扫描。返回可直接给前端/接口的结构。"""
    from .citation import load_doc_pages, load_parsed_doc

    data_dir = Path(data_dir or settings.data_dir)
    items: list[dict] = []

    for doc_id in doc_ids:
        parsed = load_parsed_doc(data_dir, doc_id)
        if parsed is None and library is not None:
            try:
                parsed = library.load_parsed(doc_id)
            except Exception:
                parsed = None
        if parsed is None:
            continue

        pages = load_doc_pages(data_dir, doc_id)
        if not pages and library is not None:
            try:
                pages = library.load_page_texts(doc_id) or {}
            except Exception:
                pages = {}

        year = None
        title = doc_id
        if library is not None:
            try:
                paper = library.get_paper(doc_id)
                if paper is not None:
                    year = paper.year
                    title = paper.title or doc_id
            except Exception:
                pass

        for item in extract_future_work(
            doc_id, parsed, gateway=gateway, page_texts=pages, max_items=max_items_per_doc
        ):
            record = item.to_dict()
            record["doc_title"] = title
            record["doc_year"] = year
            if check_timeliness:
                followups = search_followups(item.text, since_year=year)
                verdict = judge_timeliness(item, followups, gateway)
                item.status = verdict.status
                record["status"] = verdict.status
                record["confidence"] = round(verdict.confidence, 4)
                record["evidence"] = verdict.evidence
                record["resolved_by"] = verdict.resolved_by
                record["followup_count"] = len(followups)
            items.append(record)

    open_items = [i for i in items if i.get("status") == "open"]
    resolved = [i for i in items if i.get("status") == "resolved"]
    return {
        "items": items,
        "stats": {
            "total": len(items),
            "open": len(open_items),
            "resolved": len(resolved),
            "partially": sum(1 for i in items if i.get("status") == "partially"),
            "unknown": sum(1 for i in items if i.get("status") in {"unknown", None}),
        },
    }


def export_json(payload: dict, path: Path | str) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
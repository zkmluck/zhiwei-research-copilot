"""API 背后的编排逻辑：入库、去重决策、页面读取、引用跳转、翻译、图谱缓存。

这一层不碰 HTTP，只做「把子模块串起来」的事，方便单独写测试。
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..config import settings
from ..contracts import DocStatus, DedupRelation, PaperRecord, normalize_text
from ..ingest import dedup as dedup_mod
from ..ingest import pdf_parser
from ..ingest.version_diff import diff_versions
from ..store.index import locate_quote
from . import deps

_SAFE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")


def safe_name(name: str) -> str:
    name = Path(name or "upload.pdf").name
    cleaned = _SAFE.sub("_", name).strip("._") or "upload.pdf"
    return cleaned[:120]


def uploads_dir() -> Path:
    return settings.sub("uploads")


def source_path(doc_id: str) -> Path:
    return settings.data_dir / doc_id / "source.pdf"


# --------------------------------------------------------------------------
# 入库
# --------------------------------------------------------------------------


def stage_upload(filename: str, data: bytes) -> Path:
    """把上传的字节落到 uploads/ 下，返回临时路径。"""
    target = uploads_dir() / f"{int(time.time() * 1000)}_{safe_name(filename)}"
    target.write_bytes(data)
    return target


def ingest_path(
    path: Path | str,
    *,
    file_name: str = "",
    force_ocr: bool = False,
) -> dict:
    """解析 → 元数据 → 语义去重 → 落库。返回与契约一致的结果字典。"""
    path = Path(path)
    parsed = pdf_parser.parse_pdf(path, force_ocr=force_ocr)
    record = parsed.record
    if file_name:
        record.file_name = safe_name(file_name)
    doc_id = record.doc_id

    # 保留原始 PDF：阅读页要按页渲染，PDF.js 需要一个稳定的字节流
    target = source_path(doc_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(path, target)

    library = deps.library()
    existing = [p for p in library.list_papers(include_superseded=True) if p.doc_id != doc_id]
    verdict = dedup_mod.check_duplicate(record, parsed, existing=existing)
    relation = verdict.relation

    if relation in (DedupRelation.SAME_VERSION, DedupRelation.NEAR_DUPLICATE):
        status = "duplicate"
        record.status = DocStatus.DUPLICATE
        record.is_latest = False
    elif relation in (DedupRelation.OLDER_VERSION, DedupRelation.NEWER_VERSION):
        status = "needs_decision"
        record.status = DocStatus.ACTIVE
    else:
        status = "created"
        record.status = DocStatus.ACTIVE

    library.upsert_paper(record)
    page_texts = pdf_parser.load_page_texts(settings.data_dir, doc_id)
    if page_texts:
        library.save_page_texts(doc_id, page_texts)

    if status == "created":
        deps.index().index_document(parsed)

    if relation in (DedupRelation.OLDER_VERSION, DedupRelation.NEWER_VERSION) and verdict.existing_doc_id:
        _attach_version_diff(verdict, doc_id)

    return {
        "doc_id": doc_id,
        "file_name": record.file_name,
        "status": status,
        "dedup": verdict.to_dict(),
        "record": record.to_dict(),
    }


def _attach_version_diff(verdict: Any, incoming_doc_id: str) -> None:
    """两个版本都在库里时顺手把差异算出来，省掉前端一次往返。"""
    old_id = verdict.existing_doc_id
    try:
        old = pdf_parser.load_parsed(settings.data_dir, old_id)
        new = pdf_parser.load_parsed(settings.data_dir, incoming_doc_id)
        if not old or not new:
            return
        # 让 old 永远是较早的那一版
        if verdict.relation is DedupRelation.OLDER_VERSION:
            old, new = new, old
        verdict.version_diff = diff_versions(old, new)
    except Exception as exc:  # pragma: no cover - 差异算不出来不该阻塞入库
        verdict.version_diff = {"summary": f"版本差异暂时算不出来：{exc}", "changes": []}


def resolve_dedup(doc_id: str, action: str) -> dict:
    """处理版本/重复冲突。action: keep_existing | replace | keep_both。"""
    library = deps.library()
    record = library.get_paper(doc_id)
    if record is None:
        raise KeyError(doc_id)

    parsed = pdf_parser.load_parsed(settings.data_dir, doc_id)
    family = library.find_by_family(record.family_id) if record.family_id else []
    siblings = [p for p in family if p.doc_id != doc_id]
    older = [p for p in siblings if (p.version or 0) <= (record.version or 0)]

    if action == "keep_existing":
        library.delete_paper(doc_id)
        deps.index().delete_document(doc_id)
        keep = older[0].doc_id if older else ""
        return {"record": (older[0].to_dict() if older else record.to_dict()), "status": "kept_existing", "kept": keep}

    if action == "replace":
        for paper in siblings:
            library.set_status(paper.doc_id, DocStatus.SUPERSEDED)
            library.set_latest(paper.doc_id, False)
        library.set_status(doc_id, DocStatus.ACTIVE)
        library.set_latest(doc_id, True)
    else:  # keep_both
        library.set_status(doc_id, DocStatus.ACTIVE)
        library.set_latest(doc_id, True)

    if parsed:
        deps.index().index_document(parsed)
    return {"record": library.get_paper(doc_id).to_dict(), "status": action}


def delete_paper(doc_id: str) -> bool:
    library = deps.library()
    if library.get_paper(doc_id) is None:
        return False
    deps.index().delete_document(doc_id)
    library.delete_paper(doc_id)
    return True


def _record_of(doc_id: str) -> Optional[PaperRecord]:
    return deps.library().get_paper(doc_id)


def library_view() -> list[dict]:
    return [p.to_dict() for p in deps.library().list_papers(include_superseded=True)]


# --------------------------------------------------------------------------
# 阅读视图
# --------------------------------------------------------------------------


def outline_tree(sections: list[dict]) -> list[dict]:
    """把扁平的章节列表还原成层级树（前端左侧目录直接渲染）。"""
    nodes: dict[str, dict] = {}
    roots: list[dict] = []
    for raw in sections or []:
        node = {
            "section_id": raw.get("section_id") or "",
            "title": raw.get("title") or "",
            "number": raw.get("number") or "",
            "level": int(raw.get("level") or 1),
            "page": int(raw.get("page") or 0),
            "children": [],
        }
        nodes[node["section_id"]] = node
        parent = raw.get("parent")
        if parent and parent in nodes:
            nodes[parent]["children"].append(node)
        else:
            roots.append(node)
    return roots


def paper_detail(doc_id: str) -> Optional[dict]:
    record = _record_of(doc_id)
    if record is None:
        return None
    parsed = pdf_parser.load_parsed(settings.data_dir, doc_id) or {}
    sections = parsed.get("sections") or []
    return {
        "record": record.to_dict(),
        "sections": sections,
        "outline": outline_tree(sections),
        "stats": {
            "pages": record.page_count,
            "chunks": len(parsed.get("chunks") or []),
            "tables": int(parsed.get("tables") or 0),
            "figures": int(parsed.get("figures") or 0),
            "formulas": int(parsed.get("formulas") or 0),
            "references": len(parsed.get("references") or []),
        },
        "warnings": parsed.get("warnings") or [],
    }


_page_size_cache: dict[str, tuple[float, float, int]] = {}


def page_size(doc_id: str, page: int) -> tuple[float, float]:
    """页面尺寸（PDF 点）。供前端按比例摆放高亮框。"""
    src = source_path(doc_id)
    if not src.exists():
        return 612.0, 792.0
    cached = _page_size_cache.get(doc_id)
    if cached and page <= cached[2]:
        try:
            import pymupdf

            with pymupdf.open(str(src)) as doc:
                rect = doc[page - 1].rect
                _page_size_cache[doc_id] = (rect.width, rect.height, doc.page_count)
                return rect.width, rect.height
        except Exception:
            pass
    try:
        import pymupdf

        with pymupdf.open(str(src)) as doc:
            rect = doc[page - 1].rect
            _page_size_cache[doc_id] = (rect.width, rect.height, doc.page_count)
            return rect.width, rect.height
    except Exception:
        return 612.0, 792.0


def page_view(doc_id: str, page: int) -> Optional[dict]:
    record = _record_of(doc_id)
    if record is None:
        return None
    parsed = pdf_parser.load_parsed(settings.data_dir, doc_id) or {}
    blocks = [b for b in (parsed.get("blocks") or []) if int(b.get("page") or 0) == page]
    page_texts = pdf_parser.load_page_texts(settings.data_dir, doc_id)
    width, height = page_size(doc_id, page)
    return {
        "page": page,
        "width": width,
        "height": height,
        "blocks": [
            {
                "kind": b.get("kind") or "paragraph",
                "text": b.get("text") or "",
                "bbox": b.get("bbox"),
                "section_path": b.get("section_path") or [],
                "index": i,
                "extra": b.get("extra") or {},
            }
            for i, b in enumerate(blocks)
        ],
        "text": page_texts.get(page, ""),
    }


def references_view(doc_id: str) -> Optional[dict]:
    if _record_of(doc_id) is None:
        return None
    from ..graph.citation import parse_reference_string

    parsed = pdf_parser.load_parsed(settings.data_dir, doc_id) or {}
    raw_refs = parsed.get("references") or []
    page_texts = pdf_parser.load_page_texts(settings.data_dir, doc_id)

    # 库内标题索引：用于判断这条引文对应的论文是否已经在库里
    from ..ingest.dedup import normalize_title

    in_library = {
        normalize_title(p.title or ""): p.doc_id
        for p in deps.library().list_papers(include_superseded=True)
        if p.title
    }

    out: list[dict] = []
    for i, raw in enumerate(raw_refs):
        entry = parse_reference_string(str(raw), index=i + 1)
        title = entry.get("title_guess") or ""
        matched = ""
        if title:
            key = normalize_title(title)
            if key in in_library:
                matched = in_library[key]
            else:
                for lib_title, lib_doc in in_library.items():
                    if lib_title and (lib_title in normalize_title(entry["raw"]) or normalize_title(entry["raw"]) in lib_title):
                        matched = lib_doc
                        break
        anchor = locate_quote(page_texts, entry["raw"][:180]) if entry.get("raw") else None
        out.append(
            {
                "index": entry["index"],
                "raw": entry["raw"],
                "doc_id": matched,
                "in_library": bool(matched),
                "anchor": anchor.to_dict() if anchor else None,
                "resolved_title": title,
                "year": entry.get("year"),
                "url": _url_of(entry),
                "arxiv_id": entry.get("arxiv_id") or "",
            }
        )
    return {"references": out}


def _url_of(entry: dict) -> str:
    if entry.get("arxiv_id"):
        return f"https://arxiv.org/abs/{entry['arxiv_id']}"
    if entry.get("doi"):
        return f"https://doi.org/{entry['doi']}"
    return ""


def xref_view(doc_id: str, index: int) -> Optional[dict]:
    if _record_of(doc_id) is None:
        return None
    from ..graph.citation import extract_in_text_citations

    page_texts = pdf_parser.load_page_texts(settings.data_dir, doc_id)
    hits = [h for h in extract_in_text_citations(page_texts, max_contexts=400) if h["ref_index"] == index]
    anchors = []
    for hit in hits:
        anchor = locate_quote({hit["page"]: page_texts.get(hit["page"], "")}, hit.get("mark") or "")
        anchors.append(
            {
                "page": int(hit["page"]),
                "quote": hit["context"][:400],
                "mark": hit.get("mark") or "",
                "char_start": anchor.char_start if anchor else -1,
                "char_end": anchor.char_end if anchor else -1,
                "bbox": anchor.bbox.to_dict() if (anchor and anchor.bbox) else None,
                "block_kind": "text",
            }
        )
    refs = references_view(doc_id) or {"references": []}
    reference = next((r for r in refs["references"] if r["index"] == index), None)
    return {"in_text": anchors, "reference": reference}


# --------------------------------------------------------------------------
# 翻译
# --------------------------------------------------------------------------


def translate_page(doc_id: str, page: int, to: str = "zh") -> Optional[dict]:
    record = _record_of(doc_id)
    if record is None:
        return None
    from ..writing.translate import translate_document

    parsed = pdf_parser.load_parsed(settings.data_dir, doc_id) or {}
    blocks = [b for b in (parsed.get("blocks") or []) if int(b.get("page") or 0) == page]
    segments = [
        {"index": i, "text": b.get("text") or "", "bbox": b.get("bbox"), "kind": b.get("kind")}
        for i, b in enumerate(blocks)
        if len(normalize_text(b.get("text") or "")) > 1
    ]
    result = translate_document(segments, to=to, gateway=deps.gateway())
    items = result.get("segments") or []
    markdown_parts = []
    for item in items:
        src = item.get("src") or ""
        dst = item.get("dst") or item.get("translated") or ""
        if dst:
            markdown_parts.append(f"> {src}\n\n{dst}")
    return {
        "page": page,
        "to": to,
        "segments": [
            {
                "index": item.get("index", i),
                "src": item.get("src") or "",
                "dst": item.get("dst") or item.get("translated") or "",
                "bbox": item.get("bbox"),
            }
            for i, item in enumerate(items)
        ],
        "markdown": "\n\n".join(markdown_parts),
        "warnings": result.get("warnings") or [],
    }


# --------------------------------------------------------------------------
# 图谱 / 综述缓存
# --------------------------------------------------------------------------

_graph_lock = threading.RLock()
_graph_cache: dict[tuple[str, ...], tuple[Any, list, dict]] = {}


def graph_key(doc_ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(d for d in doc_ids if d))


def build_graph(doc_ids: list[str], *, classify: bool = False) -> tuple[Any, list, dict]:
    from ..graph.centrality import analyze_network
    from ..graph.citation import build_network

    network = build_network(
        doc_ids,
        library=deps.library(),
        data_dir=settings.data_dir,
        gateway=deps.gateway(),
        classify=classify,
    )
    insights, metrics = analyze_network(network)
    with _graph_lock:
        _graph_cache[graph_key(doc_ids)] = (network, insights, metrics)
    return network, insights, metrics


def get_graph(doc_ids: list[str], *, classify: bool = False) -> tuple[Any, list, dict]:
    with _graph_lock:
        cached = _graph_cache.get(graph_key(doc_ids))
    if cached is not None:
        return cached
    return build_graph(doc_ids, classify=classify)


def invalidate_graph() -> None:
    with _graph_lock:
        _graph_cache.clear()

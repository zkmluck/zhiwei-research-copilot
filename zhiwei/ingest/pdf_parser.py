"""PDF 解析：双栏阅读顺序、标题树、公式、表格、图、参考文献、切块。

这一层的质量决定上层一切。如果正文顺序错了，检索到的"证据"就是拼接出来的
乱句，后面的溯源闸门会把大量正常回答误判为不可信 —— 所以我们在这里花了
最多的力气在**阅读顺序**上。

关于双栏：学术 PDF 的取文本顺序默认常常是"左右左右"交错，直接读出来是乱码级体验。
做法是把每个文本块按水平位置归入左栏 / 右栏 / 跨栏，
再用"纵向条带"重排：跨栏块是天然的条带分隔，条带内先左栏自上而下、再右栏自上而下。

关于长文档：赛题要求处理 541 页的教材，所以这里不能把所有页的中间结构都堆在内存里。
实现上分页窗口处理，并用 `pymupdf.TOOLS.store_shrink()` 周期性收缩文档级缓存，
只在内存里保留最终产物（块、切块、页文本）。
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pymupdf

from ..config import settings
from ..contracts import (
    BBox,
    Block,
    Chunk,
    DocStatus,
    PaperRecord,
    ParsedDocument,
    Section,
    normalize_text,
)

# 原生文本层最小字符数：低于这个数认为是扫描件
TEXT_LAYER_MIN_CHARS = 50
# 每处理多少页收缩一次文档缓存（长文档内存控制）
SHRINK_EVERY_PAGES = 20
# 切块目标长度（字符）
CHUNK_TARGET = 500
CHUNK_MIN = 120

_HEADING_NUMBER = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+\S")
_HEADING_NAMED = re.compile(
    r"^\s*(abstract|introduction|related work|background|conclusion[s]?|discussion|"
    r"references|bibliography|appendix|acknowledg(e)?ments?|future work)\b",
    re.I,
)
_MATH_CHARS = set("∑∫√≤≥≈∈∀∃∇∂±×÷∞αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
_MATH_TOKEN = re.compile(r"(_\{|\^\{|\\frac|\\sum|\\alpha|\\beta|\\begin\{equation|\\theta|\\mathbb)")
_REF_ENTRY = re.compile(r"(?m)^\s*\[(\d{1,3})\]\s+")
_FIGURE_CAPTION = re.compile(r"^\s*(Figure|Fig\.|Table)\s*\d+", re.I)
_TABLE_CAPTION = re.compile(r"^\s*Table\s*\d+", re.I)


def _new_doc_id(file_name: str) -> str:
    stem = Path(file_name).stem[:40]
    safe = re.sub(r"[^0-9A-Za-z._\u4e00-\u9fff-]+", "_", stem).strip("_") or "doc"
    return f"{safe}-{uuid.uuid4().hex[:6]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# 阅读顺序
# --------------------------------------------------------------------------


@dataclass
class _RawBlock:
    page: int
    text: str
    bbox: BBox
    size: float = 0.0
    bold: bool = False
    source: str = "text"

    @property
    def width(self) -> float:
        return self.bbox.x1 - self.bbox.x0

    @property
    def mid_x(self) -> float:
        return (self.bbox.x0 + self.bbox.x1) / 2.0


def detect_columns(blocks: list[_RawBlock], page_width: float) -> int:
    """判断这一页是单栏还是双栏。

    判据：左右两半各自都有足够多的"窄块"（宽度不足页宽 62%），
    并且它们的水平中点分别落在左右半区。跨栏块不计入投票。
    """
    if not blocks or page_width <= 0:
        return 1
    mid = page_width / 2.0
    narrow_limit = page_width * 0.62
    left = right = wide = 0
    for block in blocks:
        if block.width >= narrow_limit:
            wide += 1
            continue
        if block.mid_x < mid:
            left += 1
        else:
            right += 1
    if left >= 2 and right >= 2 and wide <= max(4, (left + right) // 2):
        return 2
    return 1


def reading_order(blocks: list[_RawBlock], page_width: float, columns: int) -> list[_RawBlock]:
    """按阅读顺序重排一页的块。

    单栏：直接按纵坐标。
    双栏：跨栏块把页面切成若干"条带"，条带内先左栏后右栏。
    """
    if not blocks:
        return []
    if columns == 1:
        return sorted(blocks, key=lambda b: (round(b.bbox.y0, 1), b.bbox.x0))

    mid = page_width / 2.0
    narrow_limit = page_width * 0.62

    def is_wide(block: _RawBlock) -> bool:
        return block.width >= narrow_limit

    ordered: list[_RawBlock] = []
    band: list[_RawBlock] = []

    def flush() -> None:
        if not band:
            return
        left = [b for b in band if b.mid_x < mid]
        right = [b for b in band if b.mid_x >= mid]
        ordered.extend(sorted(left, key=lambda b: (round(b.bbox.y0, 1), b.bbox.x0)))
        ordered.extend(sorted(right, key=lambda b: (round(b.bbox.y0, 1), b.bbox.x0)))
        band.clear()

    for block in sorted(blocks, key=lambda b: (round(b.bbox.y0, 1), b.bbox.x0)):
        if is_wide(block):
            flush()
            ordered.append(block)
        else:
            band.append(block)
    flush()
    return ordered


# --------------------------------------------------------------------------
# 标题树
# --------------------------------------------------------------------------


def body_font_size(page_blocks: list[_RawBlock]) -> float:
    """正文基准字号 = 出现字符数最多的字号。"""
    weights: dict[float, int] = {}
    for block in page_blocks:
        if block.size:
            weights[round(block.size, 1)] = weights.get(round(block.size, 1), 0) + len(block.text)
    if not weights:
        return 10.0
    return max(weights.items(), key=lambda kv: kv[1])[0]


def is_heading(block: _RawBlock, base_size: float) -> Optional[int]:
    """判断是不是标题，并给出层级（1/2/3）。不是标题返回 None。"""
    text = normalize_text(block.text)
    if not text or len(text) > 120:
        return None
    # arXiv 水印行、页码、表格数据行经常"字号大 + 短"，它们不是标题。
    # 判据：字母占比过低（纯数字/符号）一律不是标题。
    letters = sum(1 for c in text if c.isalpha())
    if letters < max(4, len(text) * 0.45):
        return None
    if text.lower().startswith("arxiv:") or text.lower().startswith("preprint"):
        return None

    named = _HEADING_NAMED.match(text)
    numbered = _HEADING_NUMBER.match(text)

    if named:
        # 摘要/参考文献这类固定标题通常是 1 级
        return 1 if block.size >= base_size * 0.98 else 2

    if numbered:
        depth = numbered.group(1).count(".") + 1
        # 编号 + 字号不小于正文，才算标题（防止正文里的 "1. " 枚举）
        if block.size >= base_size * 0.95:
            return min(depth, 3)

    if block.size >= base_size * 1.18 and len(text) <= 80:
        return 1 if block.size >= base_size * 1.45 else 2
    if block.bold and block.size >= base_size * 1.05 and len(text) <= 80:
        return 3
    return None


def _make_section_id(number: str, title: str, index: int) -> str:
    base = number or re.sub(r"\W+", "-", title.lower())[:30]
    return f"s{index}-{base}".strip("-")


# --------------------------------------------------------------------------
# 公式 / 图 / 表
# --------------------------------------------------------------------------


def looks_like_formula(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < 4 or len(stripped) > 400:
        return False
    if _MATH_TOKEN.search(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    math_hits = sum(1 for c in stripped if c in _MATH_CHARS)
    symbol_hits = sum(1 for c in stripped if c in "=+-*/^_(){}[]∑∫")
    ratio = (math_hits + symbol_hits) / max(1, len(stripped))
    words = len(re.findall(r"[A-Za-z]{3,}", stripped))
    # 符号密度高、自然语言词少 -> 公式
    return (ratio > 0.18 and words <= 6) or (symbol_hits >= 6 and words <= 4)


def _table_to_markdown(rows: list[list[Any]]) -> str:
    cleaned = [
        ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
        for row in rows
        if row
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def open_table_reader(path: Path | str) -> Any:
    """一次性打开 pdfplumber。

    注意：**不能每页开一次**。541 页的教材上，逐页打开会让解析时间从几十秒
    涨到几十分钟 —— 这是实测踩过的坑。
    """
    try:
        import pdfplumber

        return pdfplumber.open(str(path))
    except Exception:
        return None


def _extract_tables(reader: Any, page_no: int) -> list[Block]:
    """从已打开的 pdfplumber 里抽表格。抽不到就返回空 —— 表格是加分项，不能拖垮主流程。"""
    out: list[Block] = []
    if reader is None:
        return out
    try:
        target = reader.pages[page_no - 1]
        for table in target.extract_tables() or []:
            markdown = _table_to_markdown(table)
            if markdown:
                out.append(
                    Block(
                        kind="table",
                        page=page_no,
                        text=markdown,
                        extra={"rows": len(table)},
                    )
                )
    except Exception:
        return out
    return out


def _extract_figures(doc: Any, page: Any, page_no: int, figure_dir: Path) -> list[Block]:
    """把页面里的位图裁出来落盘，返回 figure 块。"""
    out: list[Block] = []
    try:
        images = page.get_images(full=True)
    except Exception:
        return out
    for idx, image in enumerate(images):
        xref = image[0]
        try:
            info = doc.extract_image(xref)
        except Exception:
            continue
        data = info.get("image")
        if not data or len(data) < 2048:      # 太小的多半是线条或噪声
            continue
        ext = info.get("ext", "png")
        name = f"page{page_no:03d}_img{idx}.{ext}"
        try:
            figure_dir.mkdir(parents=True, exist_ok=True)
            (figure_dir / name).write_bytes(data)
        except OSError:
            continue
        bbox = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                rect = rects[0]
                bbox = BBox(rect.x0, rect.y0, rect.x1, rect.y1)
        except Exception:
            bbox = None
        out.append(
            Block(
                kind="figure",
                page=page_no,
                text=f"<image {name}>",
                bbox=bbox,
                extra={"image_path": str(figure_dir / name), "width": info.get("width"),
                       "height": info.get("height")},
            )
        )
    return out


# --------------------------------------------------------------------------
# 参考文献
# --------------------------------------------------------------------------

_REF_HEADING = re.compile(r"^\s*(references|bibliography|参考文献)\s*$", re.I)


def find_reference_start(blocks: list[Block]) -> int:
    for i, block in enumerate(blocks):
        if block.kind == "heading" and _REF_HEADING.match(normalize_text(block.text)):
            return i
    return -1


def split_references(blocks: list[Block]) -> list[str]:
    """把参考文献区域切成条目。

    优先按 `[n]` 编号切；没有编号时退回"每条一个块"（arXiv 上多数是前者，
    少数期刊是作者-年份格式，那种情况下也至少能拿到条目）。
    """
    region = "\n".join(b.text for b in blocks if b.kind in {"reference", "paragraph", "text"})
    if not region.strip():
        return []
    if _REF_ENTRY.search(region):
        parts = _REF_ENTRY.split(region)
        entries: list[str] = []
        # split 后结构为 [前言, 编号, 内容, 编号, 内容, ...]
        for i in range(1, len(parts) - 1, 2):
            content = normalize_text(parts[i + 1])
            if len(content) > 12:
                entries.append(content)
        if entries:
            return entries
    return [normalize_text(b.text) for b in blocks if len(normalize_text(b.text)) > 20]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _block_from_dict(item: dict) -> Block:
    bbox = item.get("bbox")
    return Block(
        kind=item.get("kind", "paragraph"),
        page=int(item.get("page") or 0),
        text=item.get("text") or "",
        bbox=BBox(**bbox) if isinstance(bbox, dict) else None,
        section_path=list(item.get("section_path") or []),
        extra=dict(item.get("extra") or {}),
    )


def parse_pdf(
    path: Path | str,
    doc_id: Optional[str] = None,
    *,
    force_ocr: bool = False,
    extract_tables: bool = True,
    extract_figures: bool = True,
    cache: bool = True,
) -> ParsedDocument:
    """解析一份 PDF，返回结构化产物，并把缓存写到 `data/<doc_id>/`。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在：{path}")

    doc_id = doc_id or _new_doc_id(path.name)
    data_root = settings.data_dir / doc_id
    figure_dir = data_root / "figures"

    doc = pymupdf.open(str(path))
    table_reader = open_table_reader(path) if extract_tables else None
    warnings: list[str] = []
    blocks: list[Block] = []
    page_texts: dict[int, str] = {}
    sections: list[Section] = []
    tables = figures = formulas = 0
    needs_ocr_pages: list[int] = []

    try:
        page_count = doc.page_count
        section_counter = 0
        section_stack: list[Section] = []
        current_path: list[str] = []

        for page_no in range(1, page_count + 1):
            page = doc[page_no - 1]
            width = page.rect.width
            page_text = page.get_text() or ""
            page_texts[page_no] = page_text

            raw_blocks = _collect_raw_blocks(page, page_no)
            if len(normalize_text(page_text)) < TEXT_LAYER_MIN_CHARS:
                needs_ocr_pages.append(page_no)

            base_size = body_font_size(raw_blocks)
            columns = detect_columns(raw_blocks, width)
            ordered = reading_order(raw_blocks, width, columns)

            # 首页字号最大的那一块是论文标题，它不该被当成章节节点混进标题树
            title_raw = max(raw_blocks, key=lambda b: b.size) if (page_no == 1 and raw_blocks) else None

            table_blocks = _extract_tables(table_reader, page_no)
            table_texts = {(b.text or "")[:60] for b in table_blocks}
            figure_blocks = (
                _extract_figures(doc, page, page_no, figure_dir) if extract_figures else []
            )

            for raw in ordered:
                text = normalize_text(raw.text)
                if not text:
                    continue
                level = None if raw is title_raw else is_heading(raw, base_size)
                if level is not None:
                    section_counter += 1
                    number_match = _HEADING_NUMBER.match(text)
                    section = Section(
                        level=level,
                        title=text[:120],
                        page=page_no,
                        number=number_match.group(1) if number_match else "",
                        bbox=raw.bbox,
                        section_id=_make_section_id(
                            number_match.group(1) if number_match else "", text, section_counter
                        ),
                    )
                    while section_stack and section_stack[-1].level >= level:
                        section_stack.pop()
                    section.parent = section_stack[-1].section_id if section_stack else None
                    section_stack.append(section)
                    sections.append(section)
                    current_path = [s.title for s in section_stack]
                    blocks.append(
                        Block(
                            kind="heading",
                            page=page_no,
                            text=text,
                            bbox=raw.bbox,
                            section_path=list(current_path),
                        )
                    )
                    continue

                if text[:60] in table_texts:
                    continue  # 表格已由 pdfplumber 单独产出，避免重复

                kind = "paragraph"
                if looks_like_formula(text):
                    kind = "formula"
                    formulas += 1
                elif _TABLE_CAPTION.match(text):
                    kind = "caption"
                elif _FIGURE_CAPTION.match(text):
                    kind = "caption"
                elif _REF_HEADING.match(text):
                    kind = "reference"

                blocks.append(
                    Block(
                        kind=kind,
                        page=page_no,
                        text=text,
                        bbox=raw.bbox,
                        section_path=list(current_path),
                    )
                )

            for block in table_blocks:
                block.section_path = list(current_path)
                blocks.append(block)
                tables += 1
            for block in figure_blocks:
                block.section_path = list(current_path)
                blocks.append(block)
                figures += 1

            # 长文档内存控制：周期性收缩 PyMuPDF 内部缓存，并释放页对象引用
            if page_no % SHRINK_EVERY_PAGES == 0:
                try:
                    pymupdf.TOOLS.store_shrink(100)
                except Exception:
                    pass
            del page
    finally:
        doc.close()
        if table_reader is not None:
            try:
                table_reader.close()
            except Exception:
                pass

    if needs_ocr_pages:
        warnings.append(
            f"检测到 {len(needs_ocr_pages)} 页缺少原生文本层（可能是扫描件）："
            + ", ".join(str(p) for p in needs_ocr_pages[:12])
            + ("..." if len(needs_ocr_pages) > 12 else "")
        )

    references = _collect_references(blocks)
    if not references:
        warnings.append("未识别出参考文献区，引用图谱可能不完整")

    chunks = build_chunks(blocks, doc_id, page_texts)
    record = PaperRecord(
        doc_id=doc_id,
        page_count=page_count,
        file_name=path.name,
        sha256=_sha256(path),
        arxiv_id="",
        needs_ocr=bool(needs_ocr_pages),
        status=DocStatus.ACTIVE,
        url="",
    )
    # 元数据与内容指纹在后面补齐（避免循环依赖）
    from .metadata import extract_metadata

    record = extract_metadata(
        ParsedDocument(
            record=record,
            sections=sections,
            blocks=blocks,
            chunks=chunks,
            references=references,
            tables=tables,
            figures=figures,
            formulas=formulas,
            warnings=warnings,
        )
    )

    parsed = ParsedDocument(
        record=record,
        sections=sections,
        blocks=blocks,
        chunks=chunks,
        references=references,
        tables=tables,
        figures=figures,
        formulas=formulas,
        warnings=warnings,
    )

    if cache:
        _write_cache(data_root, parsed, page_texts)
    return parsed


def _collect_raw_blocks(page: Any, page_no: int) -> list[_RawBlock]:
    out: list[_RawBlock] = []
    try:
        payload = page.get_text("dict")
    except Exception:
        return out
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
        text = " ".join(s.get("text", "") for s in spans)
        if not normalize_text(text):
            continue
        sizes = [s.get("size", 0) for s in spans if s.get("size")]
        font_names = " ".join(s.get("font", "") for s in spans)
        x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
        out.append(
            _RawBlock(
                page=page_no,
                text=text,
                bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                size=max(sizes) if sizes else 0.0,
                bold=("bold" in font_names.lower()) or ("black" in font_names.lower()),
            )
        )
    return out


def _collect_references(blocks: list[Block]) -> list[str]:
    start = find_reference_start(blocks)
    if start < 0:
        # 没有显式标题时，找编号最密集的尾部区域
        tail = [b for b in blocks if b.page >= max((x.page for x in blocks), default=1) - 2]
        return split_references(tail) if tail else []
    return split_references(blocks[start + 1 :])


# --------------------------------------------------------------------------
# 切块
# --------------------------------------------------------------------------


def build_chunks(blocks: list[Block], doc_id: str, page_texts: dict[int, str]) -> list[Chunk]:
    """按章节边界聚合成检索切块。

    约束：不跨章节、不跨页（跨页合并会让 bbox 失去意义，前端高亮就废了）。
    """
    chunks: list[Chunk] = []
    buffer: list[Block] = []
    buffer_len = 0
    index = 0
    # 上一次"章节边界"（标题）出现时 chunks 的长度。标题前的那一块属于上一章节，
    # 边界之后新章节的碎片不能再回并到它身上，否则章节归属就串了。
    boundary_at = -1

    def flush() -> None:
        nonlocal buffer, buffer_len, index, boundary_at
        if not buffer:
            return
        # 只有标题、没有正文的"块"没有检索价值，直接丢掉
        if all(b.kind == "heading" for b in buffer):
            boundary_at = len(chunks)
            buffer, buffer_len = [], 0
            return
        text = "\n".join(b.text for b in buffer).strip()
        if len(text) < CHUNK_MIN and chunks and len(chunks) > boundary_at:
            # 太短就并入上一块（同一章节内），避免碎片化
            prev = chunks[-1]
            if prev.section_path == buffer[0].section_path and prev.page == buffer[0].page:
                prev.text = (prev.text + "\n" + text).strip()
                prev.char_end = prev.char_end + len(text) + 1
                buffer, buffer_len = [], 0
                return
        first = buffer[0]
        page_text = page_texts.get(first.page, "")
        char_start = page_text.find(first.text[:60]) if page_text else -1
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{index}",
                doc_id=doc_id,
                text=text,
                page=first.page,
                kind="text",
                section_path=list(first.section_path),
                char_start=max(0, char_start),
                char_end=max(0, char_start) + len(first.text) if char_start >= 0 else -1,
                bbox=first.bbox,
                tokens=len(text) // 3,
            )
        )
        index += 1
        buffer, buffer_len = [], 0

    last_key: Optional[tuple] = None
    for block in blocks:
        if block.kind in {"figure", "table"}:
            flush()
            page_text = page_texts.get(block.page, "")
            char_start = page_text.find(block.text[:60]) if page_text else -1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{index}",
                    doc_id=doc_id,
                    text=block.text,
                    page=block.page,
                    kind=block.kind,
                    section_path=list(block.section_path),
                    char_start=max(0, char_start),
                    char_end=max(0, char_start) + len(block.text) if char_start >= 0 else -1,
                    bbox=block.bbox,
                    tokens=len(block.text) // 3,
                )
            )
            index += 1
            last_key = None
            continue

        if block.kind not in {"paragraph", "heading", "formula", "caption", "reference"}:
            continue

        key = (block.page, tuple(block.section_path), block.kind == "heading")
        if last_key is not None and key != last_key:
            flush()
        last_key = key
        buffer.append(block)
        buffer_len += len(block.text)
        if buffer_len >= CHUNK_TARGET:
            flush()
    flush()
    return chunks


# --------------------------------------------------------------------------
# 缓存读写
# --------------------------------------------------------------------------


def _write_cache(data_root: Path, parsed: ParsedDocument, page_texts: dict[int, str]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        (data_root / "parsed.json").write_text(
            json.dumps(parsed.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        (data_root / "pages.json").write_text(
            json.dumps({str(k): v for k, v in page_texts.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def load_parsed(data_dir: Path | str, doc_id: str) -> Optional[dict]:
    """读解析缓存。读不到返回 None（调用方应据此提示"该文献尚未解析"）。"""
    path = Path(data_dir) / doc_id / "parsed.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_page_texts(data_dir: Path | str, doc_id: str) -> dict[int, str]:
    path = Path(data_dir) / doc_id / "pages.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, str] = {}
    for key, value in (raw or {}).items():
        try:
            out[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return out


def parsed_from_dict(data: dict) -> ParsedDocument:
    """把缓存字典还原成 ParsedDocument。"""
    record_data = data.get("record") or {}
    record = PaperRecord(
        doc_id=record_data.get("doc_id", ""),
        title=record_data.get("title", ""),
        authors=list(record_data.get("authors") or []),
        affiliations=list(record_data.get("affiliations") or []),
        abstract=record_data.get("abstract", ""),
        year=record_data.get("year"),
        venue=record_data.get("venue", ""),
        arxiv_id=record_data.get("arxiv_id", ""),
        version=int(record_data.get("version") or 1),
        doi=record_data.get("doi", ""),
        url=record_data.get("url", ""),
        page_count=int(record_data.get("page_count") or 0),
        file_name=record_data.get("file_name", ""),
        sha256=record_data.get("sha256", ""),
        content_hash=record_data.get("content_hash", ""),
        family_id=record_data.get("family_id", ""),
        is_latest=bool(record_data.get("is_latest", True)),
        status=DocStatus(record_data.get("status") or "active"),
        needs_ocr=bool(record_data.get("needs_ocr")),
        ingested_at=record_data.get("ingested_at", ""),
        extra=dict(record_data.get("extra") or {}),
    )
    sections = []
    for item in data.get("sections") or []:
        bbox = item.get("bbox")
        sections.append(
            Section(
                level=int(item.get("level") or 1),
                title=item.get("title", ""),
                page=int(item.get("page") or 0),
                number=item.get("number", ""),
                bbox=BBox(**bbox) if isinstance(bbox, dict) else None,
                parent=item.get("parent"),
                section_id=item.get("section_id", ""),
            )
        )
    chunks = []
    for item in data.get("chunks") or []:
        bbox = item.get("bbox")
        chunks.append(
            Chunk(
                chunk_id=item.get("chunk_id", ""),
                doc_id=item.get("doc_id", record.doc_id),
                text=item.get("text", ""),
                page=int(item.get("page") or 0),
                kind=item.get("kind", "text"),
                section_path=list(item.get("section_path") or []),
                char_start=int(item.get("char_start", -1)),
                char_end=int(item.get("char_end", -1)),
                bbox=BBox(**bbox) if isinstance(bbox, dict) else None,
                tokens=int(item.get("tokens") or 0),
            )
        )
    return ParsedDocument(
        record=record,
        sections=sections,
        blocks=[_block_from_dict(b) for b in (data.get("blocks") or [])],
        chunks=chunks,
        references=list(data.get("references") or []),
        tables=int(data.get("tables") or 0),
        figures=int(data.get("figures") or 0),
        formulas=int(data.get("formulas") or 0),
        warnings=list(data.get("warnings") or []),
    )


def parse_pdf_to_dict(path: Path | str, doc_id: Optional[str] = None, **kwargs) -> dict:
    return parse_pdf(path, doc_id, **kwargs).to_dict()

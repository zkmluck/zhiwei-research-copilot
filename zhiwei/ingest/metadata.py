"""元数据抽取：标题、作者、机构、摘要、年份、arXiv 号。

一条原则：**不要只靠文件名。** 文件名可以随便改，评审也会故意改名来试。
所以我们从"页面上真实印着什么"推断：标题通常是第一页字号最大的那一段，
作者紧随其后且含逗号或邮箱，机构行含 University / Institute / Research 等词。

模型只在前两步都失败时兜底，而且失败也不会让流程挂掉 —— 元数据缺失是遗憾，
不是错误，绝不能因此丢掉整篇文献。
"""

from __future__ import annotations

import re
from typing import Optional

from ..config import settings
from ..contracts import PaperRecord, ParsedDocument
from ..llm.client import GatewayError, get_gateway

_ARXIV_NEW = re.compile(r"(?:arXiv[:\s]*)?(\d{4}\.\d{4,5})(?:v(\d{1,2}))?", re.I)
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v(\d{1,2}))?", re.I)
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_AFFIL_HINTS = (
    "university", "institute", "institut", "college", "school", "laboratory",
    "lab", "research", "academy", "google", "microsoft", "openai", "meta",
    "deepmind", "univ.", "department", "faculty", "center", "centre",
)
_AUTHOR_BLOCK_HINT = re.compile(r"^[\w.\-'\u4e00-\u9fff]+(\s+[\w.\-'\u4e00-\u9fff]+){1,3}$")


def guess_arxiv_id(*sources: str) -> tuple[str, int]:
    """从任意文本里找 arXiv 号与版本号。返回 `(基础号, 版本)`。"""
    for text in sources:
        if not text:
            continue
        match = _ARXIV_NEW.search(text)
        if match:
            return match.group(1), int(match.group(2) or 1)
        match = _ARXIV_OLD.search(text)
        if match:
            return match.group(1), int(match.group(2) or 1)
    return "", 1


def _year_from_arxiv(arxiv_id: str) -> Optional[int]:
    """arXiv 号前两位是年份（如 1706 -> 2017）。"""
    if not arxiv_id or "." not in arxiv_id:
        return None
    prefix = arxiv_id.split(".")[0]
    if len(prefix) != 4 or not prefix.isdigit():
        return None
    year = int(prefix[:2])
    return 1900 + year if year >= 91 else 2000 + year


def _first_page_blocks(parsed: ParsedDocument) -> list:
    return [b for b in parsed.blocks if b.page == 1]


def _guess_title(parsed: ParsedDocument) -> str:
    candidates = []
    for block in _first_page_blocks(parsed):
        text = (block.text or "").strip()
        if not 8 <= len(text) <= 220:
            continue
        if block.kind == "heading":
            return text
        if _EMAIL.search(text) or _AFFIL_HINTS[0] in text.lower():
            continue
        bbox = block.bbox
        size_proxy = (bbox.x1 - bbox.x0) if bbox else 0
        candidates.append((size_proxy, len(text), text))
    if not candidates:
        return ""
    # 标题一般"宽度大、字数适中"，用它排序
    candidates.sort(key=lambda c: (-c[0], -c[1]))
    return candidates[0][2]


_NAME_RE = re.compile(r"^[A-Z][A-Za-z.\-']*(?:\s+[A-Z][A-Za-z.\-']*){1,2}$")
_TITLE_STOP = re.compile(r"^(abstract|introduction|provided|permission|copyright|©)", re.I)


def _strip_markers(text: str) -> str:
    """去掉姓名后的角标、星号与邮箱。"""
    text = _EMAIL.sub("", text)
    text = re.sub(r"[\u2217\u2020\u2021*†‡§¶\d]+", " ", text)
    return " ".join(text.split()).strip(" ,;")


def _split_affiliation(text: str) -> tuple[str, str]:
    """把「机构 + 姓名」拆开。返回 (去机构后的文本, 机构)。"""
    lowered = text.lower()
    positions = [lowered.find(hint) for hint in _AFFIL_HINTS if hint in lowered]
    if not positions:
        return text, ""
    cut = min(p for p in positions if p >= 0)
    return text[:cut].strip(" ,;"), text[cut:].strip(" ,;")


def _guess_authors(blocks: list, title: str = "") -> list[str]:
    """从第一页顶部抽作者。

    踩过的坑：这篇论文的作者是**一人一个文本块**（"Ashish Vaswani ∗"、
    "Google Brain avaswani@google.com" ……），既没有逗号也没有 and。
    所以不能靠"块里有没有逗号"来判断，而要看**块的形状**：
    在标题之后、摘要之前，短小的、候选像人名的块就是作者。
    """
    if not blocks:
        return []
    title_norm = normalize_title_local(title) if title else ""
    start = 0
    if title_norm:
        for i, block in enumerate(blocks[:8]):
            if normalize_title_local(block.text) == title_norm:
                start = i + 1
                break

    authors: list[str] = []
    for block in blocks[start : start + 18]:
        text = (block.text or "").strip()
        if not text:
            continue
        if _TITLE_STOP.match(text) or len(text) > 200:
            break          # 走到摘要或正文了，作者区结束
        name_part, _affil = _split_affiliation(text)
        cleaned = _strip_markers(name_part)
        if not cleaned:
            continue
        if any(hint in cleaned.lower() for hint in _AFFIL_HINTS):
            continue
        for piece in re.split(r",|\band\b|、|&", cleaned):
            name = _strip_markers(piece)
            if 4 <= len(name) <= 40 and _NAME_RE.match(name):
                if name not in authors:
                    authors.append(name)
        if len(authors) >= 20:
            break
    return authors[:24]


def normalize_title_local(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", (text or "").lower()).strip()


_BOILERPLATE = re.compile(r"(provided|permission|copyright|©|granted|attribution|workshop|conference on)", re.I)


def _guess_affiliations(blocks: list) -> list[str]:
    """抽机构。要挡住版权声明这类"含 Research 字样但其实是免责声明"的干扰。"""
    out: list[str] = []
    for block in blocks[:18]:
        text = (block.text or "").strip()
        if not text or len(text) > 120:
            continue
        if _BOILERPLATE.search(text):
            continue
        _name_part, affil = _split_affiliation(text)
        candidate = affil or text
        candidate = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "", candidate)
        candidate = _strip_markers(candidate)
        if not candidate or len(candidate) > 90:
            continue
        if not any(hint in candidate.lower() for hint in _AFFIL_HINTS):
            continue
        if candidate not in out:
            out.append(candidate[:90])
        if len(out) >= 6:
            break
    return out


def _guess_abstract(parsed: ParsedDocument) -> str:
    blocks = parsed.blocks
    for i, block in enumerate(blocks):
        if block.kind == "heading" and block.text.strip().lower().startswith("abstract"):
            pieces = []
            for follow in blocks[i + 1 : i + 6]:
                if follow.kind == "heading":
                    break
                pieces.append(follow.text)
            text = " ".join(pieces).strip()
            if text:
                return text[:3000]
    # 没有 Abstract 标题时，取第一页正文最长的段落
    first = [b for b in _first_page_blocks(parsed) if b.kind == "paragraph"]
    if first:
        return max((b.text for b in first), key=len)[:3000]
    return ""


def extract_metadata(parsed: ParsedDocument) -> PaperRecord:
    """补全 PaperRecord，并写入内容指纹。"""
    record = parsed.record
    first_blocks = _first_page_blocks(parsed)
    sample_text = " ".join((b.text or "") for b in parsed.blocks[:40])

    if not record.title:
        record.title = _guess_title(parsed)
    if not record.authors:
        record.authors = _guess_authors(first_blocks, record.title)
    if not record.affiliations:
        record.affiliations = _guess_affiliations(first_blocks)
    if not record.abstract:
        record.abstract = _guess_abstract(parsed)

    arxiv_id, version = guess_arxiv_id(sample_text, record.file_name, record.url)
    if not record.arxiv_id and arxiv_id:
        record.arxiv_id = arxiv_id
        record.version = version
    if not record.doi:
        match = _DOI.search(sample_text)
        if match:
            record.doi = match.group(0)

    if not record.year:
        # 优先用 arXiv 号推断的年份：纸面上常印着"最新版本上传时间"（例如 2023），
        # 那是版本时间而不是论文发表时间。做时间线时用后者才对。
        record.year = _year_from_arxiv(record.arxiv_id)
        if not record.year:
            years = [int(y) for y in _YEAR.findall(sample_text)]
            if years:
                record.year = min(years)

    if not record.title:
        # 最后一招：问模型。仍然失败就留空，绝不编造。
        record.title = _llm_title(parsed)

    from .dedup import content_fingerprint

    record.content_hash = content_fingerprint(parsed)
    if not record.family_id:
        record.family_id = record.arxiv_id or record.content_hash[:12] or record.doc_id
    return record


def _llm_title(parsed: ParsedDocument) -> str:
    gateway = get_gateway()
    if not gateway.available:
        return ""
    head = "\n".join(b.text for b in _first_page_blocks(parsed)[:8])[:1500]
    if not head.strip():
        return ""
    try:
        data = gateway.json(
            [
                {
                    "role": "system",
                    "content": "从这篇论文的第一页文本里找出论文标题，只输出 JSON："
                    "{\"title\": \"...\"}。找不到就返回空字符串。",
                },
                {"role": "user", "content": head},
            ],
            model=settings.fast_model,
            max_tokens=200,
            kind="metadata.title",
        )
        return str(data.get("title") or "").strip()[:220]
    except (GatewayError, ValueError, TypeError):
        return ""
"""外部文献源：arXiv / OpenAlex / Semantic Scholar。

统一成 `PaperCandidate`，这样上层（推荐、下钻入库）不用关心各家字段差异。
所有网络调用都设超时、都吞掉异常并返回空列表 —— 评审环境网络不一定通，
一个外部源挂了不能让整个 Agent 崩掉。
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from ..config import settings

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_WORKS = "https://api.openalex.org/works"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_ARXIV_ID_FROM_URL = re.compile(r"abs/([^/]+)$")
_VERSION_SUFFIX = re.compile(r"v(\d+)$")


@dataclass
class PaperCandidate:
    """统一候选文献。"""

    source: str                      # arxiv | openalex | semantic_scholar | local
    identifier: str                  # arXiv id / DOI / OpenAlex id
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    venue: str = ""
    citation_count: int = 0
    has_code: bool = False
    code_url: str = ""
    published: str = ""              # ISO 日期，用于时效排序
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def base_arxiv_id(self) -> str:
        if self.source != "arxiv":
            return ""
        core = _VERSION_SUFFIX.sub("", self.identifier)
        return core

    @property
    def version(self) -> int:
        m = _VERSION_SUFFIX.search(self.identifier or "")
        return int(m.group(1)) if m else 1

    def to_dict(self) -> dict:
        data = asdict(self)
        data["base_arxiv_id"] = self.base_arxiv_id
        data["version"] = self.version
        return data


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------


class ArxivClient:
    """arXiv Atom API 客户端。"""

    def __init__(self, timeout: int = 25, pause: float = 3.0) -> None:
        self.timeout = timeout
        self.pause = pause          # arXiv 要求客户端自觉限速
        self._last_call = 0.0

    def _throttle(self) -> None:
        gap = time.time() - self._last_call
        if gap < self.pause:
            time.sleep(self.pause - gap)
        self._last_call = time.time()

    def _get(self, params: dict) -> Optional[str]:
        self._throttle()
        try:
            resp = requests.get(
                ARXIV_API,
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": f"zhiwei-research-copilot ({settings.contact_email})"},
            )
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        return resp.text

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        sort: str = "relevance",     # relevance | submitted
        category: str = "",
    ) -> list[PaperCandidate]:
        search_query = f"all:{query}"
        if category:
            search_query = f"cat:{category} AND {search_query}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max(1, min(limit, 50)),
            "sortBy": "submittedDate" if sort == "submitted" else "relevance",
            "sortOrder": "descending",
        }
        xml_text = self._get(params)
        if not xml_text:
            return []
        return self._parse(xml_text)

    def fetch_by_id(self, arxiv_id: str) -> Optional[PaperCandidate]:
        xml_text = self._get({"id_list": arxiv_id, "max_results": 1})
        if not xml_text:
            return None
        items = self._parse(xml_text)
        return items[0] if items else None

    @staticmethod
    def _parse(xml_text: str) -> list[PaperCandidate]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        out: list[PaperCandidate] = []
        for entry in root.findall(f"{_ATOM}entry"):
            raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
            match = _ARXIV_ID_FROM_URL.search(raw_id)
            identifier = match.group(1) if match else raw_id
            title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
            published = (entry.findtext(f"{_ATOM}published") or "")[:10]
            authors = [
                (a.findtext(f"{_ATOM}name") or "").strip()
                for a in entry.findall(f"{_ATOM}author")
            ]
            pdf_url = ""
            for link in entry.findall(f"{_ATOM}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href") or ""
            primary = entry.find(f"{_ARXIV_NS}primary_category")
            venue = primary.get("term") if primary is not None else ""
            out.append(
                PaperCandidate(
                    source="arxiv",
                    identifier=identifier,
                    title=title,
                    authors=[a for a in authors if a],
                    year=int(published[:4]) if len(published) >= 4 else None,
                    abstract=abstract,
                    url=f"https://arxiv.org/abs/{identifier}",
                    pdf_url=pdf_url or f"https://arxiv.org/pdf/{identifier}",
                    venue=venue,
                    published=published,
                )
            )
        return out


def download_pdf(candidate: PaperCandidate, dest_dir: Path | str) -> Optional[Path]:
    """把候选文献的 PDF 下钻下载到本地。文件名带版本号，便于版本族谱。"""
    if not candidate.pdf_url:
        return None
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", candidate.identifier)[:80]
    dest = dest_dir / f"{safe}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    try:
        resp = requests.get(
            candidate.pdf_url,
            timeout=90,
            headers={"User-Agent": f"zhiwei-research-copilot ({settings.contact_email})"},
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
        return None
    dest.write_bytes(resp.content)
    return dest


# --------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------


class OpenAlexClient:
    def __init__(self, timeout: int = 25) -> None:
        self.timeout = timeout

    def _get(self, params: dict) -> Optional[dict]:
        params = dict(params)
        params["mailto"] = settings.contact_email
        try:
            resp = requests.get(
                OPENALEX_WORKS,
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": f"zhiwei-research-copilot ({settings.contact_email})"},
            )
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def search(self, query: str, *, limit: int = 20, since_year: Optional[int] = None) -> list[PaperCandidate]:
        params: dict[str, Any] = {
            "search": query,
            "per-page": str(max(1, min(limit, 50))),
            "sort": "relevance_score:desc",
            "select": "id,doi,title,publication_year,authorships,primary_location,"
                      "cited_by_count,abstract_inverted_index,open_access",
        }
        if since_year:
            params["filter"] = f"from_publication_date:{since_year}-01-01"
        data = self._get(params)
        if not data:
            return []
        out: list[PaperCandidate] = []
        for work in data.get("results") or []:
            doi = (work.get("doi") or "").replace("https://doi.org/", "")
            location = work.get("primary_location") or {}
            source = (location.get("source") or {}) if isinstance(location, dict) else {}
            out.append(
                PaperCandidate(
                    source="openalex",
                    identifier=doi or (work.get("id") or "").split("/")[-1],
                    title=work.get("title") or "",
                    authors=[
                        ((a.get("author") or {}).get("display_name") or "")
                        for a in (work.get("authorships") or [])
                    ][:20],
                    year=work.get("publication_year"),
                    abstract=deinvert(work.get("abstract_inverted_index")),
                    url=work.get("doi") or work.get("id") or "",
                    venue=(source or {}).get("display_name") or "",
                    citation_count=work.get("cited_by_count") or 0,
                    published=str(work.get("publication_year") or ""),
                )
            )
        return out


def deinvert(inverted: Optional[dict]) -> str:
    """OpenAlex 摘要以倒排索引存储，这里还原成文本。"""
    if not inverted:
        return ""
    slots: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for pos in positions or []:
            slots.append((pos, word))
    slots.sort()
    return " ".join(w for _, w in slots)[:3000]


# --------------------------------------------------------------------------
# Semantic Scholar（可选，有 key 更稳）
# --------------------------------------------------------------------------


def search_semantic_scholar(query: str, *, limit: int = 20) -> list[PaperCandidate]:
    headers = {"User-Agent": f"zhiwei-research-copilot ({settings.contact_email})"}
    if settings.semantic_scholar_key:
        headers["x-api-key"] = settings.semantic_scholar_key
    try:
        resp = requests.get(
            S2_SEARCH,
            params={
                "query": query,
                "limit": max(1, min(limit, 50)),
                "fields": "title,abstract,year,authors,externalIds,venue,citationCount,openAccessPdf,url",
            },
            timeout=25,
            headers=headers,
        )
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    out: list[PaperCandidate] = []
    for item in data.get("data") or []:
        ext = item.get("externalIds") or {}
        pdf = (item.get("openAccessPdf") or {}).get("url") or ""
        out.append(
            PaperCandidate(
                source="semantic_scholar",
                identifier=ext.get("ArXiv") or ext.get("DOI") or item.get("paperId") or "",
                title=item.get("title") or "",
                authors=[(a.get("name") or "") for a in (item.get("authors") or [])][:20],
                year=item.get("year"),
                abstract=item.get("abstract") or "",
                url=item.get("url") or "",
                pdf_url=pdf,
                venue=item.get("venue") or "",
                citation_count=item.get("citationCount") or 0,
                published=str(item.get("year") or ""),
            )
        )
    return out


def search_all(query: str, *, limit: int = 20, since_year: Optional[int] = None) -> list[PaperCandidate]:
    """多源检索并去重（按标题归一化）。arxiv 优先，信息更全。"""
    merged: dict[str, PaperCandidate] = {}
    for cand in ArxivClient().search(query, limit=limit):
        merged[_norm(cand.title) or cand.identifier] = cand
    for cand in OpenAlexClient().search(query, limit=limit, since_year=since_year):
        key = _norm(cand.title) or cand.identifier
        if key not in merged:
            merged[key] = cand
    if len(merged) < limit // 2:
        for cand in search_semantic_scholar(query, limit=limit):
            key = _norm(cand.title) or cand.identifier
            merged.setdefault(key, cand)
    return list(merged.values())


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
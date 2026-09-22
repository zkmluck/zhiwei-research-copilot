"""个性化推荐与 arXiv 动态追踪。

赛题对这一块的要求很具体，逐条对应实现：

  - 「结合用户本地文献库计算相关性」—— 用库内文献的向量中心 + 关键词双重相关性；
  - 「优先推荐附带开源 GitHub 代码的论文」—— 从摘要里抠仓库链接，再用 GitHub 搜索兜底，
    命中就给复现性加分，并把链接一并给出；
  - 「提供推荐理由」—— 每一分都要能说清楚为什么，不许给一个黑箱排序；
  - 「点赞/踩」—— 反馈写进档案库，下一轮推荐据此调整（最简可用的多用户反馈闭环）。

推荐结果永远是「候选 + 理由 + 分数构成」，而不是「模型说这个好」。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import requests

from ..config import settings
from ..llm.client import Gateway, GatewayError
from ..reasoning.evidence import latin_terms
from .sources import PaperCandidate, search_all

_GITHUB_IN_TEXT = re.compile(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
_GH_SEARCH = "https://api.github.com/search/repositories"

# 推荐分数的权重（可被评测集校准）
W_RELEVANCE = 0.50
W_RECENCY = 0.20
W_CODE = 0.20
W_IMPACT = 0.10


# --------------------------------------------------------------------------
# 复现性：有没有开源代码
# --------------------------------------------------------------------------


def extract_code_url(text: str) -> str:
    """从摘要/正文里抠 GitHub 链接。"""
    match = _GITHUB_IN_TEXT.search(text or "")
    return match.group(0).rstrip(".,);") if match else ""


def search_github_repo(title: str, *, timeout: int = 20) -> tuple[bool, str]:
    """按论文标题在 GitHub 上找实现仓库。没有 token 也能用（限速更严）。"""
    title = (title or "").strip()
    if len(title) < 8:
        return False, ""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"zhiwei-research-copilot ({settings.contact_email})",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    try:
        resp = requests.get(
            _GH_SEARCH,
            params={"q": f"{title} in:name,description,readme", "sort": "stars", "per_page": 3},
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException:
        return False, ""
    if resp.status_code != 200:
        return False, ""
    try:
        items = (resp.json() or {}).get("items") or []
    except ValueError:
        return False, ""
    for repo in items:
        full_name = repo.get("full_name") or ""
        if not full_name:
            continue
        # 只有确实和标题有词面关系才认，避免"搜到什么都算有代码"
        title_terms = latin_terms(title)
        repo_text = f"{full_name} {repo.get('description') or ''}".lower()
        overlap = sum(1 for t in title_terms if t in repo_text)
        if overlap >= 2 or (title_terms and overlap / len(title_terms) >= 0.35):
            return True, repo.get("html_url") or ""
    return False, ""


def attach_code_links(candidates: list[PaperCandidate], *, max_queries: int = 6) -> list[PaperCandidate]:
    """给候选补上「有没有开源实现」。先查摘要，再查 GitHub，最多查前 N 篇控制调用量。"""
    queried = 0
    for cand in candidates:
        url = extract_code_url(cand.abstract)
        if url:
            cand.has_code = True
            cand.code_url = url
            cand.reasons.append("摘要中直接给出了开源仓库")
            continue
        if queried < max_queries:
            found, repo = search_github_repo(cand.title)
            queried += 1
            if found:
                cand.has_code = True
                cand.code_url = repo
                cand.reasons.append("GitHub 上找到了对应实现，复现门槛低")
    return candidates


# --------------------------------------------------------------------------
# 相关性：和用户自己的文献库比
# --------------------------------------------------------------------------


class LibraryProfile:
    """用户文献库的"画像"：关键词集合 + 向量中心。"""

    def __init__(self, keywords: set[str], centroid: Optional[list[float]] = None,
                 topics: Optional[list[str]] = None) -> None:
        self.keywords = keywords
        self.centroid = centroid
        self.topics = topics or []

    @classmethod
    def build(cls, papers: list[Any], gateway: Optional[Gateway] = None, max_docs: int = 40) -> "LibraryProfile":
        keywords: set[str] = set()
        corpus: list[str] = []
        for paper in papers[:max_docs]:
            title = getattr(paper, "title", "") or ""
            abstract = getattr(paper, "abstract", "") or ""
            keywords |= latin_terms(title)
            keywords |= set(list(latin_terms(abstract))[:60])
            corpus.append(f"{title}\n{abstract}"[:1200])

        centroid: Optional[list[float]] = None
        if gateway is not None and gateway.available and corpus:
            try:
                vectors = gateway.embed(corpus[:20])
                if vectors:
                    dim = len(vectors[0])
                    centroid = [
                        sum(v[i] for v in vectors) / len(vectors) for i in range(dim)
                    ]
                    norm = math.sqrt(sum(x * x for x in centroid)) or 1.0
                    centroid = [x / norm for x in centroid]
            except (GatewayError, ValueError, TypeError):
                centroid = None
        return cls(keywords=keywords, centroid=centroid)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def score_candidate(
    cand: PaperCandidate,
    profile: LibraryProfile,
    *,
    gateway: Optional[Gateway] = None,
    now_year: Optional[int] = None,
) -> PaperCandidate:
    """给候选打分，并把每一分的来源写进 `reasons`。"""
    now_year = now_year or time.localtime().tm_year

    # 1) 相关性：关键词覆盖率（离线可用）
    cand_terms = latin_terms(f"{cand.title} {cand.abstract}")
    if cand_terms and profile.keywords:
        kw_overlap = len(cand_terms & profile.keywords) / max(1, len(cand_terms))
    else:
        kw_overlap = 0.0
    relevance = min(1.0, kw_overlap * 3.0)

    # 2) 语义相关性：与库中心的余弦（有向量时才用）
    if gateway is not None and gateway.available and profile.centroid:
        try:
            vec = gateway.embed([f"{cand.title}\n{cand.abstract}"[:1200]])[0]
            cosine = max(0.0, _cosine(vec, profile.centroid))
            relevance = max(relevance, min(1.0, cosine * 1.2))
            cand.reasons.append(f"与你的文献库语义相似度 {cosine:.2f}")
        except (GatewayError, ValueError, TypeError, IndexError):
            pass
    if relevance > 0.15:
        cand.reasons.append(f"与库内关键词重合度较高（{kw_overlap:.0%}）")

    # 3) 时效
    age = max(0, now_year - (cand.year or now_year))
    recency = math.exp(-age / 3.0)

    # 4) 复现性
    code = 1.0 if cand.has_code else 0.0

    # 5) 影响力（对数压缩，避免高被引碾压新工作）
    impact = min(1.0, math.log1p(cand.citation_count) / math.log1p(500))

    cand.score = round(
        W_RELEVANCE * relevance + W_RECENCY * recency + W_CODE * code + W_IMPACT * impact, 4
    )
    if cand.has_code:
        cand.reasons.append("附带开源代码，易复现")
    if cand.year and now_year - cand.year <= 1:
        cand.reasons.append(f"{cand.year} 年新发表，时效性好")
    if cand.citation_count >= 50:
        cand.reasons.append(f"已被引用 {cand.citation_count} 次")
    return cand


# --------------------------------------------------------------------------
# 推荐与追踪
# --------------------------------------------------------------------------


def recommend(
    library: Any,
    *,
    query: str = "",
    days: int = 30,
    limit: int = 10,
    gateway: Optional[Gateway] = None,
    use_github: bool = True,
) -> dict:
    """推荐下一批该读的论文。

    `query` 为空时，用库内文献标题关键词自动构造检索式（这就是"结合本地库"）。
    """
    try:
        papers = library.list_papers() if library is not None else []
    except Exception:
        papers = []

    profile = LibraryProfile.build(papers, gateway=gateway)
    if not query:
        topic_terms: list[str] = []
        for paper in papers[:5]:
            terms = sorted(latin_terms(getattr(paper, "title", "") or ""))
            if terms:
                topic_terms.append(" ".join(terms[:6]))
        query = " OR ".join(topic_terms[:2]) if topic_terms else "large language model agent"

    since_year = time.localtime().tm_year - max(1, days // 365 + 1)
    candidates = search_all(query, limit=max(limit * 3, 20), since_year=since_year)
    if not candidates:
        return {"query": query, "items": [], "profile": {"keywords": len(profile.keywords)},
                "note": "外部检索无返回（网络不可用或限流）"}

    if use_github:
        candidates = attach_code_links(candidates, max_queries=min(6, len(candidates)))

    scored = [score_candidate(c, profile, gateway=gateway) for c in candidates]
    scored.sort(key=lambda c: -c.score)

    feedback = {}
    try:
        feedback = library.feedback_stats() if library is not None else {}
    except Exception:
        feedback = {}
    disliked = set(feedback.get("disliked_ids") or [])
    liked = set(feedback.get("liked_ids") or [])

    items = []
    for cand in scored[: limit * 2]:
        record = cand.to_dict()
        key = cand.identifier or cand.title
        record["feedback"] = "like" if key in liked else ("dislike" if key in disliked else "")
        record["reasons"] = list(dict.fromkeys(cand.reasons))[:5]
        items.append(record)
    # 踩过的排到后面，赞过的适当提前
    items.sort(key=lambda r: (r["feedback"] == "dislike", -(r["score"] + (0.08 if r["feedback"] == "like" else 0))))
    return {
        "query": query,
        "items": items[:limit],
        "profile": {"keywords": len(profile.keywords), "has_vector": bool(profile.centroid)},
        "weights": {"relevance": W_RELEVANCE, "recency": W_RECENCY, "code": W_CODE, "impact": W_IMPACT},
    }


def run_tracking(
    library: Any,
    *,
    topics: Optional[list[str]] = None,
    days: int = 7,
    limit_per_topic: int = 8,
    gateway: Optional[Gateway] = None,
) -> dict:
    """跑一轮 arXiv 动态追踪（等价于定时任务里调一次）。"""
    from .sources import ArxivClient

    from ..store.db import Library  # 仅用于类型提示，失败也不影响

    topics = [t for t in (topics or []) if t.strip()]
    if not topics and library is not None:
        try:
            papers = library.list_papers()
        except Exception:
            papers = []
        topics = [
            " ".join(sorted(latin_terms(getattr(p, "title", "") or ""))[:6])
            for p in papers[:3]
        ]
        topics = [t for t in topics if t.strip()]

    client = ArxivClient()
    found: list[PaperCandidate] = []
    for topic in topics[:3]:
        found.extend(client.search(topic, limit=limit_per_topic, sort="submitted"))

    known_titles = set()
    if library is not None:
        try:
            known_titles = {
                re.sub(r"[^a-z0-9]+", " ", (p.title or "").lower()).strip()
                for p in library.list_papers()
            }
        except Exception:
            known_titles = set()

    fresh = [
        c for c in found
        if re.sub(r"[^a-z0-9]+", " ", (c.title or "").lower()).strip() not in known_titles
    ]
    return {
        "topics": topics,
        "scanned": len(found),
        "new_items": [c.to_dict() for c in fresh][:40],
        "new_count": len(fresh),
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
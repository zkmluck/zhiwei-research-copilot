"""引用网络构建与「引用含金量」判定。

赛题要求「自动剔除应付式引用」——这句话的技术含义是：**引用是有类型的**。
`[1]` 出现在「我们沿用 [1] 的双塔结构」和出现在「已有许多工作研究了该问题[1,2,3]」
里，学术价值完全不同。前者是实质继承，后者只是礼貌性罗列。

所以我们把每条引用当成一个有类型的对象：先解析参考文献条目，再把正文里的引用
上下文抽出来，最后判定意图（extends/uses/compares/criticizes/background/lists），
据此给出 0-1 的「引用含金量」分。图谱里边的粗细与颜色都由它决定，
并且可以按阈值把 `lists` 一类整体过滤掉。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import settings
from ..contracts import (
    SUBSTANTIVE_INTENTS,
    CitationEdge,
    CitationIntent,
)
from ..llm import prompts
from ..llm.client import Gateway, GatewayError
from ..reasoning.evidence import latin_terms

# --------------------------------------------------------------------------
# 参考文献解析
# --------------------------------------------------------------------------

_REF_INDEX_RE = re.compile(r"^\s*\[?(\d{1,3})\]?[.)]?\s+")
_ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.I)
_ARXIV_OLD_RE = re.compile(r"(?:abs/)([a-z\-]+/\d{7})", re.I)
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_VENUE_MARKERS = (
    "In Proceedings", "In Advances", "In NIPS", "In NeurIPS", "In ICML", "In ICLR",
    "In ACL", "In EMNLP", "In CVPR", "In ICCV", "In AAAI", "In IJCAI", "In SIGIR",
    "arXiv preprint", "arXiv:", "CoRR", "In the ", "In Conference", "In International",
    "In Annual", "In Workshop", "In Findings", "In Journal", "Proceedings of",
)


# 作者串里几乎不会出现的"标题高频词"。用来把「作者块」和「标题块」分开。
_TITLE_WORD_RE = re.compile(
    r"\b(of|for|with|using|via|toward|towards|from|based|learning|networks?|models?|"
    r"analysis|approach|method|attention|transformer|training|neural)\b",
    re.I,
)


def _looks_like_authors(chunk: str) -> bool:
    """判断一个句段是不是作者列表，而不是标题。

    参考文献格式上千种，与其追求完美切分，不如抓住作者串的三个稳定特征：
    逗号密、专有名词密、几乎不含标题里的功能词。
    """
    words = chunk.split()
    if not words or len(words) > 40:
        return False
    if _TITLE_WORD_RE.search(chunk):
        return False
    commas = chunk.count(",")
    caps = sum(1 for w in words if w[:1].isupper())
    ratio = caps / len(words)
    if commas >= 2 and ratio >= 0.4:
        return True
    if commas >= 1 and len(words) <= 8 and ratio >= 0.5:
        return True
    return bool(re.search(r"\band\b", chunk)) and ratio >= 0.6 and len(words) <= 30


def _looks_like_bare_name(chunk: str, has_longer_after: bool) -> bool:
    """单作者条目的作者串：一两个全大写的词，后面还跟着更像标题的句段。

    例：「Francois Chollet. Xception: Deep learning with ...」——作者块的逗号
    和功能词都太少，前面那条规则认不出来，只能靠"后面还有更长的候选"来反推。
    """
    if not has_longer_after:
        return False
    words = chunk.split()
    if not 1 <= len(words) <= 3:
        return False
    if ":" in chunk or "," in chunk:
        return False
    return all(w[:1].isupper() for w in words if w[:1].isalpha())


def parse_reference_string(raw: str, index: int = 0) -> dict:
    """把一条参考文献字符串拆成结构化字段。

    这里刻意不追求"完美的作者/标题切分"（参考文献格式上千种，追求完美是陷阱），
    只要够用：年份、arXiv 号、DOI、场馆，以及一个**能用于与库内文献比对的标题候选**。
    跟库内文献匹配时，我们会用「库内标题是否出现在这条引文里」来兜底，这比切分更稳。
    """
    text = " ".join((raw or "").split())
    entry: dict[str, Any] = {"index": index, "raw": text, "title_guess": "", "year": None,
                             "arxiv_id": "", "doi": "", "venue": ""}
    if not text:
        return entry

    body = _REF_INDEX_RE.sub("", text)

    m = _ARXIV_RE.search(body)
    if m:
        entry["arxiv_id"] = m.group(1)
    elif (m2 := _ARXIV_OLD_RE.search(body)):
        entry["arxiv_id"] = m2.group(1)

    if (d := _DOI_RE.search(body)):
        entry["doi"] = d.group(0)

    years = _YEAR_RE.findall(body)
    if years:
        all_years = re.findall(r"\b(?:19|20)\d{2}\b", body)
        entry["year"] = int(all_years[-1])

    cut = len(body)
    for marker in _VENUE_MARKERS:
        pos = body.find(marker)
        if 10 < pos < cut:
            cut = pos
            entry["venue"] = marker.strip("In ").strip()
    head = body[:cut]

    # 标题候选：先剥掉开头的作者块，再取第一个像标题的句段
    chunks = [c.strip(" .") for c in re.split(r"\.\s+|\.$", head) if c and c.strip(" .")]
    # 标题可能是两个词（Layer normalization），也可能是十几个词；
    # 只排除明显不是标题的碎片（页码、年份、单个单词）。
    ranked = [
        c
        for c in chunks
        if len(c.split()) >= 2 and any(len(w.strip(",;:")) >= 3 for w in c.split())
    ]
    best = ""
    for pos, chunk in enumerate(ranked):
        has_longer_after = any(len(c) > len(chunk) for c in ranked[pos + 1 :])
        if _looks_like_authors(chunk) or _looks_like_bare_name(chunk, has_longer_after):
            continue
        best = chunk
        break
    if not best:
        best = max(ranked, key=len) if ranked else ""
    entry["title_guess"] = best or head[:160]
    return entry


def extract_reference_entries(parsed: dict) -> list[dict]:
    """从一个 ParsedDocument 字典里取出参考文献条目。"""
    refs = parsed.get("references") or []
    out: list[dict] = []
    for i, raw in enumerate(refs):
        entry = parse_reference_string(str(raw), index=i + 1)
        if entry["raw"]:
            out.append(entry)
    return out


# --------------------------------------------------------------------------
# 正文引用上下文
# --------------------------------------------------------------------------

_CITE_MARK_RE = re.compile(r"\[(\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*)\]")


def _sentence_around(text: str, start: int, end: int, radius: int = 260) -> str:
    """取引用标记所在的那一句（前后扩一点，保留技术关系词）。"""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    window = text[left:right]
    # 向前找最近的句末
    cut = max(window.rfind(". ", 0, start - left), window.rfind("\n", 0, start - left))
    if cut >= 0:
        window = window[cut + 2 :]
    cut2 = window.find(". ", end - left)
    if cut2 >= 0:
        window = window[: cut2 + 1]
    return " ".join(window.split())


def extract_in_text_citations(page_texts: dict[int, str], max_contexts: int = 200) -> list[dict]:
    """扫出正文里的 `[n]` 引用及所在句子。

    返回 `[{"ref_index": n, "page": p, "context": "..."}]`。
    """
    found: list[dict] = []
    for page in sorted(page_texts):
        text = page_texts[page] or ""
        if not text:
            continue
        for match in _CITE_MARK_RE.finditer(text):
            context = _sentence_around(text, match.start(), match.end())
            if len(context) < 20:
                continue
            for piece in re.split(r"\s*[,–\-]\s*", match.group(1)):
                if not piece.strip().isdigit():
                    continue
                found.append(
                    {
                        "ref_index": int(piece),
                        "page": page,
                        "context": context[:600],
                        "mark": match.group(0),
                    }
                )
                if len(found) >= max_contexts:
                    return found
    return found


# --------------------------------------------------------------------------
# 引用意图判定
# --------------------------------------------------------------------------

_INTENT_WEIGHT = {
    CitationIntent.EXTENDS: 1.00,
    CitationIntent.USES: 0.90,
    CitationIntent.CRITICIZES: 0.80,
    CitationIntent.COMPARES: 0.70,
    CitationIntent.BACKGROUND: 0.40,
    CitationIntent.UNKNOWN: 0.30,
    CitationIntent.LISTS: 0.10,
}

_EXTEND_PAT = re.compile(r"\b(extend|extends|build on|builds on|built on|improve upon|improves upon|inspired by|following)\b", re.I)
_USE_PAT = re.compile(r"\b(we use|we adopt|we employ|using the|taken from|we apply|based on)\b", re.I)
_COMPARE_PAT = re.compile(r"\b(compare|compared|baseline|outperform|outperforms|better than|worse than)\b", re.I)
_CRITIC_PAT = re.compile(r"\b(fail|fails|limitation|limitations|however|suffer|suffers|cannot|unable)\b", re.I)


def heuristic_intent(context: str, citation_count: int = 1) -> tuple[CitationIntent, float]:
    """没有模型时的兜底判定：按上下文里的关系词猜意图。

    规则很朴素，但方向是对的：**关系词缺失 + 多个编号堆叠 => 应付式引用**。
    """
    text = context or ""
    if _EXTEND_PAT.search(text):
        return CitationIntent.EXTENDS, 0.6
    if _USE_PAT.search(text):
        return CitationIntent.USES, 0.6
    if _COMPARE_PAT.search(text):
        return CitationIntent.COMPARES, 0.55
    if _CRITIC_PAT.search(text):
        return CitationIntent.CRITICIZES, 0.5
    if citation_count >= 3:
        return CitationIntent.LISTS, 0.6
    if len(latin_terms(text)) <= 4:
        return CitationIntent.LISTS, 0.45
    return CitationIntent.BACKGROUND, 0.4


def _specificity(context: str) -> float:
    """上下文的具体程度：是否给出了可验证的技术关系，而不是空泛铺陈。"""
    terms = latin_terms(context)
    has_number = bool(re.search(r"\d", context or ""))
    score = min(1.0, len(terms) / 18.0)
    if has_number:
        score = min(1.0, score + 0.2)
    if _EXTEND_PAT.search(context or "") or _USE_PAT.search(context or ""):
        score = min(1.0, score + 0.15)
    return score


def classify_intents(
    items: list[dict],
    gateway: Optional[Gateway] = None,
    *,
    batch_size: int = 20,
) -> dict[int, dict]:
    """批量判定引用意图。有模型就用模型（一次一批），失败自动退回规则。

    `items` 形如 `[{"id":1,"citing_title":"","cited_ref":"","context":""}]`。
    返回 `{id: {"intent": CitationIntent, "confidence": float, "reason": str, "by": "llm|heuristic"}}`。
    """
    results: dict[int, dict] = {}
    pending = list(items)

    if gateway is not None and gateway.available and pending:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            try:
                data = gateway.json(
                    [
                        {"role": "system", "content": prompts.CITATION_BATCH_SYSTEM},
                        {"role": "user", "content": prompts.build_citation_batch_user(chunk)},
                    ],
                    model=settings.fast_model,
                    max_tokens=1600,
                    kind="graph.intent",
                )
                for row in data.get("items") or []:
                    try:
                        rid = int(row.get("id"))
                    except (TypeError, ValueError):
                        continue
                    intent = _coerce_intent(row.get("intent"))
                    results[rid] = {
                        "intent": intent,
                        "confidence": float(row.get("confidence") or 0.6),
                        "reason": str(row.get("reason") or "")[:200],
                        "by": "llm",
                    }
            except (GatewayError, ValueError, TypeError):
                pass

    for item in pending:
        if item["id"] in results:
            continue
        intent, conf = heuristic_intent(item.get("context", ""), item.get("citation_count", 1))
        results[item["id"]] = {
            "intent": intent,
            "confidence": conf,
            "reason": "规则判定（未使用模型或模型不可用）",
            "by": "heuristic",
        }
    return results


def _coerce_intent(value: Any) -> CitationIntent:
    text = str(value or "").strip().lower()
    for intent in CitationIntent:
        if intent.value == text:
            return intent
    mapping = {
        "extension": CitationIntent.EXTENDS,
        "use": CitationIntent.USES,
        "compare": CitationIntent.COMPARES,
        "criticize": CitationIntent.CRITICIZES,
        "list": CitationIntent.LISTS,
        "mention": CitationIntent.BACKGROUND,
    }
    return mapping.get(text, CitationIntent.UNKNOWN)


def value_score(intent: CitationIntent, context: str, confidence: float = 0.6) -> float:
    """引用含金量：意图权重为主，上下文具体程度为辅。"""
    base = _INTENT_WEIGHT.get(intent, 0.3)
    score = 0.75 * base + 0.25 * _specificity(context)
    # 判定置信度低时向中间收缩，避免把不确定的引用当成"高质量"
    score = score * (0.6 + 0.4 * min(1.0, confidence))
    return round(max(0.0, min(1.0, score)), 4)


# --------------------------------------------------------------------------
# 组网
# --------------------------------------------------------------------------


@dataclass
class CitationNetwork:
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[CitationEdge] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs) -> None:
        node = self.nodes.setdefault(node_id, {"node_id": node_id})
        node.update({k: v for k, v in attrs.items() if v not in (None, "")})

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes.values()),
            "edges": [e.to_dict() for e in self.edges],
            "stats": self.stats,
            "warnings": self.warnings,
        }


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (text or "").lower()).strip()


def load_parsed_doc(data_dir: Path | str, doc_id: str) -> Optional[dict]:
    """读取解析缓存（由 ingest 层写出），读不到返回 None。"""
    path = Path(data_dir) / doc_id / "parsed.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_doc_pages(data_dir: Path | str, doc_id: str) -> dict[int, str]:
    path = Path(data_dir) / doc_id / "pages.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[int, str] = {}
    for k, v in (raw or {}).items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out


def build_network(
    doc_ids: list[str],
    *,
    library: Any = None,
    data_dir: Optional[Path | str] = None,
    gateway: Optional[Gateway] = None,
    classify: bool = True,
    max_contexts_per_doc: int = 60,
) -> CitationNetwork:
    """为一批文献构建局部引用网络。

    节点 = 库内文献 + 被它们引用到的外部文献；边 = 一条引用，带意图与含金量。
    """
    data_dir = Path(data_dir or settings.data_dir)
    net = CitationNetwork()

    papers: dict[str, Any] = {}
    if library is not None:
        for doc_id in doc_ids:
            try:
                paper = library.get_paper(doc_id)
            except Exception:
                paper = None
            if paper is not None:
                papers[doc_id] = paper

    # 1) 库内节点
    for doc_id in doc_ids:
        paper = papers.get(doc_id)
        title = getattr(paper, "title", "") or doc_id
        net.add_node(
            doc_id,
            title=title,
            year=getattr(paper, "year", None),
            in_library=True,
            arxiv_id=getattr(paper, "arxiv_id", ""),
        )

    title_index = {doc_id: _norm_title(getattr(p, "title", "")) for doc_id, p in papers.items()}
    arxiv_index = {
        doc_id: (getattr(p, "arxiv_id", "") or "").lower()
        for doc_id, p in papers.items()
        if getattr(p, "arxiv_id", "")
    }

    pending_intents: list[dict] = []
    pending_edges: dict[int, CitationEdge] = {}
    intent_id = 0

    # 2) 逐篇解析引用
    for doc_id in doc_ids:
        parsed = load_parsed_doc(data_dir, doc_id)
        if parsed is None:
            cached = None
            if library is not None:
                try:
                    cached = library.load_parsed(doc_id)
                except Exception:
                    cached = None
            parsed = cached
        if parsed is None:
            net.warnings.append(f"{doc_id}：缺少解析缓存，无法解析其参考文献（请确认已入库解析）")
            continue

        entries = extract_reference_entries(parsed)
        pages = load_doc_pages(data_dir, doc_id)
        if not pages and library is not None:
            try:
                pages = library.load_page_texts(doc_id) or {}
            except Exception:
                pages = {}
        contexts = extract_in_text_citations(pages, max_contexts=max_contexts_per_doc)
        ctx_by_ref: dict[int, list[dict]] = {}
        for c in contexts:
            ctx_by_ref.setdefault(c["ref_index"], []).append(c)

        citing_title = net.nodes.get(doc_id, {}).get("title", doc_id)

        for entry in entries:
            ref_no = entry["index"]
            ctxs = ctx_by_ref.get(ref_no, [])
            context = ctxs[0]["context"] if ctxs else ""
            page = ctxs[0]["page"] if ctxs else 0
            citation_count = len(_CITE_MARK_RE.findall(context)) or 1

            dst_doc_id, dst_title = _match_target(entry, title_index, arxiv_index, net)
            edge = CitationEdge(
                src_doc_id=doc_id,
                dst_ref=entry["raw"][:500],
                dst_doc_id=dst_doc_id,
                dst_title=dst_title or entry.get("title_guess", ""),
                dst_year=entry.get("year"),
                context=context,
                page=page,
            )
            net.edges.append(edge)

            node_id = dst_doc_id or _external_node_id(entry, dst_title)
            net.add_node(
                node_id,
                title=dst_title or entry.get("title_guess", ""),
                year=entry.get("year"),
                in_library=bool(dst_doc_id),
                arxiv_id=entry.get("arxiv_id", ""),
            )

            if classify:
                intent_id += 1
                pending_intents.append(
                    {
                        "id": intent_id,
                        "citing_title": citing_title,
                        "cited_ref": entry["raw"][:300],
                        "context": context or "（正文未找到引用该条目的上下文）",
                        "citation_count": citation_count,
                    }
                )
                pending_edges[intent_id] = edge

    # 3) 意图判定 + 含金量
    if pending_intents and classify:
        verdicts = classify_intents(pending_intents, gateway)
        for rid, edge in pending_edges.items():
            verdict = verdicts.get(rid)
            if not verdict:
                continue
            intent = verdict["intent"]
            edge.intent = intent
            edge.value_score = value_score(intent, edge.context, verdict.get("confidence", 0.6))
            edge.substantive = intent in SUBSTANTIVE_INTENTS and edge.value_score >= 0.5

    substantive = sum(1 for e in net.edges if e.substantive)
    lists_only = sum(1 for e in net.edges if e.intent is CitationIntent.LISTS)
    net.stats = {
        "nodes": len(net.nodes),
        "edges": len(net.edges),
        "in_library_nodes": sum(1 for n in net.nodes.values() if n.get("in_library")),
        "external_nodes": sum(1 for n in net.nodes.values() if not n.get("in_library")),
        "substantive_edges": substantive,
        "list_only_edges": lists_only,
        "substantive_ratio": round(substantive / len(net.edges), 4) if net.edges else 0.0,
        "contexts_found": sum(1 for e in net.edges if e.context),
    }
    return net


def _external_node_id(entry: dict, title: str) -> str:
    if entry.get("arxiv_id"):
        return f"ext:arxiv:{entry['arxiv_id']}"
    if entry.get("doi"):
        return f"ext:doi:{entry['doi']}"
    key = _norm_title(title)[:60].replace(" ", "-")
    return f"ext:{key or 'unknown'}"


def _match_target(
    entry: dict,
    title_index: dict[str, str],
    arxiv_index: dict[str, str],
    net: CitationNetwork,
) -> tuple[str, str]:
    """把一条参考文献指到库内某篇文献（能指到就指，指不到就返回空）。"""
    ref_arxiv = (entry.get("arxiv_id") or "").lower()
    if ref_arxiv:
        for doc_id, ax in arxiv_index.items():
            if ax == ref_arxiv:
                return doc_id, net.nodes.get(doc_id, {}).get("title", "")

    raw_norm = _norm_title(entry.get("raw", ""))
    if raw_norm:
        for doc_id, norm_title in title_index.items():
            if norm_title and len(norm_title) >= 12 and norm_title in raw_norm:
                return doc_id, net.nodes.get(doc_id, {}).get("title", "")

    guess = _norm_title(entry.get("title_guess", ""))
    if guess and len(guess) >= 12:
        for doc_id, norm_title in title_index.items():
            if not norm_title:
                continue
            if guess in norm_title or norm_title in guess:
                return doc_id, net.nodes.get(doc_id, {}).get("title", "")
    return "", ""

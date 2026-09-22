"""语义级去重与版本族谱。

赛题的验收用例很具体：先入 `1706.03762v7.pdf`，再入 `1706.03762v1.pdf`，
系统必须**基于内容**（而不是文件名）发现"库里已经有这篇论文的更新版本"，
并主动提示保留最新版。

我们用了四路信号，任何一路命中都会进入人工确认流程：

  1. **arXiv 基础号**（去掉版本后缀）—— 最硬，但改名就失效；
  2. **标题相似度** —— 对"同一论文不同版本"很有效；
  3. **正文 MinHash 指纹** —— 真正意义上的"基于内容"，改标题也躲不掉；
  4. **摘要向量余弦** —— 语义级，能抓住"标题被改写"的情况。

最后给出 `relation`（同版本 / 更旧 / 更新 / 近似重复 / 不同论文）、
`recommendation` 和一句**人话提示**，前端直接弹给用户。
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from ..contracts import DedupRelation, DedupVerdict, PaperRecord, ParsedDocument
from ..llm.client import GatewayError, get_gateway

NUM_HASHES = 16
SHINGLE_SIZE = 5
_TITLE_NOISE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


def normalize_title(title: str) -> str:
    return _TITLE_NOISE.sub(" ", (title or "").lower()).strip()


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if len(na) >= 12 and (na in nb or nb in na):
        return 0.95
    return SequenceMatcher(None, na, nb).ratio()


# --------------------------------------------------------------------------
# 内容指纹（MinHash）
# --------------------------------------------------------------------------


def _body_text(target: ParsedDocument | dict | str) -> str:
    if isinstance(target, str):
        return target
    blocks = target.get("blocks") if isinstance(target, dict) else target.blocks
    if not blocks:
        return ""
    pieces = []
    for block in blocks:
        kind = block.get("kind") if isinstance(block, dict) else block.kind
        text = block.get("text") if isinstance(block, dict) else block.text
        if kind in {"paragraph", "heading", "table", "formula"} and text:
            pieces.append(str(text))
    return "\n".join(pieces)


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[int]:
    compact = " ".join((text or "").lower().split())
    if len(compact) < k:
        return {hash(compact)} if compact else set()
    out = set()
    for i in range(len(compact) - k + 1):
        gram = compact[i : i + k]
        out.add(int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big"))
    return out


def content_fingerprint(target: ParsedDocument | dict | str) -> str:
    """正文 MinHash 签名，编码成十六进制字符串存进档案库。

    为什么不用纯哈希：纯哈希只能判"完全相同"，而论文新版本改几个字就变了。
    MinHash 签名可以在**只存一个短字符串**的前提下估计 Jaccard 相似度。
    """
    shingles = _shingles(_body_text(target))
    if not shingles:
        return ""
    signature = []
    for seed in range(NUM_HASHES):
        best = min(
            (h ^ (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) for h in shingles
        ) if seed else min(shingles)
        signature.append(best & 0xFFFF)
    return "".join(f"{value:04x}" for value in signature)


def signature_similarity(sig_a: str, sig_b: str) -> float:
    """两个 MinHash 签名的位置一致率（即 Jaccard 的估计值）。"""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    step = 4
    a = [sig_a[i : i + step] for i in range(0, len(sig_a), step)]
    b = [sig_b[i : i + step] for i in range(0, len(sig_b), step)]
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _embedding_similarity(incoming: PaperRecord, existing: PaperRecord) -> Optional[float]:
    """摘要向量余弦。没有 key 或调用失败就返回 None（表示"这一路信号不可用"）。"""
    text_a = (incoming.abstract or incoming.title or "")[:1200]
    text_b = (existing.abstract or existing.title or "")[:1200]
    if not text_a or not text_b:
        return None
    gateway = get_gateway()
    if not gateway.available:
        return None
    try:
        vectors = gateway.embed([text_a, text_b])
    except (GatewayError, ValueError, TypeError):
        return None
    if len(vectors) < 2:
        return None
    return _cosine(vectors[0], vectors[1])


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------


def _relation_for_versions(newer_is_incoming: bool, same: bool) -> DedupRelation:
    if same:
        return DedupRelation.SAME_VERSION
    return DedupRelation.NEWER_VERSION if newer_is_incoming else DedupRelation.OLDER_VERSION


def check_duplicate(
    incoming: PaperRecord,
    incoming_parsed: Optional[ParsedDocument] = None,
    existing: Optional[Iterable[PaperRecord]] = None,
    *,
    use_embedding: bool = True,
) -> DedupVerdict:
    """入库前查重。返回的结论直接驱动前端弹窗与用户的三个选择。"""
    existing = list(existing or [])
    if not existing:
        return DedupVerdict(
            relation=DedupRelation.DISTINCT,
            similarity=0.0,
            message="库中无同类文献，已直接入库。",
            recommendation="keep_both",
            detected_by=["empty_library"],
        )

    incoming_fp = incoming.content_hash
    if not incoming_fp and incoming_parsed is not None:
        incoming_fp = content_fingerprint(incoming_parsed)

    best: Optional[dict] = None
    for record in existing:
        if record.doc_id == incoming.doc_id:
            continue
        signals: list[str] = []
        scores: list[float] = []

        same_arxiv = bool(
            incoming.arxiv_id
            and record.arxiv_id
            and incoming.arxiv_id.lower() == record.arxiv_id.lower()
        )
        if same_arxiv:
            signals.append("arxiv_base")
            scores.append(1.0)

        tsim = title_similarity(incoming.title, record.title)
        if tsim >= 0.72:
            signals.append("title")
            scores.append(tsim)

        fsim = signature_similarity(incoming_fp, record.content_hash)
        if fsim >= 0.35:
            signals.append("content_hash")
            scores.append(fsim)

        if use_embedding and not same_arxiv:
            esim = _embedding_similarity(incoming, record)
            if esim is not None and esim >= 0.90:
                signals.append("embedding")
                scores.append(esim)

        if not signals:
            continue
        similarity = max(scores)
        candidate = {
            "record": record,
            "signals": signals,
            "similarity": similarity,
            "same_arxiv": same_arxiv,
        }
        if best is None or similarity > best["similarity"]:
            best = candidate

    if best is None or best["similarity"] < 0.6:
        return DedupVerdict(
            relation=DedupRelation.DISTINCT,
            similarity=best["similarity"] if best else 0.0,
            message="未发现内容高度重合的文献，已作为新文献入库。",
            recommendation="keep_both",
            detected_by=best["signals"] if best else [],
        )

    record: PaperRecord = best["record"]
    similarity: float = best["similarity"]
    signals: list[str] = best["signals"]

    if best["same_arxiv"] or similarity >= 0.9:
        if best["same_arxiv"]:
            if incoming.version == record.version:
                relation = DedupRelation.SAME_VERSION
            else:
                relation = _relation_for_versions(
                    incoming.version > record.version, same=False
                )
        else:
            relation = DedupRelation.NEAR_DUPLICATE
    else:
        relation = DedupRelation.NEAR_DUPLICATE

    if relation is DedupRelation.OLDER_VERSION:
        message = (
            f"检测到库中已有该论文的更新版本（v{record.version}），"
            f"你正在上传的是较早的 v{incoming.version}。建议保留库中现有版本。"
        )
        recommendation = "keep_existing"
    elif relation is DedupRelation.NEWER_VERSION:
        message = (
            f"库中已存在同一论文的较早版本 v{record.version}，"
            f"本次上传的是 v{incoming.version}。建议用新版本替换旧版本。"
        )
        recommendation = "replace"
    elif relation is DedupRelation.SAME_VERSION:
        message = "库中已存在内容完全相同的同一版本，无需重复入库。"
        recommendation = "keep_existing"
    else:
        message = (
            f"库中发现内容高度重合的文献（相似度 {similarity:.0%}），"
            "可能是同一工作的不同版本或改写稿，请确认如何处理。"
        )
        recommendation = "keep_existing"

    return DedupVerdict(
        relation=relation,
        similarity=similarity,
        existing_doc_id=record.doc_id,
        existing_version=record.version,
        incoming_version=incoming.version,
        message=message,
        recommendation=recommendation,
        detected_by=signals,
    )
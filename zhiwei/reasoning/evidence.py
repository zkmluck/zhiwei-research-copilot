"""证据组装与校验的纯函数集合。

这一层不调用模型：把「一段文字里有哪些东西是可以被核验的」拆成可单测的小函数。
数字、专有名词、实词覆盖率 —— 幻觉最常出现在这些地方。

**跨语言问题（实测踩到的坑）**：用户看到的主张是中文，证据原文是英文，
两者的词面重合度天然接近 0。如果直接用它当支撑度，所有好主张都会被误降级。
所以这里的核验策略是分层的：

  1. 数字/单位 —— 跨语言最稳的锚点，中英文都照写 `28.4`、`P100`、`d_model`；
  2. 拉丁技术词（模型名、指标名、超参名）—— 中文写作通常保留原文写法；
  3. 中文实词重合 —— 只在原文本身是中文时才可用（例如中文摘要）；
  4. 以上都不可用时，**明确返回「无法判定」**，而不是假装判出了一个低分。
     诚实地说「判不了」，让上层去用大模型裁判或降级处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..contracts import Anchor, Claim, Evidence, normalize_text

# --------------------------------------------------------------------------
# 词法工具
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_.]*|\d+(?:[.,]\d+)*|[\u4e00-\u9fff]")

# 数字 + 可选单位：捕捉 27.3、8、1e-4、3.5、12
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)"
    r"([A-Za-z%]{0,6})",
)

# 典型超参/指标名，命中说明这个数字是"有量纲的关键数字"
_METRIC_HINTS = (
    "bleu", "rouge", "f1", "accuracy", "acc", "wer", "perplexity", "ppl", "map",
    "batch", "lr", "epoch", "dropout", "warmup", "gpu", "tpu", "p100", "v100",
    "a100", "h100", "param", "params", "parameter", "layer", "layers", "head",
    "heads", "dim", "hidden", "token", "tokens", "step", "steps", "hour",
    "hours", "day", "days", "beta", "alpha", "gamma", "k", "m", "b",
)

_STOPWORDS_EN = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are",
    "was", "were", "be", "been", "with", "that", "this", "these", "those", "it",
    "its", "as", "by", "we", "our", "which", "at", "from", "than", "then",
    "have", "has", "had", "can", "may", "not", "but", "also", "such", "using",
    "used", "use", "more", "most", "all", "each", "both", "other", "into",
}

_STOPWORDS_ZH = {
    "的", "了", "是", "在", "和", "与", "及", "对", "为", "被", "把", "上", "中",
    "本文", "这篇", "论文", "方法", "模型", "所示", "表明", "显示", "进行", "使用",
    "因此", "并且", "而且", "但是", "以及", "其中", "这个", "该文", "其",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _ascii_fold(text: str) -> str:
    """把常见的全角字符折成半角，避免「２７.３」这类写法绕过数字核验。"""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19:
            out.append(chr(code - 0xFF10 + ord("0")))
        elif 0xFF21 <= code <= 0xFF3A:
            out.append(chr(code - 0xFF21 + ord("A")))
        elif 0xFF41 <= code <= 0xFF5A:
            out.append(chr(code - 0xFF41 + ord("a")))
        elif code == 0xFF0E:
            out.append(".")
        else:
            out.append(ch)
    return "".join(out)


def latin_terms(text: str) -> set[str]:
    """抽取拉丁技术词：英文单词、缩写、模型名、超参名。跨语言核验的主力。"""
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-_.]{1,}", _ascii_fold(text or "")):
        low = token.lower().strip("._-")
        if low and low not in _STOPWORDS_EN:
            terms.add(low)
    return terms


def cjk_terms(text: str) -> set[str]:
    """抽取中文实词：按双字滑窗切，比单字更能反映语义重合。"""
    chars = [c for c in (text or "") if _CJK_RE.match(c)]
    if not chars:
        return set()
    bigrams = {"".join(chars[i : i + 2]) for i in range(len(chars) - 1)}
    singles = {c for c in chars}
    tokens = bigrams | singles
    return {t for t in tokens if t not in _STOPWORDS_ZH}


def content_terms(text: str) -> set[str]:
    """全量实词（含中英文），用于同语言场景。"""
    return latin_terms(text) | cjk_terms(text) | numbers_in(text)


def numbers_in(text: str) -> set[str]:
    """抽取可核验的数字。返回归一化后的数字集合（含带单位写法）。

    例：`27.3 BLEU` -> {`27.3bleu`, `27.3`}；`1e-4` -> {`1e-4`}。
    """
    found: set[str] = set()
    folded = _ascii_fold(text or "").replace(",", "")
    for num, unit in _NUMBER_RE.findall(folded):
        unit_low = unit.lower()
        if unit_low in _METRIC_HINTS:
            found.add(f"{num}{unit_low}")
        found.add(num)
    return found


def numeric_support(claim_text: str, sources: list[str]) -> tuple[bool, list[str]]:
    """核验主张中的数字是否都被来源支持。返回 (是否全部支持, 找不到的数字)。"""
    claim_numbers = numbers_in(claim_text)
    if not claim_numbers:
        return True, []
    source_numbers: set[str] = set()
    for s in sources:
        source_numbers |= numbers_in(s)
    missing = sorted(
        n for n in claim_numbers if n not in source_numbers and not _number_loosely_present(n, sources)
    )
    return (not missing), missing


def _number_loosely_present(number: str, sources: list[str]) -> bool:
    """带单位的数字允许"数字在、单位换了写法"（如 28.4 BLEU vs BLEU 28.4）。"""
    digits = number.rstrip("abcdefghijklmnopqrstuvwxyz%")
    if not digits or digits == number:
        return False
    return any(digits in s for s in sources)


def cross_lingual_support(claim_text: str, sources: list[str]) -> tuple[Optional[float], str]:
    """跨语言支撑度。

    返回 (分数或 None, 方法名)。None 表示**这一层判不了**，
    调用方应当转向语义裁判，而不是把 None 当作 0 分。
    """
    lt = latin_terms(claim_text)
    source_latin = latin_terms(" ".join(sources))
    if lt:
        hit = sum(1 for t in lt if t in source_latin)
        return hit / len(lt), "latin_terms"
    # 纯中文主张：只有当来源也是中文时，词面比对才有意义
    ct = cjk_terms(claim_text)
    if ct and cjk_terms(" ".join(sources)):
        source_cjk = cjk_terms(" ".join(sources))
        hit = sum(1 for t in ct if t in source_cjk)
        return hit / len(ct), "cjk_terms"
    return None, "undecidable"


def lexical_overlap(claim_text: str, sources: list[str]) -> float:
    """同语言词面重合度（保留给中文对中文、英文对英文的场景）。"""
    score, method = cross_lingual_support(claim_text, sources)
    if score is not None:
        return score
    needle = content_terms(claim_text)
    if not needle:
        return 1.0
    haystack = content_terms(" ".join(sources))
    if not haystack:
        return 0.0
    return sum(1 for t in needle if t in haystack) / len(needle)


def split_sentences(text: str) -> list[str]:
    """粗粒度断句，供逐句溯源用（中英标点都认）。"""
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])", text)
    return [p.strip() for p in parts if p and p.strip()]


# --------------------------------------------------------------------------
# 证据组装
# --------------------------------------------------------------------------


@dataclass
class GateContext:
    """闸门做判断时需要的外部信息。

    page_texts 是 `{doc_id: {page: 该页纯文本}}`，用来把引文反查回精确位置。
    拿不到 page_texts 时，闸门会退化为「只能判引文一致性，不能判位置」。
    """

    page_texts: dict[str, dict[int, str]] = field(default_factory=dict)
    blocks_by_page: dict[str, dict[int, list]] = field(default_factory=dict)
    doc_titles: dict[str, str] = field(default_factory=dict)
    strict_location: bool = True

    def pages_of(self, doc_id: str) -> dict[int, str]:
        return self.page_texts.get(doc_id, {})


def build_prompt_blocks(evidences: list[Evidence]) -> list[dict]:
    """把证据整理成给模型看的编号材料块。"""
    blocks: list[dict] = []
    for i, ev in enumerate(evidences):
        blocks.append(
            {
                "index": i + 1,
                "doc_id": ev.doc_id,
                "doc_title": ev.doc_title or ev.doc_id,
                "page": ev.anchor.page,
                "text": ev.quote,
                "section_path": [],
            }
        )
    return blocks


def locate_evidence(
    ev: Evidence,
    page_texts: dict[int, str],
    blocks_by_page: Optional[dict[int, list]] = None,
) -> Optional[Anchor]:
    """把一条证据的引文反查回精确锚点。

    优先使用 index 层的 locate_quote（更强），拿不到就退回本模块的简易实现。
    """
    try:
        from ..store.index import locate_quote  # 延迟导入，避免循环依赖

        anchor = locate_quote(page_texts, ev.quote, blocks_by_page)
        if anchor is not None:
            return anchor
    except Exception:
        pass
    return _simple_locate(page_texts, ev.quote, ev.anchor.page)


def _simple_locate(page_texts: dict[int, str], quote: str, page_hint: int = 0) -> Optional[Anchor]:
    """简易反查：空白归一化后做大小写不敏感包含匹配。找不到就返回 None。"""
    target = normalize_text(quote)
    if len(target) < 8:
        return None
    pages = [page_hint] if page_hint in page_texts else []
    pages += [p for p in sorted(page_texts) if p not in pages]
    for page in pages:
        raw = page_texts.get(page) or ""
        if not raw:
            continue
        hay = normalize_text(raw)
        pos = hay.find(target)
        matched = target
        if pos < 0:
            for cut in (200, 120, 64, 32):
                head = target[:cut]
                if len(head) < 24:
                    continue
                pos = hay.find(head)
                if pos >= 0:
                    matched = head
                    break
        if pos >= 0:
            return Anchor(
                page=page,
                quote=quote,
                char_start=pos,
                char_end=pos + len(matched),
                bbox=None,
                block_kind="text",
            )
    return None


def attach_anchors(
    raw_claims: list[dict],
    context: GateContext,
    *,
    fallback_doc_id: str = "",
) -> list[Claim]:
    """把模型返回的原始主张（带 quotes 字符串）转成带锚点的 Claim。

    这是「溯源」真正落地的地方：引文能被反查回原文，就带上精确锚点；
    反查不到，就保留引文但**不伪造位置** —— 后续闸门会据此拦下它。
    """
    claims: list[Claim] = []
    for idx, raw in enumerate(raw_claims):
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        quotes = [str(q).strip() for q in (raw.get("quotes") or []) if str(q).strip()]
        doc_hint = str(raw.get("doc_id") or fallback_doc_id or "")
        evidence: list[Evidence] = []
        for qi, quote in enumerate(quotes):
            doc_id = doc_hint
            if not doc_id and context.doc_titles:
                doc_id = next(iter(context.doc_titles))
            hint_page = 0
            pages = raw.get("pages") or []
            if isinstance(pages, list) and qi < len(pages) and isinstance(pages[qi], int):
                hint_page = pages[qi]
            anchor = locate_evidence(
                Evidence(doc_id=doc_id, anchor=Anchor(page=hint_page, quote=quote)),
                context.pages_of(doc_id),
                (context.blocks_by_page or {}).get(doc_id),
            )
            if anchor is None:
                anchor = Anchor(page=hint_page, quote=quote, char_start=-1, char_end=-1)
            evidence.append(
                Evidence(
                    doc_id=doc_id,
                    anchor=anchor,
                    score=1.0 if anchor.is_located() else 0.0,
                    source="text",
                    doc_title=context.doc_titles.get(doc_id, ""),
                )
            )
        claims.append(
            Claim(
                text=text,
                kind=str(raw.get("kind") or "fact"),
                evidence=evidence,
                claim_id=f"c{idx + 1}",
            )
        )
    return claims
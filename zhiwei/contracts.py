"""全系统共享的数据契约。

这是"溯源"能成立的地基：任何一句要交给用户的结论，都必须能表达成
`Claim(text=..., evidence=[Evidence(anchor=Anchor(...))])`。
形状固定，就不需要去解析散文，也不会出现"看起来像引用"的假引用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------------
# 定位：从"页"到"字符区间"
# --------------------------------------------------------------------------


@dataclass
class BBox:
    """页面坐标系下的矩形（PDF 点，左上原点）。"""

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Anchor:
    """原文锚点：一次引用到底指向哪里。

    page 从 1 开始计数；char_start/char_end 是该页纯文本中的字符区间；
    bbox 用于前端在 PDF 原文上画高亮框。
    """

    page: int
    quote: str
    char_start: int = -1
    char_end: int = -1
    bbox: Optional[BBox] = None
    block_kind: str = "text"  # text | table | figure | formula | caption | reference

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data

    def label(self) -> str:
        return f"p.{self.page}"

    def is_located(self) -> bool:
        """有没有可点击/可高亮的精确位置。"""
        return self.bbox is not None or (self.char_start >= 0 and self.char_end > self.char_start)


@dataclass
class Evidence:
    """一条证据：指向某篇文献某处的原文片段。"""

    doc_id: str
    anchor: Anchor
    score: float = 0.0
    source: str = "text"  # text | ocr | table | figure | formula | reference
    doc_title: str = ""
    chunk_id: str = ""

    @property
    def quote(self) -> str:
        return self.anchor.quote

    def to_dict(self) -> dict:
        data = asdict(self)
        data["anchor"] = self.anchor.to_dict()
        return data

    def citation_key(self) -> str:
        return f"{self.doc_id}#p{self.anchor.page}"


# --------------------------------------------------------------------------
# 主张与闸门裁决
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    ALLOW = "allow"        # 有证据、够确定 —— 直接出口
    DOWNGRADE = "downgrade"  # 证据薄弱或不确定 —— 降级为"待核实"
    BLOCK = "block"        # 无证据 / 被判定为幻觉 —— 不许出口


@dataclass
class GateDecision:
    """闸门对一条主张的类型化裁决。"""

    verdict: Verdict
    confidence: float
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "checks": self.checks,
        }


@dataclass
class Claim:
    """一句准备交给用户的话，以及它的证据。"""

    text: str
    evidence: list[Evidence] = field(default_factory=list)
    kind: str = "fact"  # fact | comparison | summary | inference | number
    decision: Optional[GateDecision] = None
    claim_id: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "kind": self.kind,
            "evidence": [e.to_dict() for e in self.evidence],
            "decision": self.decision.to_dict() if self.decision else None,
        }


@dataclass
class Answer:
    """一次问答的完整结果：文本 + 逐句溯源 + 拒答说明。"""

    question: str
    text: str
    claims: list[Claim] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    trace: list[dict] = field(default_factory=list)

    @property
    def support_rate(self) -> float:
        if not self.claims:
            return 0.0
        allowed = sum(1 for c in self.claims if c.decision and c.decision.allowed)
        return allowed / len(self.claims)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "text": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "support_rate": round(self.support_rate, 4),
            "claims": [c.to_dict() for c in self.claims],
            "trace": self.trace,
        }


# --------------------------------------------------------------------------
# 文献与解析产物
# --------------------------------------------------------------------------


class DocStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"   # 被更新版本取代
    DUPLICATE = "duplicate"
    FAILED = "failed"


@dataclass
class PaperRecord:
    """一篇入库存档的文献。"""

    doc_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    arxiv_id: str = ""
    version: int = 1
    doi: str = ""
    url: str = ""
    page_count: int = 0
    file_name: str = ""
    sha256: str = ""
    content_hash: str = ""       # 正文语义指纹，用于"不看文件名"的去重
    family_id: str = ""          # 版本族谱 ID
    is_latest: bool = True
    status: DocStatus = DocStatus.ACTIVE
    needs_ocr: bool = False
    ingested_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    def display_name(self) -> str:
        base = self.title or self.file_name or self.doc_id
        if self.arxiv_id and self.version:
            return f"{base} (arXiv:{self.arxiv_id}v{self.version})"
        return base


@dataclass
class Section:
    """标题树上的一个节点。"""

    level: int
    title: str
    page: int
    number: str = ""
    bbox: Optional[BBox] = None
    parent: Optional[str] = None
    section_id: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class Block:
    """粗粒度解析块：正文段落 / 表格 / 图 / 公式 / 标题 / 参考文献。"""

    kind: str            # paragraph | heading | table | figure | formula | reference | caption
    page: int
    text: str
    bbox: Optional[BBox] = None
    section_path: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class Chunk:
    """入库检索的最小单元：带完整位置信息的文本片。"""

    chunk_id: str
    doc_id: str
    text: str
    page: int
    kind: str = "text"
    section_path: list[str] = field(default_factory=list)
    char_start: int = -1
    char_end: int = -1
    bbox: Optional[BBox] = None
    tokens: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        return data

    def as_anchor(self, quote: Optional[str] = None) -> Anchor:
        return Anchor(
            page=self.page,
            quote=(quote if quote is not None else self.text)[:600],
            char_start=self.char_start,
            char_end=self.char_end,
            bbox=self.bbox,
            block_kind=self.kind,
        )


@dataclass
class ParsedDocument:
    """一次解析的完整产物（可序列化后缓存）。"""

    record: PaperRecord
    sections: list[Section] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    tables: int = 0
    figures: int = 0
    formulas: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "blocks": [b.to_dict() for b in self.blocks],
            "chunks": [c.to_dict() for c in self.chunks],
            "references": list(self.references),
            "tables": self.tables,
            "figures": self.figures,
            "formulas": self.formulas,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# 版本族谱与去重
# --------------------------------------------------------------------------


class DedupRelation(str, Enum):
    SAME_VERSION = "same_version"
    OLDER_VERSION = "older_version"
    NEWER_VERSION = "newer_version"
    NEAR_DUPLICATE = "near_duplicate"
    DISTINCT = "distinct"


@dataclass
class DedupVerdict:
    """入库前的查重结论，直接驱动前端弹窗。"""

    relation: DedupRelation
    similarity: float
    existing_doc_id: str = ""
    existing_version: int = 0
    incoming_version: int = 0
    message: str = ""
    recommendation: str = ""      # keep_existing | replace | keep_both
    detected_by: list[str] = field(default_factory=list)  # title | content_hash | embedding | arxiv_base
    version_diff: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "relation": self.relation.value,
            "similarity": round(self.similarity, 4),
            "existing_doc_id": self.existing_doc_id,
            "existing_version": self.existing_version,
            "incoming_version": self.incoming_version,
            "message": self.message,
            "recommendation": self.recommendation,
            "detected_by": self.detected_by,
            "version_diff": self.version_diff,
        }


# --------------------------------------------------------------------------
# 引用图谱
# --------------------------------------------------------------------------


class CitationIntent(str, Enum):
    """引用意图。用于剔除"应付式引用"。"""

    BACKGROUND = "background"   # 背景铺陈：“许多工作研究了……”
    EXTENDS = "extends"         # 实质继承：本文在其基础上扩展
    USES = "uses"               # 直接使用其方法/数据/代码
    COMPARES = "compares"       # 作为基线对照
    CRITICIZES = "criticizes"   # 指出其不足
    LISTS = "lists"             # 纯罗列，无具体论断（应付式引用）
    UNKNOWN = "unknown"


SUBSTANTIVE_INTENTS = {
    CitationIntent.EXTENDS,
    CitationIntent.USES,
    CitationIntent.COMPARES,
    CitationIntent.CRITICIZES,
}


@dataclass
class CitationEdge:
    """引用网络的一条边，带上下文与含金量判定。"""

    src_doc_id: str
    dst_ref: str                 # 原始参考文献字符串
    dst_doc_id: str = ""         # 若能在库内/外部数据源解析出实体
    dst_title: str = ""
    dst_year: Optional[int] = None
    context: str = ""            # 引用出现的原句
    page: int = 0
    intent: CitationIntent = CitationIntent.UNKNOWN
    value_score: float = 0.0     # 引用含金量 0-1
    substantive: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["intent"] = self.intent.value
        return data


@dataclass
class GraphInsight:
    """图谱挖掘的结论：谁是基石，谁是边缘衍生，谁被架空了。"""

    doc_id: str
    title: str = ""
    year: Optional[int] = None
    pagerank: float = 0.0
    betweenness: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    k_core: int = 0
    role: str = ""               # cornerstone | bridge | derivative | isolated
    reason: str = ""
    in_library: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FutureWorkItem:
    """一条 Future Work 主张，以及它在时间线上是否还活着。"""

    doc_id: str
    text: str
    anchor: Optional[Anchor] = None
    status: str = "unknown"      # open | resolved | stale | unknown
    resolved_by: list[str] = field(default_factory=list)
    evidence: str = ""
    checked_at: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.anchor is not None:
            data["anchor"] = self.anchor.to_dict()
        return data


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def dedupe_evidence(items: Iterable[Evidence]) -> list[Evidence]:
    """按 (doc, page, 前 60 字) 去重，保留分数最高的一条。"""
    best: dict[tuple[str, int, str], Evidence] = {}
    for ev in items:
        key = (ev.doc_id, ev.anchor.page, (ev.quote or "")[:60].strip())
        cur = best.get(key)
        if cur is None or ev.score > cur.score:
            best[key] = ev
    return sorted(best.values(), key=lambda e: -e.score)


def normalize_text(text: str) -> str:
    """压缩空白，便于把模型返回的引文片段定位回原文。"""
    return " ".join((text or "").split())
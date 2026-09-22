"""推理层：检索、抽取、对比，以及整个系统的守门人 —— 主张闸门。"""

from .evidence import (
    GateContext,
    attach_anchors,
    build_prompt_blocks,
    content_terms,
    lexical_overlap,
    numbers_in,
    split_sentences,
)
from .claim_gate import ClaimGate, GateConfig

__all__ = [
    "ClaimGate",
    "GateConfig",
    "GateContext",
    "attach_anchors",
    "build_prompt_blocks",
    "content_terms",
    "lexical_overlap",
    "numbers_in",
    "split_sentences",
]
"""学术翻译：中英对照与整篇翻译。

两个容易被忽略但很关键的细节：

  1. **公式与代码必须原样保留**。翻译前把它们替换成占位符，翻译后填回，
     否则模型会把 `\\beta_1` 翻译成别的什么，或者把变量名改掉。
  2. **按段翻译，不按整篇**。段落边界对齐是"中英联动滚动"能同步滚动的前提，
     所以翻译产物是 segment 列表而不是一坨文本。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ..config import settings
from ..llm.client import Gateway, GatewayError, get_gateway

# 需要保护、不参与翻译的片段
_PROTECT_PATTERNS = [
    (re.compile(r"\$\$[^$]+\$\$", re.S), "FORMULA"),          # 行间公式
    (re.compile(r"\$[^$\n]{1,200}\$"), "FORMULA"),            # 行内公式
    (re.compile(r"\\\([^)]{1,200}\\\)", re.S), "FORMULA"),
    (re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.S), "BLOCK"),
    (re.compile(r"```.*?```", re.S), "CODE"),                 # 代码块
    (re.compile(r"`[^`\n]{1,120}`"), "CODE"),                 # 行内代码
    (re.compile(r"\[[0-9]+(?:[,\-–][0-9]+)*\]"), "CITE"),      # 引用编号
]


def protect(text: str) -> tuple[str, list[str]]:
    """把公式/代码/引用编号抽出来，换成占位符。"""
    stash: list[str] = []

    def _sub(pattern: re.Pattern, tag: str, source: str) -> str:
        def repl(match: re.Match) -> str:
            stash.append(match.group(0))
            return f"@@{tag}{len(stash) - 1}@@"

        return pattern.sub(repl, source)

    out = text
    for pattern, tag in _PROTECT_PATTERNS:
        out = _sub(pattern, tag, out)
    return out, stash


def restore(text: str, stash: list[str]) -> str:
    """把占位符填回原文。模型偶尔会改动占位符大小写或空格，这里做容错匹配。"""
    def repl(match: re.Match) -> str:
        idx = int(match.group(1))
        return stash[idx] if 0 <= idx < len(stash) else match.group(0)

    return re.sub(r"@@\s*[A-Za-z]+\s*(\d+)\s*@@", repl, text)


def translate_segment(
    text: str,
    *,
    to: str = "en",
    gateway: Optional[Gateway] = None,
    keep_terms: Optional[list[str]] = None,
) -> dict:
    """翻译一段文本。返回 `{"src","dst","terms"}`。"""
    gateway = gateway or get_gateway()
    if not (text or "").strip():
        return {"src": text, "dst": "", "terms": []}

    protected, stash = protect(text)
    prompt = protected
    if keep_terms:
        prompt += "\n\n（以下术语保持原文： " + "、".join(keep_terms[:20]) + " ）"

    try:
        translated = gateway.translate(prompt, to=to)
    except GatewayError as exc:
        return {"src": text, "dst": "", "terms": [], "error": f"{exc.kind}: {exc}"}

    return {
        "src": text,
        "dst": restore(translated, stash),
        "terms": keep_terms or [],
        "to": to,
        "protected": len(stash),
    }


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text or "")
    return [p.strip() for p in parts if p.strip()]


def translate_document(
    segments: list[dict],
    *,
    to: str = "zh",
    gateway: Optional[Gateway] = None,
    max_segments: int = 40,
) -> dict:
    """逐段翻译，保留段落索引与位置信息，供前端做中英联动滚动。"""
    gateway = gateway or get_gateway()
    out: list[dict] = []
    for i, seg in enumerate(segments[:max_segments]):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        result = translate_segment(text, to=to, gateway=gateway)
        out.append(
            {
                "index": i,
                "page": seg.get("page"),
                "src": text,
                "dst": result.get("dst", ""),
                "bbox": seg.get("bbox"),
                "error": result.get("error"),
            }
        )
    return {
        "to": to,
        "segments": out,
        "count": len(out),
        "usage": gateway.ledger.summary(),
    }
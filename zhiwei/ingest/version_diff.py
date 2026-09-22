"""版本差异：v1 到 v7 到底改了什么。

赛题把它列为"高级表现"（"甚至能够自动对比并总结出两个版本之间的主要修订差异"）。
做法是先做**确定性**的章节对齐与段落差集（纯 Python，可复现、可离线），
再让模型把差异归纳成人话；模型不可用时，用统计量拼一句摘要 —— 不会没有产出。
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import settings
from ..contracts import ParsedDocument
from ..llm import prompts
from ..llm.client import Gateway, GatewayError
from .dedup import normalize_title

_MAX_CHANGES = 120


def _blocks_of(target: ParsedDocument | dict) -> list[dict]:
    blocks = target.get("blocks") if isinstance(target, dict) else target.blocks
    out = []
    for block in blocks or []:
        if isinstance(block, dict):
            item = {
                "kind": block.get("kind", "paragraph"),
                "page": int(block.get("page") or 0),
                "text": str(block.get("text") or "").strip(),
                "path": [str(p) for p in (block.get("section_path") or [])],
            }
        else:
            item = {
                "kind": block.kind,
                "page": block.page,
                "text": (block.text or "").strip(),
                "path": [str(p) for p in (block.section_path or [])],
            }
        if item["text"] and item["kind"] in {"paragraph", "heading", "table", "formula", "caption"}:
            out.append(item)
    return out


def _key(text: str, size: int = 64) -> str:
    return " ".join(text[:size].split()).lower()


def diff_versions(
    old: ParsedDocument | dict,
    new: ParsedDocument | dict,
    *,
    gateway: Optional[Gateway] = None,
    max_changes: int = _MAX_CHANGES,
) -> dict:
    """返回 `{"summary","highlights","changes","stats"}`。"""
    old_blocks = _blocks_of(old)
    new_blocks = _blocks_of(new)

    old_map: dict[tuple, dict] = {}
    for block in old_blocks:
        section = normalize_title(block["path"][-1] if block["path"] else "")
        old_map[(section, _key(block["text"]))] = block

    matched_new_keys: set[tuple] = set()
    changes: list[dict] = []

    # 1) 新增 / 修改
    for block in new_blocks:
        section = normalize_title(block["path"][-1] if block["path"] else "")
        key = (section, _key(block["text"]))
        if key in old_map:
            matched_new_keys.add(key)
            continue
        # 同章节下是否有"位置对应但内容变了"的段落
        same_section = [
            (k, v) for k, v in old_map.items() if k[0] == section and k not in matched_new_keys
        ]
        replaced = None
        for k, candidate in same_section:
            if _overlap(candidate["text"], block["text"]) >= 0.45:
                replaced = (k, candidate)
                break
        if replaced:
            k, candidate = replaced
            matched_new_keys.add(k)
            changes.append(
                {
                    "kind": "modified",
                    "section": block["path"][-1] if block["path"] else "",
                    "page": block["page"],
                    "before": candidate["text"][:600],
                    "after": block["text"][:600],
                    "overlap": round(_overlap(candidate["text"], block["text"]), 3),
                }
            )
        else:
            changes.append(
                {
                    "kind": "added",
                    "section": block["path"][-1] if block["path"] else "",
                    "page": block["page"],
                    "before": "",
                    "after": block["text"][:600],
                }
            )
        if len(changes) >= max_changes:
            break

    # 2) 删除
    if len(changes) < max_changes:
        for key, block in old_map.items():
            if key in matched_new_keys:
                continue
            changes.append(
                {
                    "kind": "removed",
                    "section": block["path"][-1] if block["path"] else "",
                    "page": block["page"],
                    "before": block["text"][:600],
                    "after": "",
                }
            )
            if len(changes) >= max_changes:
                break

    stats = {
        "added": sum(1 for c in changes if c["kind"] == "added"),
        "removed": sum(1 for c in changes if c["kind"] == "removed"),
        "modified": sum(1 for c in changes if c["kind"] == "modified"),
        "old_blocks": len(old_blocks),
        "new_blocks": len(new_blocks),
    }

    summary = ""
    highlights: list[str] = []
    gateway = gateway
    if gateway is not None and gateway.available and changes:
        old_record = old.get("record") if isinstance(old, dict) else old.record
        new_record = new.get("record") if isinstance(new, dict) else new.record
        try:
            data = gateway.json(
                [
                    {"role": "system", "content": prompts.VERSION_DIFF_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_version_diff_user(
                            (new_record or {}).get("title") if isinstance(new_record, dict)
                            else getattr(new_record, "title", ""),
                            int((old_record or {}).get("version", 1)) if isinstance(old_record, dict)
                            else getattr(old_record, "version", 1),
                            int((new_record or {}).get("version", 1)) if isinstance(new_record, dict)
                            else getattr(new_record, "version", 1),
                            changes,
                        ),
                    },
                ],
                model=settings.fast_model,
                max_tokens=800,
                kind="version.diff",
            )
            summary = str(data.get("summary") or "").strip()
            highlights = [str(h) for h in (data.get("highlights") or [])][:4]
        except (GatewayError, ValueError, TypeError):
            summary = ""

    if not summary:
        summary = (
            f"共识别出 {len(changes)} 处章节级差异："
            f"新增 {stats['added']} 段、修改 {stats['modified']} 段、删除 {stats['removed']} 段。"
            f"（未配置归纳模型，以上为确定性统计结果）"
        )

    return {"summary": summary, "highlights": highlights, "changes": changes, "stats": stats}


def _overlap(a: str, b: str) -> float:
    """两段文字的词面重叠率，用来判断"这是改过的同一段"还是"全新的一段"。"""
    from ..reasoning.evidence import content_terms

    ta, tb = content_terms(a), content_terms(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))
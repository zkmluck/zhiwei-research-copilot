"""Agent 的工作记忆。

做三件事，每件都对应赛题里的一个考察点：

1. **留痕**：每一步调了什么工具、拿到什么、花了多久，全部记下来，前端能看到 —— 不做黑盒。
2. **预算**：观察结果按字符预算压缩。超预算就把较早的观察折叠成计数摘要，
   而不是让上下文无上限地膨胀（Context Compression）。
3. **证据池**：所有工具抓到的证据（带页码锚点）汇总去重。工具不是白调的 ——
   它们捞到的证据会进入最终答案的候选材料，答案仍然要过闸门。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


def clip(text: Any, limit: int = 240) -> str:
    """把任意值压成一句短文本，用于摘要与前端展示。"""
    value = str(text or "").strip().replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def jsonable(value: Any, *, max_str: int = 400, max_items: int = 40, depth: int = 0) -> Any:
    """把工具返回值压成可 JSON 序列化、且不会爆炸的结构。"""
    if depth > 6:
        return clip(value, 60)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_str else value[: max_str - 1] + "…"
    if isinstance(value, dict):
        return {str(k): jsonable(v, max_str=max_str, max_items=max_items, depth=depth + 1) for k, v in list(value.items())[:max_items]}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v, max_str=max_str, max_items=max_items, depth=depth + 1) for v in list(value)[:max_items]]
    for attr in ("to_dict", "model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return jsonable(method(), max_str=max_str, max_items=max_items, depth=depth + 1)
            except Exception:  # noqa: BLE001
                break
    return clip(value, max_str)


@dataclass
class Observation:
    """一条工具调用记录。"""

    step: int
    tool: str
    ok: bool
    summary: str
    payload: dict = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    ms: int = 0
    folded: bool = False

    def brief(self) -> str:
        mark = "OK" if self.ok else "FAIL"
        head = f"[step {self.step}] {self.tool} {mark} {self.ms}ms | {self.summary}"
        if self.highlights:
            head += " | 要点：" + "；".join(clip(h, 90) for h in self.highlights[:2])
        return head

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
            "highlights": list(self.highlights),
            "ms": self.ms,
            "payload": self.payload,
            "evidence_count": len(self.evidence),
            "folded": self.folded,
        }


class WorkingMemory:
    """带预算的观察台账。"""

    def __init__(self, goal: str, *, budget_chars: int = 8000, keep_per_tool: int = 2) -> None:
        self.goal = goal
        self.budget_chars = budget_chars
        self.keep_per_tool = keep_per_tool
        self._observations: list[Observation] = []
        self._folded: list[Observation] = []
        self.compressions = 0

    # ---------------------------------------------------------------- 写入
    def add(self, observation: Observation) -> Observation:
        self._observations.append(observation)
        if self.chars > self.budget_chars:
            self.compress()
        return observation

    # ---------------------------------------------------------------- 读取
    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    @property
    def folded(self) -> list[Observation]:
        return list(self._folded)

    @property
    def chars(self) -> int:
        return sum(len(json.dumps(o.to_dict(), ensure_ascii=False)) for o in self._observations)

    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obs in list(self._folded) + list(self._observations):
            counts[obs.tool] = counts.get(obs.tool, 0) + 1
        return counts

    # ---------------------------------------------------------------- 压缩
    def compress(self) -> dict:
        """按工具分组，只保留每个工具最近的 N 条完整观察，其余折叠成一句摘要。"""
        keep: list[Observation] = []
        seen: dict[str, int] = {}
        for obs in reversed(self._observations):
            seen[obs.tool] = seen.get(obs.tool, 0) + 1
            if seen[obs.tool] <= self.keep_per_tool:
                keep.append(obs)
            else:
                folded = Observation(
                    step=obs.step,
                    tool=obs.tool,
                    ok=obs.ok,
                    summary=clip(obs.summary, 120),
                    payload={},
                    highlights=[clip(h, 60) for h in obs.highlights[:1]],
                    evidence=[],
                    ms=obs.ms,
                    folded=True,
                )
                self._folded.append(folded)
        keep.reverse()
        self._observations = keep
        self.compressions += 1
        return self.stats()

    # ---------------------------------------------------------------- 证据
    def evidence_pool(self) -> list[dict]:
        """把所有工具捞到的证据汇总去重。最终的答案只能从这里取料。"""
        pool: list[dict] = []
        seen: set[tuple] = set()
        for obs in self._observations + self._folded:
            for item in obs.evidence:
                key = (
                    str(item.get("doc_id") or ""),
                    int(item.get("page") or 0),
                    clip(item.get("quote") or "", 60),
                )
                if key in seen:
                    continue
                seen.add(key)
                pool.append(dict(item))
        return pool

    def relevant_doc_ids(self, limit: int = 6) -> list[str]:
        """按证据命中次数给文献排序，用于把最终答案收窄到真正相关的几篇。"""
        counts: dict[str, int] = {}
        for item in self.evidence_pool():
            doc_id = str(item.get("doc_id") or "")
            if doc_id:
                counts[doc_id] = counts.get(doc_id, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [doc_id for doc_id, _ in ranked[:limit]]

    # ---------------------------------------------------------------- 给模型看
    def brief(self, *, limit_chars: int = 2600) -> str:
        lines = [f"目标：{self.goal}", f"已完成 {len(self._observations) + len(self._folded)} 步观察："]
        if self._folded:
            counts = {}
            for obs in self._folded:
                counts[obs.tool] = counts.get(obs.tool, 0) + 1
            folded_text = "、".join(f"{k}×{v}" for k, v in sorted(counts.items()))
            lines.append(f"（较早的 {len(self._folded)} 条观察已折叠：{folded_text}）")
        text = "\n".join(lines)
        for obs in self._observations:
            line = obs.brief()
            if len(text) + len(line) > limit_chars:
                text += "\n（超预算，剩余观察略）"
                break
            text += "\n" + line
        return text

    def stats(self) -> dict:
        return {
            "observations": len(self._observations),
            "folded": len(self._folded),
            "compressions": self.compressions,
            "chars": self.chars,
            "budget_chars": self.budget_chars,
            "tools": self.tool_counts(),
            "evidence": len(self.evidence_pool()),
        }

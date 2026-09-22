"""Agent 的规划器：把一句调研目标拆成可执行的工具调用序列。

两条路：

* **有模型**：把工具名册交给模型，由它产出 JSON 计划。工具名必须逐字命中名册，
  名册里没有的一律丢弃 —— 规划器不许发明工具。
* **没模型**：走确定性规则计划（关键词 → 工具），保证评审环境没网没 key 时，
  Agent 仍然是一个会规划、会调工具的 Agent，而不是退化成一句报错。

规划器只负责「下一步做什么」，不负责「结论是什么」。结论必须由工具捞到的证据支撑，
并且最终还要过主张闸门。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from ..llm import prompts
from .memory import WorkingMemory, clip


@dataclass
class Step:
    index: int
    tool: str
    args: dict = field(default_factory=dict)
    why: str = ""
    label: str = ""
    source: str = "llm"

    def key(self) -> tuple:
        return (self.tool, json.dumps(self.args, sort_keys=True, ensure_ascii=False, default=str))

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "tool": self.tool,
            "args": self.args,
            "why": self.why,
            "label": self.label or self.tool,
            "source": self.source,
        }


@dataclass
class Plan:
    steps: list[Step]
    source: str
    stop_when: str = ""
    note: str = ""
    replanned: bool = False

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "source": self.source,
            "stop_when": self.stop_when,
            "note": self.note,
            "replanned": self.replanned,
        }


# 关键词 → 追加工具。顺序即优先级，避免"万能计划"把每一步都塞满。
_GRAPH_WORDS = ("图谱", "引用", "脉络", "关系", "基石", "桥接", "流派", "演化", "谱系", "谁影响")
_FUTURE_WORDS = ("future", "未来", "趋势", "空白", "还没", "未解决", "新方向", "能不能做", "值得做", "下一步")
_SURVEY_WORDS = ("综述", "梳理", "总结", "概览", "领域", "线")
_OUTLINE_WORDS = ("写作", "开题", "写作框架", "框架", "写一篇", "投稿", "论文草稿", "idea")
_DISCOVER_WORDS = ("最新", "外部", "新论文", "有没有人", "检索一下", "arxiv", "最近")


def library_summary(library: Any) -> dict:
    papers = library.list_papers()
    return {
        "papers": len(papers),
        "titles": [f"{p.title or p.doc_id}({p.year or '?'})" for p in papers[:8]],
    }


def _label_for(tool: str, args: dict) -> str:
    mapping = {
        "library.stats": "先看清库里有什么",
        "library.search": f"检索证据：{clip(args.get('query'), 30)}",
        "graph.build": "构建引用图谱",
        "graph.key_nodes": "挖核心节点",
        "future_work.scan": "扫描 Future Work 时效性",
        "survey.generate": "生成领域综述草稿",
        "writing.outline": "生成写作框架",
        "discover.search": "去外部源找新论文",
    }
    return mapping.get(tool, tool)


def offline_plan(
    goal: str,
    *,
    library: Any,
    registry: Any,
    doc_ids: Optional[list[str]] = None,
    max_steps: int = 6,
    source: str = "offline",
) -> Plan:
    """确定性规则计划：没有模型时的兜底，也是模型计划的形状参照。"""
    text = (goal or "").lower()
    wanted: list[tuple[str, dict, str]] = [
        ("library.stats", {}, "先知道库里有什么，才知道能回答到什么程度"),
        ("library.search", {"query": goal, "top_k": 8}, "调研目标本身就是最好的检索式"),
    ]
    if any(w in text for w in _GRAPH_WORDS):
        wanted.append(("graph.build", {}, "目标涉及脉络与关系，需要引用图谱"))
        wanted.append(("graph.key_nodes", {"limit": 5}, "找出绕不开的核心工作"))
    if any(w in text for w in _FUTURE_WORDS):
        wanted.append(("future_work.scan", {"max_items": 4}, "目标是找空白，需要知道哪些待办已被解决"))
    if any(w in text for w in _SURVEY_WORDS):
        wanted.append(("survey.generate", {"topic": goal}, "目标要求横向梳理，走综述生成"))
    if any(w in text for w in _OUTLINE_WORDS):
        wanted.append(("writing.outline", {"idea": goal}, "目标指向写作，先生成带无幻觉引用的框架"))
    if any(w in text for w in _DISCOVER_WORDS):
        wanted.append(("discover.search", {"query": goal, "limit": 6}, "目标关心外部进展，需要出库检索"))

    steps: list[Step] = []
    seen: set[tuple] = set()
    for tool, args, why in wanted:
        if len(steps) >= max_steps:
            break
        if registry.get(tool) is None:
            continue
        if tool in {"library.search", "graph.build", "graph.key_nodes", "future_work.scan", "survey.generate", "writing.outline"} and doc_ids:
            args = {**args, "doc_ids": list(doc_ids)}
        step = Step(index=len(steps), tool=tool, args=args, why=why, label=_label_for(tool, args), source=source)
        if step.key() in seen:
            continue
        seen.add(step.key())
        steps.append(step)
    return Plan(steps=steps, source=source, stop_when="拿到足以支撑结论的原文证据即可收尾")


def _coerce_steps(raw: Any, *, registry: Any, doc_ids: Optional[list[str]], max_steps: int, source: str) -> list[Step]:
    steps: list[Step] = []
    seen: set[tuple] = set()
    for item in list(raw or [])[: max_steps * 2]:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("name") or "").strip()
        if registry.get(tool) is None:
            continue
        args = item.get("args") or item.get("arguments") or {}
        args = dict(args) if isinstance(args, dict) else {}
        if doc_ids and tool.startswith(("library.", "graph.", "future_work.", "survey.", "writing.")):
            args.setdefault("doc_ids", list(doc_ids))
        step = Step(
            index=len(steps),
            tool=tool,
            args=args,
            why=clip(item.get("why") or item.get("reason") or "", 160),
            label=clip(item.get("label") or _label_for(tool, args), 60),
            source=source,
        )
        if step.key() in seen:
            continue
        seen.add(step.key())
        steps.append(step)
        if len(steps) >= max_steps:
            break
    return steps


def plan(
    goal: str,
    *,
    library: Any,
    registry: Any,
    gateway: Any = None,
    doc_ids: Optional[list[str]] = None,
    max_steps: int = 6,
) -> Plan:
    """优先让模型规划；模型不可用或计划不可解析时，退到确定性计划。"""
    if gateway is not None and getattr(gateway, "available", False):
        try:
            data = gateway.json(
                [
                    {"role": "system", "content": prompts.AGENT_PLANNER_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_agent_planner_user(
                            goal, registry.catalog(), library_summary(library), doc_ids or []
                        ),
                    },
                ],
                model=None,
                max_tokens=900,
                kind="agent.plan",
            )
            steps = _coerce_steps(
                (data or {}).get("steps"), registry=registry, doc_ids=doc_ids, max_steps=max_steps, source="llm"
            )
            if steps:
                return Plan(
                    steps=steps,
                    source="llm",
                    stop_when=clip((data or {}).get("stop_when") or "", 160),
                    note=clip((data or {}).get("note") or "", 160),
                )
        except Exception as exc:  # noqa: BLE001
            fallback = offline_plan(goal, library=library, registry=registry, doc_ids=doc_ids, max_steps=max_steps)
            fallback.note = f"模型规划不可用（{type(exc).__name__}），已退到确定性计划"
            return fallback
    return offline_plan(goal, library=library, registry=registry, doc_ids=doc_ids, max_steps=max_steps)


def replan(
    goal: str,
    *,
    memory: WorkingMemory,
    registry: Any,
    gateway: Any = None,
    doc_ids: Optional[list[str]] = None,
    already: Optional[set[tuple]] = None,
    max_steps: int = 2,
) -> Optional[Plan]:
    """看一眼已有观察，判断还缺什么。模型不可用就返回 None（不硬凑）。"""
    if gateway is None or not getattr(gateway, "available", False):
        return None
    try:
        data = gateway.json(
            [
                {"role": "system", "content": prompts.AGENT_REPLAN_SYSTEM},
                {
                    "role": "user",
                    "content": prompts.build_agent_replan_user(goal, registry.catalog(), memory.brief()),
                },
            ],
            model=None,
            max_tokens=600,
            kind="agent.replan",
        )
    except Exception:  # noqa: BLE001
        return None
    steps = _coerce_steps(
        (data or {}).get("steps"), registry=registry, doc_ids=doc_ids, max_steps=max_steps, source="llm-replan"
    )
    blocked = already or set()
    steps = [s for s in steps if s.key() not in blocked]
    if not steps:
        return None
    for i, step in enumerate(steps):
        step.index = i
    return Plan(
        steps=steps,
        source="llm-replan",
        stop_when=clip((data or {}).get("stop_when") or "", 160),
        replanned=True,
    )

"""Agent 主循环。

    规划 → 调工具 → 记观察 →（必要时）再规划 → 收尾成可溯源的答案

三条纪律：

1. **不做黑盒**：每一步的 tool / 参数 / 耗时 / 结果都通过 SSE 推给前端，谁都能审。
2. **有预算**：步骤数、总耗时、记忆字符数都设上限。超了就带着已有证据收尾，
   并在结果里说明「因为预算收尾」，而不是假装调研做完了。
3. **收尾仍要过闸门**：Agent 的新增能力是「会去找」，不是「答案免检」。
   最终答案走的是同一个 ScholarEngine，逐句主张照旧过主张闸门。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from .memory import Observation, WorkingMemory, clip
from .planner import Plan, Step, plan as make_plan, replan as make_replan


@dataclass
class AgentRun:
    goal: str
    plan: dict = field(default_factory=dict)
    observations: list[dict] = field(default_factory=list)
    findings: dict = field(default_factory=dict)
    answer: Any = None
    memory: dict = field(default_factory=dict)
    doc_ids: list[str] = field(default_factory=list)
    refused: bool = False
    stopped_early: bool = False
    stop_reason: str = ""
    ms: int = 0
    usage: dict = field(default_factory=dict)

    def summary(self) -> dict:
        claims = list(getattr(self.answer, "claims", []) or [])
        verdicts = {"allow": 0, "downgrade": 0, "block": 0}
        for claim in claims:
            decision = getattr(claim, "decision", None)
            verdict = getattr(getattr(decision, "verdict", None), "value", None)
            if verdict in verdicts:
                verdicts[verdict] += 1
        return {
            "goal": self.goal,
            "plan": self.plan,
            "observations": self.observations,
            "findings": self.findings,
            "memory": self.memory,
            "doc_ids": self.doc_ids,
            "text": getattr(self.answer, "text", "") or "",
            "refused": self.refused,
            "support_rate": getattr(self.answer, "support_rate", None),
            "claim_counts": verdicts,
            "claims": [c.to_dict() if hasattr(c, "to_dict") else {} for c in claims],
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "ms": self.ms,
            "usage": self.usage,
        }


class ResearchAgent:
    def __init__(
        self,
        *,
        engine: Any,
        library: Any,
        registry: Any,
        gateway: Any = None,
        memory_budget: int = 8000,
        max_steps: int = 6,
        max_replan_steps: int = 2,
        max_total_ms: int = 240_000,
    ) -> None:
        self.engine = engine
        self.library = library
        self.registry = registry
        self.gateway = gateway
        self.memory_budget = memory_budget
        self.max_steps = max_steps
        self.max_replan_steps = max_replan_steps
        self.max_total_ms = max_total_ms

    # ------------------------------------------------------------------ 主循环
    def run(
        self,
        goal: str,
        *,
        doc_ids: Optional[list[str]] = None,
        mode: str = "auto",
        on_event: Optional[Callable[[str, dict], None]] = None,
        stream_claims: bool = True,
    ) -> AgentRun:
        started = time.time()
        goal = (goal or "").strip()
        doc_ids = [d for d in (doc_ids or []) if d]
        run = AgentRun(goal=goal)
        memory = WorkingMemory(goal, budget_chars=self.memory_budget)

        def emit(kind: str, payload: dict) -> None:
            if on_event is not None:
                on_event(kind, payload)

        if not goal:
            emit("error", {"kind": "empty_goal", "message": "先说清楚要调研什么"})
            run.stop_reason = "目标为空"
            run.ms = int((time.time() - started) * 1000)
            return run

        current = make_plan(
            goal, library=self.library, registry=self.registry,
            gateway=self.gateway, doc_ids=doc_ids, max_steps=self.max_steps,
        )
        emit("plan", {
            "goal": goal,
            "source": current.source,
            "stop_when": current.stop_when,
            "steps": [s.to_dict() for s in current.steps],
            "note": current.note,
            "catalog": self.registry.catalog(),
        })

        executed: set[tuple] = set()
        planned: list[Step] = []
        sources: list[str] = [current.source]
        step_index = 0
        rounds = 0
        while current is not None and current.steps:
            rounds += 1
            for step in current.steps:
                planned.append(step)
                if int((time.time() - started) * 1000) > self.max_total_ms:
                    run.stopped_early = True
                    run.stop_reason = f"总耗时超过 {self.max_total_ms // 1000} 秒预算，带着已有证据收尾"
                    break
                if step_index >= self.max_steps + self.max_replan_steps:
                    run.stopped_early = True
                    run.stop_reason = "步骤数达到上限，带着已有证据收尾"
                    break
                if step.key() in executed:
                    continue
                executed.add(step.key())
                step_index += 1
                emit("step", {
                    "step": step_index,
                    "tool": step.tool,
                    "status": "running",
                    "label": step.label,
                    "why": step.why,
                    "args": step.args,
                    "source": step.source,
                    "round": rounds,
                })
                emit("tool", {
                    "step": step_index,
                    "name": step.tool,
                    "detail": step.why or step.label,
                    "args": step.args,
                })
                result = self.registry.run(step.tool, **step.args)
                observation = Observation(
                    step=step_index,
                    tool=result.name,
                    ok=result.ok,
                    summary=result.summary,
                    payload=result.payload,
                    highlights=result.highlights,
                    evidence=result.evidence,
                    ms=result.ms,
                )
                memory.add(observation)
                run.observations.append(observation.to_dict())
                emit("observation", {
                    **observation.to_dict(),
                    "highlights": observation.highlights,
                    "memory": memory.stats(),
                })
                emit("step", {
                    "step": step_index,
                    "tool": step.tool,
                    "status": "done" if result.ok else "failed",
                    "label": step.label,
                    "why": step.why,
                    "ms": result.ms,
                    "ok": result.ok,
                    "summary": result.summary,
                    "round": rounds,
                })
            if run.stopped_early:
                break
            current = make_replan(
                goal, memory=memory, registry=self.registry, gateway=self.gateway,
                doc_ids=doc_ids, already=executed, max_steps=self.max_replan_steps,
            )
            if current is not None:
                sources.append(current.source)
                emit("plan", {
                    "goal": goal,
                    "source": current.source,
                    "stop_when": current.stop_when,
                    "steps": [s.to_dict() for s in current.steps],
                    "note": "根据已有观察追加的步骤",
                    "replanned": True,
                })

        run.plan = {
            "rounds": rounds,
            "sources": sources,
            "steps": [s.to_dict() for s in planned],
            "catalog": self.registry.names(),
        }
        run.findings = self._findings(memory)
        run.memory = memory.stats()

        answer = self._synthesize(goal, memory, doc_ids, mode=mode, run=run)
        run.doc_ids = list(run.doc_ids)
        run.answer = answer
        run.refused = bool(getattr(answer, "refused", False))
        claims = list(getattr(answer, "claims", []) or [])
        if stream_claims:
            for claim in claims:
                emit("claim", claim.to_dict() if hasattr(claim, "to_dict") else {})
        run.ms = int((time.time() - started) * 1000)
        run.usage = self._usage()
        emit("done", run.summary())
        return run

    # ------------------------------------------------------------------ 收尾
    def _synthesize(
        self,
        goal: str,
        memory: WorkingMemory,
        doc_ids: list[str],
        *,
        mode: str,
        run: AgentRun,
    ) -> Any:
        scope = list(doc_ids)
        if not scope:
            # Agent 自己找到了相关文献，就把最终答案收窄到这几篇上 —— 这比全库乱撞准得多
            scope = memory.relevant_doc_ids(limit=4)
        run.doc_ids = scope
        try:
            return self.engine.answer(goal, doc_ids=scope or None, mode=mode)
        except Exception as exc:  # noqa: BLE001
            run.stopped_early = True
            run.stop_reason = f"收尾时出错（{type(exc).__name__}：{clip(exc, 120)}）"
            return None

    def _findings(self, memory: WorkingMemory) -> dict:
        findings: dict = {"evidence": memory.evidence_pool()[:20]}
        for obs in memory.observations:
            payload = obs.payload or {}
            if obs.tool == "graph.build" and payload.get("roles"):
                findings["graph_roles"] = payload["roles"]
                findings["graph_insights"] = payload.get("insights", [])[:8]
            if obs.tool == "graph.key_nodes" and payload.get("nodes"):
                findings["key_nodes"] = payload["nodes"][:8]
            if obs.tool == "future_work.scan" and payload.get("items"):
                findings["future_work"] = payload["items"][:8]
                findings["future_work_status"] = payload.get("status_counts", {})
            if obs.tool == "survey.generate" and payload.get("claims"):
                findings["survey_claims"] = payload["claims"][:12]
            if obs.tool == "writing.outline" and payload.get("sections"):
                findings["outline_sections"] = payload["sections"][:12]
            if obs.tool == "discover.search" and payload.get("candidates"):
                findings["external_candidates"] = payload["candidates"][:8]
            if obs.tool == "library.stats":
                findings["library"] = {
                    "papers": len(payload.get("papers") or []),
                    "total_pages": payload.get("total_pages"),
                    "total_chunks": payload.get("total_chunks"),
                }
        return findings

    def _usage(self) -> dict:
        usage = getattr(self.gateway, "usage", None)
        if usage is None or not hasattr(usage, "summary"):
            return {}
        try:
            return dict(usage.summary())
        except Exception:  # noqa: BLE001
            return {}


def stream_agent(
    agent: ResearchAgent,
    goal: str,
    *,
    doc_ids: Optional[list[str]] = None,
    mode: str = "auto",
) -> Iterator[tuple[str, dict]]:
    """把 Agent 的事件流变成 SSE 生成器。事件在产生时就推出去，不攒到最后。"""
    queue: list[tuple[str, dict]] = []

    def on_event(kind: str, payload: dict) -> None:
        queue.append((kind, payload))

    import threading

    done_flag = threading.Event()

    def worker() -> None:
        try:
            agent.run(goal, doc_ids=doc_ids, mode=mode, on_event=on_event)
        finally:
            done_flag.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while not done_flag.is_set() or queue:
        if queue:
            yield queue.pop(0)
        else:
            time.sleep(0.05)
    thread.join(timeout=5)

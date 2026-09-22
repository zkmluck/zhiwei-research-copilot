"""Agent 层的离线测试。

要证明的不是「Agent 能跑」，而是这三件事：

1. 规划器不会发明工具（名册里没有的一律丢掉）；
2. 工具执行失败不会炸掉流程，而是变成一条可读的失败观察；
3. 记忆超预算会真的折叠，且证据池会去重 —— 上下文不会无上限膨胀。

全部离线，不需要任何 API key。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Optional

from zhiwei.agents.memory import Observation, WorkingMemory
from zhiwei.agents.orchestrator import ResearchAgent
from zhiwei.agents.planner import offline_plan, plan as make_plan
from zhiwei.agents.tools import ToolRegistry


class StubGateway:
    """假的模型网关：available=False 时等价于「没配 key」。"""

    def __init__(self, available: bool = False, reply: Optional[dict] = None) -> None:
        self.available = available
        self.reply = reply
        self.calls: list[str] = []
        self.usage = self

    def json(self, messages, **kwargs: Any) -> Any:
        self.calls.append(kwargs.get("kind", "?"))
        if self.reply is None:
            raise RuntimeError("no reply configured")
        if isinstance(self.reply, list):
            return self.reply.pop(0)
        return self.reply

    def summary(self) -> dict:
        return {"calls": len(self.calls)}


@dataclass
class StubPaper:
    doc_id: str
    title: str
    year: Optional[int] = 2020
    version: int = 1
    page_count: int = 10
    arxiv_id: str = ""
    is_latest: bool = True


class StubLibrary:
    def __init__(self, papers: Optional[list[StubPaper]] = None) -> None:
        self.papers = papers or [StubPaper("doc-1", "Attention Is All You Need", 2017)]

    def list_papers(self) -> list[StubPaper]:
        return list(self.papers)

    def load_chunks(self, doc_id: str) -> list:
        return [1, 2, 3]


class StubClaim:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_dict(self) -> dict:
        return {"text": self.text, "verdict": "allow"}


class StubAnswer:
    def __init__(self, text: str = "答案", claims: Optional[list] = None) -> None:
        self.text = text
        self.claims = claims if claims is not None else [StubClaim("基础模型用 8 张 P100 训练 12 小时。")]
        self.refused = False
        self.refusal_reason = ""
        self.support_rate = 1.0


class StubEngine:
    def __init__(self, answer: Optional[StubAnswer] = None) -> None:
        self.answer_obj = answer or StubAnswer()
        self.calls: list[tuple[str, Any]] = []

    def answer(self, question: str, *, doc_ids=None, mode: str = "auto", **_: Any) -> StubAnswer:
        self.calls.append((question, doc_ids))
        return self.answer_obj

    def retrieve(self, question: str, *, doc_ids=None, top_k: int = 8, **_: Any) -> list:
        return []


def make_registry(**handlers: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for name, handler in handlers.items():
        tool_name = name.replace("__", ".")
        registry.register(tool_name, f"{tool_name} 的测试替身", {}, handler)
    return registry


class TestToolRegistry(unittest.TestCase):
    def test_unknown_tool_is_reported_not_raised(self):
        registry = make_registry()
        result = registry.run("does.not.exist")
        self.assertFalse(result.ok)
        self.assertIn("名册里没有", result.summary)

    def test_handler_exception_becomes_failed_observation(self):
        def boom(**_: Any) -> dict:
            raise ValueError("网络不通")

        registry = make_registry(boom=boom)
        result = registry.run("boom")
        self.assertFalse(result.ok)
        self.assertIn("ValueError", result.error)
        self.assertIn("网络不通", result.summary)

    def test_reserved_keys_are_unpacked(self):
        def tool(**_: Any) -> dict:
            return {
                "summary": "干完了",
                "highlights": ["要点一"],
                "evidence": [{"doc_id": "d", "page": 1, "quote": "q"}],
                "extra": {"k": "v"},
            }

        registry = make_registry(tool=tool)
        result = registry.run("tool")
        self.assertTrue(result.ok)
        self.assertEqual(result.summary, "干完了")
        self.assertEqual(result.highlights, ["要点一"])
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.payload, {"extra": {"k": "v"}})

    def test_huge_payload_is_shrunk(self):
        def heavy(**_: Any) -> dict:
            return {"summary": "大负载", "rows": [{"i": i, "text": "x" * 200} for i in range(200)]}

        registry = make_registry(heavy=heavy)
        registry.max_payload_chars = 800
        result = registry.run("heavy")
        self.assertTrue(result.ok)
        self.assertLessEqual(len(result.payload.get("rows", [])), 12)
        self.assertTrue(result.payload.get("_truncated"))


class TestWorkingMemory(unittest.TestCase):
    @staticmethod
    def obs(step: int, tool: str, quote: str = "q") -> Observation:
        return Observation(
            step=step, tool=tool, ok=True, summary=f"{tool} 第 {step} 步",
            payload={"filler": "x" * 400}, highlights=["h"],
            evidence=[{"doc_id": "d1", "page": 1, "quote": quote}], ms=10,
        )

    def test_budget_triggers_compression(self):
        memory = WorkingMemory("目标", budget_chars=1200, keep_per_tool=1)
        for i in range(6):
            memory.add(self.obs(i, "library.search", quote=f"q{i}"))
        self.assertGreater(memory.compressions, 0)
        self.assertGreater(len(memory.folded), 0)
        self.assertLess(len(memory.observations), 6)
        self.assertLessEqual(memory.chars, 1200)

    def test_evidence_pool_dedupes(self):
        memory = WorkingMemory("目标", budget_chars=100000)
        memory.add(self.obs(1, "library.search", quote="同一句"))
        memory.add(self.obs(2, "graph.build", quote="同一句"))
        self.assertEqual(len(memory.evidence_pool()), 1)

    def test_relevant_doc_ids_ranked_by_hits(self):
        memory = WorkingMemory("目标", budget_chars=100000)
        for step in range(3):
            observation = self.obs(step, "library.search", quote=f"q{step}")
            observation.evidence[0]["doc_id"] = "hot"
            memory.add(observation)
        other = self.obs(9, "library.search", quote="x")
        other.evidence[0]["doc_id"] = "cold"
        memory.add(other)
        self.assertEqual(memory.relevant_doc_ids()[0], "hot")

    def test_brief_mentions_folded(self):
        memory = WorkingMemory("目标", budget_chars=1200, keep_per_tool=1)
        for i in range(6):
            memory.add(self.obs(i, "library.search"))
        self.assertIn("折叠", memory.brief())


class TestPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = make_registry(
            library__stats=lambda **_: {},
            library__search=lambda **_: {},
            graph__build=lambda **_: {},
            future_work__scan=lambda **_: {},
            discover__search=lambda **_: {},
        )
        self.library = StubLibrary()

    def test_offline_plan_follows_keywords(self):
        planned = offline_plan(
            "这批论文的引用脉络和未来方向", library=self.library, registry=self.registry
        )
        tools = [s.tool for s in planned.steps]
        self.assertEqual(tools[0], "library.stats")
        self.assertIn("graph.build", tools)
        self.assertIn("future_work.scan", tools)
        self.assertNotIn("discover.search", tools)

    def test_offline_plan_always_starts_with_inventory(self):
        planned = offline_plan("随便问点什么", library=self.library, registry=self.registry)
        self.assertEqual([s.tool for s in planned.steps][:2], ["library.stats", "library.search"])

    def test_llm_plan_drops_unknown_tools(self):
        gateway = StubGateway(
            available=True,
            reply={
                "steps": [
                    {"tool": "library.stats", "why": "先看库存"},
                    {"tool": "magic.answer_everything", "why": "一步到位"},
                ]
            },
        )
        planned = make_plan("目标", library=self.library, registry=self.registry, gateway=gateway)
        self.assertEqual(planned.source, "llm")
        self.assertEqual([s.tool for s in planned.steps], ["library.stats"])

    def test_llm_plan_failure_falls_back_to_offline(self):
        gateway = StubGateway(available=True, reply=None)
        planned = make_plan("目标", library=self.library, registry=self.registry, gateway=gateway)
        self.assertEqual(planned.source, "offline")
        self.assertTrue(planned.steps)


class TestAgentLoop(unittest.TestCase):
    def build_agent(self, **handlers: Any) -> tuple[ResearchAgent, list[tuple[str, dict]], StubEngine]:
        registry = make_registry(**handlers)
        engine = StubEngine()
        agent = ResearchAgent(
            engine=engine, library=StubLibrary(), registry=registry, gateway=StubGateway(available=False)
        )
        events: list[tuple[str, dict]] = []
        return agent, events, engine

    def test_loop_emits_plan_tool_observation_and_done(self):
        def search(**_: Any) -> dict:
            return {
                "summary": "命中 2 条证据",
                "evidence": [
                    {"doc_id": "doc-1", "page": 7, "quote": "We trained on 8 P100 GPUs."},
                ],
            }

        agent, events, _engine = self.build_agent(library__stats=lambda **_: {"summary": "库里 1 篇"}, library__search=search)
        run = agent.run("训了多久", on_event=lambda kind, payload: events.append((kind, payload)))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "plan")
        self.assertIn("observation", kinds)
        self.assertIn("claim", kinds)
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(len(run.observations), 2)
        self.assertTrue(all(obs["ok"] for obs in run.observations))
        self.assertEqual(run.doc_ids, ["doc-1"])
        self.assertEqual(run.memory["evidence"], 1)

    def test_failed_tool_does_not_stop_the_run(self):
        def broken(**_: Any) -> dict:
            raise RuntimeError("外部接口超时")

        agent, events, _engine = self.build_agent(library__stats=broken, library__search=lambda **_: {"summary": "ok"})
        run = agent.run("目标", on_event=lambda kind, payload: events.append((kind, payload)))
        statuses = [payload["status"] for kind, payload in events if kind == "step"]
        self.assertIn("failed", statuses)
        self.assertIn("done", [k for k, _ in events])
        self.assertFalse(run.observations[0]["ok"])
        self.assertTrue(run.observations[1]["ok"])

    def test_steps_are_not_repeated(self):
        calls = {"n": 0}

        def counted(**_: Any) -> dict:
            calls["n"] += 1
            return {"summary": "一次就够"}

        registry = make_registry(library__stats=counted, library__search=counted)
        agent = ResearchAgent(
            engine=StubEngine(), library=StubLibrary(), registry=registry, gateway=StubGateway(available=True, reply={"steps": []})
        )
        agent.run("目标")
        self.assertEqual(calls["n"], 2)

    def test_replan_adds_steps_when_model_asks(self):
        def tool(**_: Any) -> dict:
            return {"summary": "ok"}

        gateway = StubGateway(
            available=True,
            reply=[
                {"steps": [{"tool": "library.stats", "why": "先看库存"}]},
                {"steps": [{"tool": "library.search", "args": {"query": "补一轮检索"}, "why": "补充证据"}]},
            ],
        )
        registry = make_registry(library__stats=tool, library__search=tool)
        agent = ResearchAgent(engine=StubEngine(), library=StubLibrary(), registry=registry, gateway=gateway)
        run = agent.run("目标")
        self.assertEqual([o["tool"] for o in run.observations], ["library.stats", "library.search"])
        self.assertIn("llm-replan", run.plan["sources"])

    def test_empty_goal_is_rejected_early(self):
        agent, events, _engine = self.build_agent(library__stats=lambda **_: {})
        run = agent.run("   ", on_event=lambda kind, payload: events.append((kind, payload)))
        self.assertEqual([k for k, _ in events], ["error"])
        self.assertEqual(run.observations, [])


if __name__ == "__main__":
    unittest.main()

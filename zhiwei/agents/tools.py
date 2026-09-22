"""Agent 的工具箱。

每个工具都是**真实调用**底层模块，不是占位符：检索真的走混合检索，图谱真的算中心性，
未来工作真的跑时效性判定。工具执行失败不会中断 Agent —— 失败本身也是一条观察，
会带着错误类型写进记忆，让规划器知道这条路走不通。

工具的返回值统一约定三个保留键：

* ``summary``     —— 一句话说清这次调用拿到了什么（给人看，也给规划器看）
* ``highlights``  —— 2-4 条要点（给前端卡片用）
* ``evidence``    —— 抓到的证据（带页码锚点），会进入 Agent 的证据池
其余键会作为结构化负载（payload）返回，过长会自动裁剪。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .memory import clip, jsonable


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, str]
    handler: Callable[..., dict]
    network: bool = False
    cost: str = "cheap"          # cheap / model / network

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "network": self.network,
            "cost": self.cost,
        }


@dataclass
class ToolResult:
    name: str
    ok: bool
    summary: str
    payload: dict = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    ms: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "tool": self.name,
            "ok": self.ok,
            "summary": self.summary,
            "highlights": list(self.highlights),
            "ms": self.ms,
            "error": self.error,
        }


def evidence_to_dict(evidence: Any) -> dict:
    """把 Evidence 契约对象压成前端与记忆都能用的字典。"""
    anchor = getattr(evidence, "anchor", None)
    doc_id = getattr(evidence, "doc_id", "")
    return {
        "doc_id": doc_id,
        "doc_title": getattr(evidence, "doc_title", "") or "",
        "page": int(getattr(anchor, "page", 0) or 0),
        "quote": clip(getattr(anchor, "quote", ""), 400),
        "char_start": int(getattr(anchor, "char_start", -1) or -1),
        "char_end": int(getattr(anchor, "char_end", -1) or -1),
        "chunk_id": getattr(evidence, "chunk_id", "") or "",
        "score": round(float(getattr(evidence, "score", 0.0) or 0.0), 4),
        "source": getattr(evidence, "source", "") or "",
    }


class ToolRegistry:
    """工具名册。执行一律不抛异常 —— 异常会被包装成失败的观察。"""

    def __init__(self, *, max_payload_chars: int = 4000) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.max_payload_chars = max_payload_chars

    # ---------------------------------------------------------------- 注册
    def register(
        self,
        name: str,
        description: str,
        params: dict[str, str],
        handler: Callable[..., dict],
        *,
        network: bool = False,
        cost: str = "cheap",
    ) -> None:
        self._tools[name] = ToolSpec(name, description, params, handler, network, cost)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self, *, include_network: bool = True) -> list[dict]:
        return [
            spec.to_dict()
            for spec in self._tools.values()
            if include_network or not spec.network
        ]

    @property
    def has_network_tools(self) -> bool:
        return any(spec.network for spec in self._tools.values())

    # ---------------------------------------------------------------- 执行
    def run(self, name: str, **kwargs: Any) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                name=name,
                ok=False,
                summary=f"名册里没有这个工具：{name}（可选：{'、'.join(self.names())}）",
                error="unknown_tool",
            )
        started = time.time()
        try:
            raw = spec.handler(**kwargs) or {}
            ok = True
            error = ""
        except Exception as exc:  # noqa: BLE001
            raw = {"summary": f"{name} 执行失败：{type(exc).__name__}: {clip(exc, 160)}"}
            ok = False
            error = f"{type(exc).__name__}: {clip(exc, 200)}"
        ms = int((time.time() - started) * 1000)
        return self._pack(name, raw, ok=ok, error=error, ms=ms)

    def _pack(self, name: str, raw: dict, *, ok: bool, error: str, ms: int) -> ToolResult:
        payload = {k: v for k, v in raw.items() if k not in {"summary", "highlights", "evidence"}}
        payload, truncated = self._shrink(payload)
        if truncated:
            payload["_truncated"] = truncated
        return ToolResult(
            name=name,
            ok=ok,
            summary=clip(raw.get("summary") or f"{name} 执行完成", 300),
            payload=jsonable(payload),
            highlights=[clip(h, 200) for h in list(raw.get("highlights") or [])][:4],
            evidence=[jsonable(e) for e in list(raw.get("evidence") or [])][:40],
            ms=ms,
            error=error,
        )

    def _shrink(self, payload: dict, *, limit: int = 200) -> tuple[dict, list[str]]:
        """负载过大时，优先丢掉最占地方的那几个字段，而不是整包扔掉。"""
        dropped: list[str] = []
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) <= self.max_payload_chars:
            return payload, dropped
        sizes = sorted(
            ((k, len(json.dumps(v, ensure_ascii=False, default=str))) for k, v in payload.items()),
            key=lambda kv: -kv[1],
        )
        current = payload
        for key, _size in sizes:
            if len(json.dumps(current, ensure_ascii=False, default=str)) <= self.max_payload_chars:
                break
            if key in {"papers", "nodes", "edges", "items", "claims", "roles"}:
                value = current[key]
                if isinstance(value, list):
                    current[key] = value[:12]
                    dropped.append(f"{key}:{len(value)}→12")
                    continue
            current.pop(key, None)
            dropped.append(key)
        return current, dropped


# ---------------------------------------------------------------------------
# 默认工具集：真实接线到解析 / 检索 / 图谱 / 发现 / 写作各层
# ---------------------------------------------------------------------------


def build_default_registry(
    *,
    library: Any,
    engine: Any,
    gateway: Any = None,
    data_dir: Any = None,
    gate: Any = None,
    graph_provider: Optional[Callable[[list[str]], tuple[Any, list, dict]]] = None,
    max_payload_chars: int = 4000,
) -> ToolRegistry:
    from ..graph.centrality import classify_roles
    from ..graph.future_work import scan_future_work
    from ..graph.survey import generate_survey
    from ..writing.outline import build_outline

    registry = ToolRegistry(max_payload_chars=max_payload_chars)

    def library_stats(**_: Any) -> dict:
        papers = library.list_papers()
        rows = []
        chunks = 0
        for paper in papers:
            try:
                chunks += len(library.load_chunks(paper.doc_id))
            except Exception:  # noqa: BLE001
                pass
            rows.append({
                "doc_id": paper.doc_id,
                "title": paper.title,
                "year": paper.year,
                "version": paper.version,
                "pages": paper.page_count,
                "arxiv_id": paper.arxiv_id,
                "is_latest": paper.is_latest,
            })
        total_pages = sum(int(p.page_count or 0) for p in papers)
        return {
            "summary": f"本地库共 {len(papers)} 篇文献、{total_pages} 页、{chunks} 个检索切块",
            "highlights": [f"{r['title'] or r['doc_id']}（{r['year'] or '?'}，v{r['version']}）" for r in rows[:3]],
            "papers": rows,
            "total_pages": total_pages,
            "total_chunks": chunks,
        }

    def library_search(query: str = "", top_k: int = 6, doc_ids: Optional[list[str]] = None, **_: Any) -> dict:
        query = str(query or "").strip()
        if not query:
            return {"summary": "检索词为空，没有可执行的检索", "highlights": []}
        hits = engine.retrieve(query, doc_ids=doc_ids or None, top_k=int(top_k))
        evidence = [evidence_to_dict(e) for e in hits]
        return {
            "summary": f"检索「{clip(query, 40)}」命中 {len(evidence)} 条带锚点的证据",
            "highlights": [f"p{e['page']}：{clip(e['quote'], 90)}" for e in evidence[:3]],
            "evidence": evidence,
            "query": query,
        }

    def _graph(doc_ids: Optional[list[str]] = None) -> tuple[Any, list, dict]:
        ids = [d for d in (doc_ids or []) if d] or [p.doc_id for p in library.list_papers()]
        if graph_provider is not None:
            return graph_provider(ids)
        from ..graph.citation import build_network
        from ..graph.centrality import analyze_network

        network = build_network(ids, library=library, data_dir=data_dir, gateway=gateway, classify=False)
        insights, metrics = analyze_network(network)
        return network, insights, metrics

    def graph_build(doc_ids: Optional[list[str]] = None, **_: Any) -> dict:
        network, insights, metrics = _graph(doc_ids)
        roles = classify_roles(insights)
        return {
            "summary": (
                f"引用图谱：{len(network.nodes)} 个节点、{len(network.edges)} 条边；"
                + "、".join(f"{k} {len(v)}" for k, v in roles.items())
            ),
            "highlights": [f"{i.title or i.doc_id}：{i.reason}" for i in insights[:3] if getattr(i, 'reason', '')],
            "nodes": [n.to_dict() if hasattr(n, "to_dict") else dict(n) for n in network.nodes],
            "edges": [e.to_dict() if hasattr(e, "to_dict") else dict(e) for e in network.edges],
            "roles": roles,
            "metrics": metrics,
            "insights": [
                {
                    "doc_id": i.doc_id,
                    "title": i.title,
                    "role": i.role,
                    "reason": i.reason,
                    "pagerank": round(float(i.pagerank), 5),
                    "betweenness": round(float(i.betweenness), 5),
                    "k_core": i.k_core,
                    "in_library": i.in_library,
                }
                for i in insights[:20]
            ],
        }

    def graph_key_nodes(doc_ids: Optional[list[str]] = None, limit: int = 5, **_: Any) -> dict:
        _network, insights, _metrics = _graph(doc_ids)
        ranked = sorted(insights, key=lambda i: -float(i.pagerank))[: int(limit)]
        return {
            "summary": f"挖出 {len(ranked)} 个核心节点（按 PageRank 排序）",
            "highlights": [f"{i.title or i.doc_id}（{i.role}）：{i.reason}" for i in ranked[:4]],
            "nodes": [
                {
                    "doc_id": i.doc_id,
                    "title": i.title,
                    "role": i.role,
                    "reason": i.reason,
                    "pagerank": round(float(i.pagerank), 5),
                    "in_library": i.in_library,
                }
                for i in ranked
            ],
        }

    def future_work_scan(doc_ids: Optional[list[str]] = None, max_items: int = 4, **_: Any) -> dict:
        ids = [d for d in (doc_ids or []) if d] or [p.doc_id for p in library.list_papers()]
        payload = scan_future_work(
            ids,
            library=library,
            data_dir=data_dir,
            gateway=gateway,
            max_items_per_doc=int(max_items),
        )
        items = list(payload.get("items") or [])
        by_status: dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "summary": f"扫出 {len(items)} 条 Future Work，状态分布：" + "、".join(f"{k} {v}" for k, v in by_status.items()),
            "highlights": [f"[{i.get('status')}] {clip(i.get('text'), 90)}" for i in items[:4]],
            "items": items,
            "status_counts": by_status,
        }

    def survey_generate(topic: str = "", doc_ids: Optional[list[str]] = None, **_: Any) -> dict:
        ids = [d for d in (doc_ids or []) if d] or [p.doc_id for p in library.list_papers()]
        topic = str(topic or "").strip() or "这批文献在解决什么问题"
        _network, insights, _metrics = _graph(ids)
        payload = generate_survey(
            topic, ids, library=library, insights=insights, data_dir=data_dir,
            gateway=gateway, engine=engine, gate=gate,
        )
        claims = list(payload.get("claims") or [])
        stats = payload.get("stats") or {}
        return {
            "summary": (
                f"生成综述草稿：{len(claims)} 条结论通过闸门"
                + (f"（拦下 {stats.get('blocked')} 条）" if stats.get("blocked") else "")
            ),
            "highlights": [clip(c.get("text"), 100) for c in claims[:3]],
            "claims": claims,
            "markdown": payload.get("markdown") or "",
            "stats": stats,
        }

    def outline_draft(idea: str = "", doc_ids: Optional[list[str]] = None, **_: Any) -> dict:
        ids = [d for d in (doc_ids or []) if d] or [p.doc_id for p in library.list_papers()]
        payload = build_outline(
            str(idea or "").strip(), doc_ids=ids, library=library, gateway=gateway, gate=gate, engine=engine
        )
        return {
            "summary": f"生成写作框架，{len(payload.get('sections') or [])} 个小节",
            "highlights": [str(s.get("title")) for s in (payload.get("sections") or [])[:4]],
            "sections": payload.get("sections") or [],
            "references": payload.get("references") or [],
        }

    def discover_search(query: str = "", limit: int = 6, **_: Any) -> dict:
        from ..discover.sources import search_all

        query = str(query or "").strip()
        if not query:
            return {"summary": "检索词为空", "highlights": []}
        found = search_all(query, limit=int(limit))
        rows = [
            {
                "title": c.title,
                "year": c.year,
                "venue": c.venue,
                "arxiv_id": c.arxiv_id,
                "url": c.url,
                "has_code": getattr(c, "has_code", None),
            }
            for c in found
        ]
        return {
            "summary": f"外部检索「{clip(query, 40)}」找到 {len(rows)} 篇候选",
            "highlights": [f"{r['title']}（{r['year'] or '?'}）" for r in rows[:4]],
            "candidates": rows,
        }

    registry.register(
        "library.stats",
        "看清本地文献库里有什么：篇数、页数、切块数、版本情况。任何调研都该先跑这一步。",
        {},
        library_stats,
    )
    registry.register(
        "library.search",
        "在本地文献库里做混合检索（BM25 + 向量 + RRF），返回带页码锚点的原文证据。",
        {"query": "检索词，用自然语言描述要找什么", "top_k": "返回条数，默认 6"},
        library_search,
        cost="model",
    )
    registry.register(
        "graph.build",
        "构建这批文献的引用图谱：节点、边、四种角色（基石 / 桥接 / 衍生 / 孤岛）。离线启发式分类，不调用模型。",
        {"doc_ids": "只看这几篇，留空表示全库"},
        graph_build,
    )
    registry.register(
        "graph.key_nodes",
        "从引用图谱里挖出最核心的几个节点，并给出「为什么它是核心」的人话理由。",
        {"limit": "返回条数，默认 5"},
        graph_key_nodes,
    )
    registry.register(
        "future_work.scan",
        "抽取各篇论文的 Future Work，并判定它是否已经被后续工作解决（回答「这个题还能不能做」）。",
        {"max_items": "每篇最多几条，默认 4"},
        future_work_scan,
        cost="model",
    )
    registry.register(
        "survey.generate",
        "沿引用图谱生成领域综述草稿，每一句都要过主张闸门。",
        {"topic": "综述主题", "doc_ids": "只看这几篇，留空表示全库"},
        survey_generate,
        cost="model",
    )
    registry.register(
        "writing.outline",
        "从 Idea 生成论文框架，References 只能来自本地库（模型碰不到引用格式）。",
        {"idea": "论文想法", "doc_ids": "可用文献，留空表示全库"},
        outline_draft,
        cost="model",
    )
    registry.register(
        "discover.search",
        "去外部学术源（arXiv / OpenAlex / Semantic Scholar）检索新论文。需要外网。",
        {"query": "检索词", "limit": "返回条数，默认 6"},
        discover_search,
        network=True,
        cost="network",
    )
    return registry

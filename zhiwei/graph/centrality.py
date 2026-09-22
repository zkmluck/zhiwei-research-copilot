"""图论挖掘：谁是这个领域的基石，谁只是边缘衍生。

赛题原话是「运用图论算法计算并挖掘『基石论文』与『边缘衍生论文』」。
我们做三件事，并且**每种判定都要给出人话理由**（可解释性比分数重要）：

  1. PageRank（带引用含金量加权）—— 被多少「真有分量的引用」指向；
  2. Betweenness —— 跨子领域的桥接节点；
  3. k-core 分解 —— 处在网络核心层还是边缘层。

角色判定把三者与入度、年份结合起来。所有阈值都是分位数，不写死绝对值，
这样 20 篇的局域图和 200 篇的图用同一套规则都能给出合理结果。
"""

from __future__ import annotations

from statistics import median

import networkx as nx

from ..contracts import CitationEdge, GraphInsight
from .citation import CitationNetwork


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _build_graph(network: CitationNetwork) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, attrs in network.nodes.items():
        graph.add_node(node_id, **attrs)
    for edge in network.edges:
        target = edge.dst_doc_id or _edge_target_id(edge)
        if not target or target == edge.src_doc_id:
            continue
        weight = max(0.05, edge.value_score or 0.2)
        if graph.has_edge(edge.src_doc_id, target):
            graph[edge.src_doc_id][target]["weight"] += weight
            graph[edge.src_doc_id][target]["count"] += 1
        else:
            graph.add_edge(edge.src_doc_id, target, weight=weight, count=1, intent=edge.intent.value)
    return graph


def _edge_target_id(edge: CitationEdge) -> str:
    """边指向外部文献时，用与 citation 层一致的规则还原节点 id。"""
    if edge.dst_doc_id:
        return edge.dst_doc_id
    if edge.dst_title:
        key = "".join(ch if ch.isalnum() or "\u4e00" <= ch <= "\u9fff" else "-" for ch in edge.dst_title.lower())[:60]
        return f"ext:{key}"
    return ""


def analyze_network(
    network: CitationNetwork,
    *,
    max_nodes_for_exact_betweenness: int = 200,
) -> tuple[list[GraphInsight], dict]:
    """返回 (每个节点的洞察, 图级别指标)。"""
    graph = _build_graph(network)
    if graph.number_of_nodes() == 0:
        return [], {"nodes": 0, "edges": 0, "density": 0.0, "components": 0, "core_size": 0}

    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except Exception:
        pagerank = {n: 0.0 for n in graph.nodes}

    if graph.number_of_nodes() <= max_nodes_for_exact_betweenness:
        betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)
    else:
        # 大图用 k 采样近似，避免评审环境里 500 节点算到超时
        k = min(80, graph.number_of_nodes())
        betweenness = nx.betweenness_centrality(graph, k=k, normalized=True, seed=7)

    undirected = graph.to_undirected()
    try:
        core_number = nx.core_number(undirected)
    except Exception:
        core_number = {n: 0 for n in graph.nodes}

    pr_values = [pagerank.get(n, 0.0) for n in graph.nodes]
    bt_values = [betweenness.get(n, 0.0) for n in graph.nodes]
    pr_high = _quantile(pr_values, 0.75)
    bt_high = _quantile(bt_values, 0.75)
    pr_mid = median(pr_values) if pr_values else 0.0

    max_core = max(core_number.values()) if core_number else 0
    years = [graph.nodes[n].get("year") for n in graph.nodes if graph.nodes[n].get("year")]
    newest_year = max(years) if years else None

    insights: list[GraphInsight] = []
    for node_id in graph.nodes:
        attrs = graph.nodes[node_id]
        in_deg = graph.in_degree(node_id)
        out_deg = graph.out_degree(node_id)
        pr = pagerank.get(node_id, 0.0)
        bt = betweenness.get(node_id, 0.0)
        core = core_number.get(node_id, 0)
        year = attrs.get("year")

        role = "peripheral"
        reason_parts: list[str] = []

        if in_deg == 0 and out_deg == 0:
            role = "isolated"
            reason_parts.append("在该局域网络中没有任何引用关系（可能是解析或元数据缺失）")
        elif pr >= pr_high and in_deg >= 2:
            role = "cornerstone"
            reason_parts.append(
                f"PageRank {pr:.4f} 位居前 25%，被 {in_deg} 篇文献引用，处在网络核心层（k-core={core}）"
            )
        elif bt >= bt_high and (in_deg + out_deg) >= 2:
            role = "bridge"
            reason_parts.append(
                f"中介中心性 {bt:.4f} 位居前 25%，连接不同引用簇，是跨方向的桥接工作"
            )
        elif pr <= pr_mid and in_deg <= 1:
            role = "derivative"
            if newest_year and year and year >= newest_year - 1:
                reason_parts.append(
                    f"入度仅 {in_deg} 且发表于 {year}（接近最新年份 {newest_year}），属于新近的边缘衍生工作"
                )
            else:
                reason_parts.append(f"入度仅 {in_deg} 且 PageRank 低于中位数，属于边缘节点")

        insights.append(
            GraphInsight(
                doc_id=node_id,
                title=attrs.get("title", ""),
                year=year,
                pagerank=round(pr, 6),
                betweenness=round(bt, 6),
                in_degree=in_deg,
                out_degree=out_deg,
                k_core=core,
                role=role,
                reason="；".join(reason_parts) or "各项指标接近网络中位水平",
                in_library=bool(attrs.get("in_library")),
            )
        )

    metrics = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 6),
        "components": nx.number_weakly_connected_components(graph),
        "core_size": max_core,
        "pagerank_p75": round(pr_high, 6),
        "betweenness_p75": round(bt_high, 6),
        "role_counts": _role_counts(insights),
    }
    return insights, metrics


def _role_counts(insights: list[GraphInsight]) -> dict:
    counts: dict[str, int] = {}
    for ins in insights:
        counts[ins.role] = counts.get(ins.role, 0) + 1
    return counts


def classify_roles(insights: list[GraphInsight]) -> dict[str, list[str]]:
    """按角色分组，便于前端直接渲染。"""
    grouped: dict[str, list[str]] = {}
    for ins in insights:
        grouped.setdefault(ins.role, []).append(ins.doc_id)
    return grouped


def prune_non_substantive(network: CitationNetwork, threshold: float = 0.5) -> CitationNetwork:
    """剔除「应付式引用」后重建一张干净的网络（用于对比视图）。

    这是赛题「自动剔除应付式引用」的直接落地：把 lists/background 类引用去掉后，
    网络的骨架会明显更清晰。
    """
    kept = [e for e in network.edges if e.substantive and e.value_score >= threshold]
    if not kept:
        return CitationNetwork(nodes=dict(network.nodes), edges=[], stats=dict(network.stats))
    alive: set[str] = set()
    for e in kept:
        alive.add(e.src_doc_id)
        target = e.dst_doc_id or _edge_target_id(e)
        if target:
            alive.add(target)
    nodes = {k: v for k, v in network.nodes.items() if k in alive}
    stats = dict(network.stats)
    stats["edges"] = len(kept)
    stats["nodes"] = len(nodes)
    stats["pruned_from"] = len(network.edges)
    return CitationNetwork(nodes=nodes, edges=kept, stats=stats, warnings=list(network.warnings))
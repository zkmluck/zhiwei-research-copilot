"""架构图脚本生成：Mermaid / Graphviz / TikZ。

赛题要求能生成这三种格式的拓扑图脚本。这里刻意分成两件事：

  - `build_diagram(kind, spec)`：把**结构化描述**转成脚本。因为输入是结构化的，
    输出就是确定性的，不会出现"图里少了一个节点"这种幻觉；
  - `diagram_from_network(network)`：直接从引用图谱长出一张图，
    这样"图谱"和"论文里的架构图"是同一份数据的两个视图。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_SAFE_ID = re.compile(r"[^0-9A-Za-z_\u4e00-\u9fff]+")


def _safe(node_id: str, prefix: str = "n") -> str:
    cleaned = _SAFE_ID.sub("_", str(node_id)).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:60]


def _escape_label(text: str, limit: int = 40) -> str:
    return (text or "").replace("\n", " ").strip()[:limit]


def normalize_spec(spec: Any) -> dict:
    """接受字典或 JSON 字符串，统一成 {title, nodes, edges}。"""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError:
            spec = {"title": spec, "nodes": [], "edges": []}
    spec = spec or {}
    nodes = []
    for i, node in enumerate(spec.get("nodes") or []):
        if isinstance(node, str):
            nodes.append({"id": node, "label": node, "group": ""})
        elif isinstance(node, dict):
            nid = str(node.get("id") or node.get("label") or f"n{i}")
            nodes.append(
                {
                    "id": nid,
                    "label": str(node.get("label") or nid),
                    "group": str(node.get("group") or ""),
                }
            )
    edges = []
    for edge in spec.get("edges") or []:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append({"source": str(edge[0]), "target": str(edge[1]), "label": ""})
        elif isinstance(edge, dict):
            edges.append(
                {
                    "source": str(edge.get("source") or edge.get("from") or ""),
                    "target": str(edge.get("target") or edge.get("to") or ""),
                    "label": str(edge.get("label") or ""),
                }
            )
    return {"title": str(spec.get("title") or "系统架构"), "nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# Mermaid
# --------------------------------------------------------------------------


def to_mermaid(spec: dict) -> str:
    spec = normalize_spec(spec)
    lines = ["%% ```mermaid", f"%% {spec['title']}", "flowchart LR"]
    for node in spec["nodes"]:
        nid = _safe(node["id"])
        label = _escape_label(node["label"])
        if node["group"]:
            lines.append(f"    {nid}[\"{label}\"]:::{_safe(node['group'], 'g')}")
        else:
            lines.append(f"    {nid}[\"{label}\"]")
    for edge in spec["edges"]:
        src, dst = _safe(edge["source"]), _safe(edge["target"])
        if edge["label"]:
            lines.append(f"    {src} -->|\"{_escape_label(edge['label'], 20)}\"| {dst}")
        else:
            lines.append(f"    {src} --> {dst}")
    groups = {}
    for node in spec["nodes"]:
        if node["group"]:
            groups.setdefault(node["group"], []).append(_safe(node["id"]))
    for group, members in groups.items():
        lines.append(f"    classDef {_safe(group, 'g')} fill:#1f2a37,stroke:#5b7f9a,color:#e8eef3")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Graphviz DOT
# --------------------------------------------------------------------------


def to_graphviz(spec: dict) -> str:
    spec = normalize_spec(spec)
    lines = [
        "digraph architecture {",
        '    rankdir="LR";',
        '    graph [bgcolor="transparent", fontname="Helvetica"];',
        '    node [shape="box", style="rounded,filled", fillcolor="#1f2a37", '
        'fontcolor="#e8eef3", color="#5b7f9a", fontname="Helvetica"];',
        '    edge [color="#7d8fa1", fontcolor="#c9d4de", fontname="Helvetica"];',
        f'    label="{_escape_label(spec["title"], 60)}";',
    ]
    for node in spec["nodes"]:
        colour = ""
        if node["group"]:
            palette = {
                "input": "#1d3b4d", "core": "#26414f", "output": "#3d3a2a",
                "agent": "#25384a", "tool": "#2f3a2c",
            }
            colour = f', fillcolor="{palette.get(node["group"], "#1f2a37")}"'
        lines.append(f'    "{_safe(node["id"])}" [label="{_escape_label(node["label"])}"{colour}];')
    for edge in spec["edges"]:
        src, dst = _safe(edge["source"]), _safe(edge["target"])
        if edge["label"]:
            lines.append(f'    "{src}" -> "{dst}" [label="{_escape_label(edge["label"], 24)}"];')
        else:
            lines.append(f'    "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# TikZ
# --------------------------------------------------------------------------


def to_tikz(spec: dict) -> str:
    spec = normalize_spec(spec)
    nodes = spec["nodes"][:24]
    positions = _layout(len(nodes))
    lines = [
        "% 需要 \\usepackage{tikz} 与 \\usetikzlibrary{arrows.meta,positioning}",
        "\\begin{tikzpicture}[",
        "  node distance=1.2cm,",
        "  box/.style={draw=black!55, rounded corners=2pt, fill=black!4,",
        "               minimum width=2.6cm, minimum height=0.85cm, align=center, font=\\small},",
        "  arr/.style={-{Latex[length=2mm]}, draw=black!60, thick}",
        "]",
    ]
    id_map: dict[str, str] = {}
    for i, node in enumerate(nodes):
        key = _safe(node["id"], "t")
        id_map[node["id"]] = key
        x, y = positions[i]
        lines.append(f"  \\node[box] ({key}) at ({x:.2f},{y:.2f}) {{{_escape_label(node['label'], 28)}}};")
    for edge in spec["edges"]:
        src = id_map.get(edge["source"])
        dst = id_map.get(edge["target"])
        if not src or not dst:
            continue
        label = f" node[midway, above, font=\\scriptsize] {{{_escape_label(edge['label'], 16)}}}" if edge["label"] else ""
        lines.append(f"  \\draw[arr] ({src}) --{label} ({dst});")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def _layout(count: int, columns: int = 4, dx: float = 3.2, dy: float = 1.6) -> list[tuple[float, float]]:
    """极简网格布局：保证生成的 TikZ 打开就能编译，不需要额外布局库。"""
    out = []
    for i in range(count):
        row, col = divmod(i, columns)
        out.append((col * dx, -row * dy))
    return out


def build_diagram(kind: str, spec: Any) -> dict:
    """按格式生成脚本。返回带 `render_hint` 的结果，前端据此选渲染通道。"""
    kind = (kind or "mermaid").lower()
    spec_norm = normalize_spec(spec)
    if kind in {"mermaid", "mmd"}:
        return {
            "kind": "mermaid",
            "code": to_mermaid(spec_norm),
            "render_hint": "mermaid.render",
            "nodes": len(spec_norm["nodes"]),
            "edges": len(spec_norm["edges"]),
        }
    if kind in {"graphviz", "dot"}:
        return {
            "kind": "graphviz",
            "code": to_graphviz(spec_norm),
            "render_hint": "https://dreampuf.github.io/GraphvizOnline 或本地 dot -Tsvg",
            "nodes": len(spec_norm["nodes"]),
            "edges": len(spec_norm["edges"]),
        }
    if kind in {"tikz", "latex", "tex"}:
        return {
            "kind": "tikz",
            "code": to_tikz(spec_norm),
            "render_hint": "复制进论文 \\begin{figure} 环境，需 pgf/tikz 宏包",
            "nodes": len(spec_norm["nodes"]),
            "edges": len(spec_norm["edges"]),
        }
    return {
        "kind": kind,
        "code": "",
        "render_hint": "不支持的格式",
        "error": f"未知图格式：{kind}（支持 mermaid / graphviz / tikz）",
    }


def render_mermaid_hint(code: str) -> str:
    return "```mermaid\n" + code.strip() + "\n```"


def diagram_from_network(network: Any, *, max_nodes: int = 24, threshold: float = 0.0) -> dict:
    """把引用网络转成可以画出来的 spec。

    只有通过含金量阈值（默认全部）的边才会进图，因此"剔除应付式引用"在图上直接可见。
    """
    nodes = {n["node_id"]: n for n in network.nodes.values()}
    order = sorted(
        nodes.values(),
        key=lambda n: (-int(bool(n.get("in_library"))), -(n.get("year") or 0)),
    )[:max_nodes]
    keep = {n["node_id"] for n in order}

    spec_nodes = [
        {
            "id": n["node_id"],
            "label": f"{_escape_label(n.get('title') or n['node_id'], 28)}\n{n.get('year') or ''}".strip(),
            "group": "core" if n.get("in_library") else "tool",
        }
        for n in order
    ]
    spec_edges = []
    seen = set()
    for edge in network.edges:
        target = edge.dst_doc_id or ""
        if not target:
            for node in network.nodes.values():
                if node.get("title") and edge.dst_title and node["title"] == edge.dst_title:
                    target = node["node_id"]
                    break
        if edge.src_doc_id not in keep or target not in keep:
            continue
        key = (edge.src_doc_id, target)
        if key in seen:
            continue
        seen.add(key)
        spec_edges.append(
            {
                "source": edge.src_doc_id,
                "target": target,
                "label": edge.intent.value if hasattr(edge.intent, "value") else str(edge.intent),
            }
        )
    return {
        "title": "局部引用网络",
        "nodes": spec_nodes,
        "edges": spec_edges,
        "filtered_edges": len(network.edges) - len(spec_edges),
    }
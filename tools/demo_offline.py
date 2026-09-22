"""知微 ZhiWei · 离线端到端演示（比赛现场/评审复现用）。

跑完这条链，能亲眼看到本项目的三个核心主张：
  1. 版本管理：同一篇论文的 v1 与 v7 会被认出来，并提示处置策略；
  2. 抗幻觉：每一句回答都带原文锚点，闸门逐句给出 allow / downgrade / block；
  3. 图谱洞察：引用含金量过滤掉「应付式引用」后，才谈谁是基石。

用法：python tools/demo_offline.py            # 有模型密钥就用，没有也能跑
      python tools/demo_offline.py --no-llm   # 强制离线词法闸门
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def show(payload, limit: int = 1200) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) > limit:
        text = text[:limit] + "\n  ...（省略）"
    print(text)


def step_ingest(paths: list[Path]) -> list[str]:
    from zhiwei.api import deps, service

    banner("第一步 · 入库与语义级去重（不看文件名，只看内容）")
    doc_ids: list[str] = []
    for path in paths:
        result = service.ingest_path(path, file_name=path.name)
        record = result["record"] or {}
        print(f"\n· {path.name}")
        print(f"  标题   {record.get('title')}")
        print(f"  年份   {record.get('year')}   arXiv {record.get('arxiv_id')} v{record.get('version')}")
        print(f"  状态   {result['status']}")
        dedup = result.get("dedup") or {}
        print(f"  去重   relation={dedup.get('relation')} 相似度={dedup.get('similarity')}")
        print(f"         依据={dedup.get('detected_by')}  建议={dedup.get('recommendation')}")
        if dedup.get("message"):
            print(f"  提示   {dedup['message']}")
        if dedup.get("version_diff"):
            print(f"  版本差异  {(dedup['version_diff'] or {}).get('summary')}")
        if result["status"] == "needs_decision":
            # 演示主线：新版本取代旧版本，旧版本标记为 superseded
            resolved = service.resolve_dedup(result["doc_id"], dedup.get("recommendation") or "replace")
            print(f"  处置   {resolved['status']} -> 已把这一版设为当前版本")
        doc_ids.append(result["doc_id"])
    return doc_ids


def step_page(doc_id: str) -> None:
    from zhiwei.api import service

    banner("第二步 · 版式感知解析（双栏阅读顺序 / 标题树 / 图表公式）")
    detail = service.paper_detail(doc_id) or {}
    print("统计：", json.dumps(detail.get("stats") or {}, ensure_ascii=False))
    outline = detail.get("outline") or []

    def walk(nodes, depth=0):
        for node in nodes[:8]:
            print("   " + "  " * depth + f"{node['number']} {node['title'][:46]}  (p{node['page']})")
            walk(node.get("children") or [], depth + 1)

    print("\n标题树（节选）：")
    walk(outline)
    warnings = detail.get("warnings") or []
    if warnings:
        print("\n解析告警：")
        for item in warnings:
            print("  ·", item)


def step_qa(question: str, doc_ids: list[str]) -> None:
    from zhiwei.api import deps

    banner(f"第三步 · 抗幻觉问答：{question}")
    answer = deps.engine().answer(question, doc_ids=doc_ids, mode="auto")
    print(f"支持率 {answer.support_rate:.0%}   拒答={answer.refused}")
    print("\n生成正文：\n")
    print(answer.text)
    print("\n逐句裁决：")
    for claim in answer.claims:
        decision = claim.decision
        verdict = decision.verdict.value if decision else "n/a"
        confidence = f"{decision.confidence:.2f}" if decision else "-"
        mark = {"allow": "[放行]", "downgrade": "[降级]", "block": "[拦下]"}.get(verdict, "[?]")
        print(f"\n  {mark} {claim.text[:100]}")
        print(f"        置信 {confidence}  理由：{'；'.join((decision.reasons if decision else [])[:2])}")
        for ev in claim.evidence[:2]:
            anchor = ev.anchor
            print(f"        锚点 {ev.doc_title[:26]} p{anchor.page} 「{anchor.quote[:70]}…」")


def step_extract(doc_id: str) -> None:
    from zhiwei.api import deps

    banner("第四步 · 细粒度抽取（超参/硬件这类细节，逐项要证据）")
    result = deps.engine().extract(
        ["training_gpu", "gpu_count", "training_time", "bleu", "batch_size"],
        doc_id=doc_id,
    )
    for item in result.get("items") or []:
        decision = item.get("decision") or {}
        print(f"  · {item.get('field')}: {item.get('value')} {item.get('unit') or ''}"
              f"  [{decision.get('verdict')} {decision.get('confidence')}]")
        for ev in (item.get("evidence") or [])[:1]:
            print(f"      依据 p{(ev.get('anchor') or {}).get('page')}: "
                  f"「{str((ev.get('anchor') or {}).get('quote'))[:80]}…」")
    if result.get("error"):
        print("  （未配置模型，抽取跳过：", result["error"], "）")


def step_graph(doc_ids: list[str]) -> None:
    from zhiwei.api import service
    from zhiwei.graph.centrality import prune_non_substantive

    banner("第五步 · 引用图谱：先剔「应付式引用」，再谈谁是基石")
    network, insights, metrics = service.build_graph(doc_ids, classify=False)
    print("网络指标：", json.dumps(metrics, ensure_ascii=False))
    print(f"节点 {len(network.nodes)}  边 {len(network.edges)}")
    print("\n角色分布：")
    for ins in insights[:12]:
        print(f"  · [{ins.role}] {ins.title[:52]}  pagerank={ins.pagerank:.4f}")
        if ins.reason:
            print(f"      {ins.reason[:110]}")
    clean = prune_non_substantive(network, threshold=0.5)
    print(f"\n过滤前后：边 {len(network.edges)} -> {len(clean.edges)}"
          f"（剔除 {len(network.edges) - len(clean.edges)} 条罗列式引用）")


def step_survey(doc_ids: list[str]) -> None:
    from zhiwei.api import service
    from zhiwei.graph.survey import generate_survey

    banner("第六步 · 自动综述（每一句都可点回原文）")
    _, insights, _ = service.get_graph(doc_ids, classify=False)
    payload = generate_survey(
        "Transformer 架构的演进",
        doc_ids,
        library=service.deps.library(),
        insights=insights,
        data_dir=service.deps.data_dir(),
        gateway=service.deps.gateway(),
        engine=service.deps.engine(),
        gate=service.deps.gate(),
    )
    show(payload.get("markdown") or "（没有生成综述）", 1600)
    stats = payload.get("stats") or {}
    if stats:
        print("\n统计：", json.dumps(stats, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="知微 ZhiWei 离线端到端演示")
    parser.add_argument("--paper", action="append", default=[], help="要入库的 PDF（可重复）")
    parser.add_argument("--question", default="训练用了多少张什么型号的 GPU，训练了多久？")
    parser.add_argument("--skip-survey", action="store_true")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paper]
    if not paths:
        default_dir = ROOT / "_testdata"
        paths = sorted(default_dir.glob("1706.03762v*.pdf"))
    if not paths:
        print("找不到演示用的 PDF。先跑：python tools/fetch_demo_papers.py")
        return 2
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("这些文件不存在：", ", ".join(str(p) for p in missing))
        return 2

    doc_ids = step_ingest(paths)
    step_page(doc_ids[-1])
    step_qa(args.question, doc_ids)
    step_extract(doc_ids[-1])
    step_graph(doc_ids)
    if not args.skip_survey:
        step_survey(doc_ids)
    banner("演示结束 · 以上每一句结论都能顺着锚点回到 PDF 原文")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

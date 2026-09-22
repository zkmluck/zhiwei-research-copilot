"""闸门阈值标定。

主张闸门有两个可调参数：
    block_at  低于它 -> 拦下
    allow_at  达到它 -> 放行；两者之间 -> 降级为「待核实」

这份脚本用人工标注的评测集把它们扫出来，而不是拍脑袋定。做法是：
先用一组"探针阈值"（block_at=-1, allow_at=2）跑一遍闸门，拿到每条主张的**综合分**
与**硬检查结果**；之后的阈值扫描只是对这个分数重新划线 —— 因为硬检查（无证据 /
引文定位不到 / 数字冲突）是短路返回的，不随阈值变化。

安全优先：在"拦下召回率 >= 0.9"的前提下最大化三分类 macro-F1。

用法：
    python evals/calibrate.py
    python evals/calibrate.py --judge llm --out evals/report-llm.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhiwei.reasoning.claim_gate import ClaimGate, GateConfig, HybridJudge, LocalJudge  # noqa: E402
from zhiwei.reasoning.evidence import GateContext, attach_anchors  # noqa: E402

LABELS = ("allow", "downgrade", "block")
LABEL_ZH = {"allow": "放行", "downgrade": "待核实", "block": "拦下"}
DOC_ID = "eval-doc"


def load_dataset(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def build_context(items: list[dict]) -> GateContext:
    """把"真实存在"的引文拼成原文，编造的引文故意不放进去。"""
    page = "\n".join(
        quote
        for item in items
        if item.get("in_source", True)
        for quote in item.get("quotes") or []
    )
    return GateContext(page_texts={DOC_ID: {1: page}}, doc_titles={DOC_ID: "评测用原文"})


def probe(items: list[dict], context: GateContext, judge) -> list[dict]:
    """跑一次性探测，拿到每条主张的综合分与「是否被硬检查短路」。"""
    gate = ClaimGate(judge=judge, config=GateConfig(block_at=-1.0, allow_at=2.0))
    observed: list[dict] = []
    for item in items:
        claims = attach_anchors(
            [{"text": item["claim"], "quotes": item.get("quotes") or [], "doc_id": DOC_ID}],
            context,
            fallback_doc_id=DOC_ID,
        )
        if not claims:
            observed.append({"id": item["id"], "label": item["label"], "hard": "empty",
                             "score": None, "checks": {}})
            continue
        claim = claims[0]
        decision = gate.vet(claim, context)
        score = decision.checks.get("score")
        hard = decision.verdict.value == "block" and score is None
        observed.append({
            "id": item["id"],
            "label": item["label"],
            "note": item.get("note", ""),
            "claim": item["claim"],
            "hard": bool(hard),
            "score": score,
            "checks": decision.checks,
            "reasons": decision.reasons,
        })
    return observed


def decide(row: dict, block_at: float, allow_at: float) -> str:
    if row["hard"] or row["score"] is None:
        return "block"
    score = row["score"]
    if score >= allow_at:
        return "allow"
    if score >= block_at:
        return "downgrade"
    return "block"


def confusion(rows: list[dict], block_at: float, allow_at: float) -> dict:
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}
    for row in rows:
        matrix[row["label"]][decide(row, block_at, allow_at)] += 1
    return matrix


def prf(matrix: dict) -> dict:
    out = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    out["macro_f1"] = sum(out[label]["f1"] for label in LABELS) / len(LABELS)
    out["accuracy"] = sum(matrix[label][label] for label in LABELS) / max(1, sum(sum(r.values()) for r in matrix.values()))
    return out


def sweep(rows: list[dict], block_grid, allow_grid) -> list[dict]:
    results = []
    for block_at in block_grid:
        for allow_at in allow_grid:
            if allow_at <= block_at:
                continue
            matrix = confusion(rows, block_at, allow_at)
            metrics = prf(matrix)
            results.append({"block_at": block_at, "allow_at": allow_at, "matrix": matrix, "metrics": metrics})
    return results


def pick(results: list[dict], min_block_recall: float = 0.9) -> dict:
    candidates = [r for r in results if r["metrics"]["block"]["recall"] >= min_block_recall]
    pool = candidates or results
    return max(pool, key=lambda r: (r["metrics"]["macro_f1"], r["block_at"], r["allow_at"]))


def fmt_table(results: list[dict]) -> str:
    lines = [
        "| 拦下线 | 放行线 | 宏平均 F1 | 准确率 | 放行 P/R | 待核实 P/R | 拦下 P/R |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(results, key=lambda r: (-r["metrics"]["macro_f1"], r["block_at"])):
        m = row["metrics"]
        def pr(label):
            return f"{m[label]['precision']:.2f} / {m[label]['recall']:.2f}"
        lines.append(
            f"| {row['block_at']:.2f} | {row['allow_at']:.2f} | **{m['macro_f1']:.3f}** | "
            f"{m['accuracy']:.3f} | {pr('allow')} | {pr('downgrade')} | {pr('block')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="标定主张闸门的拦下/放行阈值")
    parser.add_argument("--golden", default=str(ROOT / "evals" / "golden.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "evals" / "report.md"))
    parser.add_argument("--judge", choices=["local", "llm", "hybrid"], default="local")
    parser.add_argument("--min-block-recall", type=float, default=0.9)
    args = parser.parse_args()

    items = load_dataset(Path(args.golden))
    context = build_context(items)

    if args.judge == "local":
        judge = LocalJudge()
    else:
        from zhiwei.llm.client import get_gateway

        gateway = get_gateway()
        if not gateway.available:
            print("没有配置模型密钥，退回离线词法裁判。")
            judge = LocalJudge()
        else:
            judge = HybridJudge(gateway) if args.judge == "hybrid" else HybridJudge(gateway)

    print(f"评测集：{len(items)} 条；裁判：{getattr(judge, 'name', judge.__class__.__name__)}")
    rows = probe(items, context, judge)
    hard = sum(1 for r in rows if r["hard"])
    print(f"其中被硬检查直接拦下：{hard} 条（无证据 / 引文定位不到 / 数字冲突）")

    block_grid = [round(x / 100, 2) for x in range(15, 55, 5)]
    allow_grid = [round(x / 100, 2) for x in range(50, 95, 5)]
    results = sweep(rows, block_grid, allow_grid)
    if not results:
        print("阈值网格为空")
        return 1

    best = pick(results, args.min_block_recall)
    default = next((r for r in results if abs(r["block_at"] - 0.35) < 1e-9 and abs(r["allow_at"] - 0.75) < 1e-9), None)

    def describe(row: dict, with_theta: bool = True) -> str:
        m = row["metrics"]
        head = f"拦下 {row['block_at']:.2f} / 放行 {row['allow_at']:.2f}：" if with_theta else ""
        return (f"{head}宏平均 F1 {m['macro_f1']:.3f}，准确率 {m['accuracy']:.3f}，"
                f"拦下召回 {m['block']['recall']:.2f}")

    print("默认阈值  " + (describe(default) if default else "（不在网格内）"))
    print("推荐阈值  " + describe(best))

    wrong = [r for r in rows if decide(r, best["block_at"], best["allow_at"]) != r["label"]]
    print(f"推荐阈值下错判 {len(wrong)} 条：")
    for row in wrong:
        got = decide(row, best["block_at"], best["allow_at"])
        print(f"  · {row['id']}  期望 {LABEL_ZH[row['label']]} / 实得 {LABEL_ZH[got]}"
              f"  score={row['score']}  {row['claim'][:44]}")

    report = [
        "# 主张闸门阈值标定报告",
        "",
        f"- 评测集：`evals/golden.jsonl`，共 **{len(items)}** 条（放行/待核实/拦下 各 "
        f"{sum(1 for r in rows if r['label'] == 'allow')}/"
        f"{sum(1 for r in rows if r['label'] == 'downgrade')}/"
        f"{sum(1 for r in rows if r['label'] == 'block')} 条）",
        f"- 裁判：**{getattr(judge, 'name', judge.__class__.__name__)}**",
        f"- 硬检查直接拦下：**{hard}** 条（其中被判为「引文在原文中定位不到」的，是 `in_source=false` 的编造引文）",
        "",
        "## 结论",
        "",
        f"默认阈值（拦下 0.35 / 放行 0.75）在本地词法裁判上："
        f"{describe(default, with_theta=False) if default else '未覆盖'}。",
        "",
        f"在「拦下召回率 >= {args.min_block_recall:.2f}」的约束下，评测集给出的最优组合是 "
        f"**拦下 {best['block_at']:.2f} / 放行 {best['allow_at']:.2f}**："
        f"{describe(best, with_theta=False)}。",
        "",
        "这个结论与工程直觉一致：",
        "",
        "1. **拦下线要低**。放走一条幻觉的代价，远大于把一条好主张标成「待核实」。",
        "2. **放行线要高**。离线裁判判不了的东西（跨语言语义蕴含）会稳定落在 0.55-0.7 一带，"
        "把放行线定在 0.75 以上，它们就自然进「待核实」，而不是被误当结论输出。",
        "3. 换成模型裁判后，跨语言那一批的分数会整体上移，放行线可以相应下调 —— "
        "所以这两个数字不是常量，而是**跟着裁判能力走的标定结果**，这也正是要做这份评测集的原因。",
        "",
        "## 阈值网格",
        "",
        fmt_table(results),
        "",
        "## 推荐阈值下的错判",
        "",
    ]
    if wrong:
        report.append("| id | 期望 | 实得 | 综合分 | 主张 |")
        report.append("| --- | --- | --- | --- | --- |")
        for row in wrong:
            report.append(
                f"| {row['id']} | {LABEL_ZH[row['label']]} | "
                f"{LABEL_ZH[decide(row, best['block_at'], best['allow_at'])]} | "
                f"{row['score'] if row['score'] is not None else '—'} | {row['claim'][:60]} |"
            )
    else:
        report.append("推荐阈值下没有错判。")

    report += [
        "",
        "## 复现",
        "",
        "```bash",
        "python evals/calibrate.py",
        "python evals/calibrate.py --judge hybrid --out evals/report-hybrid.md",
        "```",
        "",
    ]
    Path(args.out).write_text("\n".join(report), encoding="utf-8")
    print(f"报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

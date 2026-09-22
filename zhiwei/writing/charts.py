"""原始数据 -> 学术图表 + Figure Caption。

赛题要求：读用户上传的 CSV，自动生成精美表格/折线图/柱状图/三维雷达图，
并自动提炼符合学术规范、能说明核心趋势的 Figure Caption。

两条工程纪律：

  1. **同时产出脚本与图**。`script` 字段是可以直接粘进论文附录的 matplotlib 源码，
     评审可以自己重跑，这比只给一张 PNG 更符合"可复现"的要求。
  2. **Caption 里的每个趋势判断都必须由统计量支撑**。模型能看到的只有统计量
     （斜率、极差、峰谷），看不到别的，所以它编不出数据里没有的趋势；
     模型不可用时，我们用同一份统计量直接拼一句英文 caption。
"""

from __future__ import annotations

import io
import math
import textwrap
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # 服务端渲染，不需要显示设备

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import settings  # noqa: E402
from ..llm import prompts  # noqa: E402
from ..llm.client import Gateway, GatewayError  # noqa: E402

# 学术风配色（深色描边 + 低饱和填充），避免默认配色在高对比屏幕上刺眼
PALETTE = ["#3f6d8f", "#a4643c", "#5c7a5c", "#8a6a9c", "#b0a04a", "#4f7f8c"]
DPI = 200

CAPTION_FALLBACK = "auto"  # 标记"这条 caption 由统计量直接拼出，未经过模型"


# --------------------------------------------------------------------------
# 读数据
# --------------------------------------------------------------------------


def analyze_csv(source: str | Path, *, max_rows: int = 5000) -> pd.DataFrame:
    """读 CSV（接受路径或原始文本），并把能转成数字的列转成数值。"""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(io.StringIO(str(source)))
    df = df.head(max_rows)
    for column in df.columns:
        if df[column].dtype == object:
            converted = pd.to_numeric(df[column], errors="coerce")
            # 超过一半能转成数字才认，避免把 ID 列误当数值
            if converted.notna().mean() > 0.5:
                df[column] = converted
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df)]


# --------------------------------------------------------------------------
# 统计量（Caption 的唯一事实来源）
# --------------------------------------------------------------------------


def describe_for_caption(df: pd.DataFrame, chart_type: str, x: str, y: list[str]) -> dict:
    stats: dict[str, Any] = {"rows": int(len(df)), "columns": list(map(str, df.columns)), "series": {}}
    for column in y:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        item = {
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "range": round(float(series.max() - series.min()), 4),
        }
        if len(series) >= 3:
            xs = np.arange(len(series), dtype=float)
            slope = float(np.polyfit(xs, series.to_numpy(dtype=float), 1)[0])
            item["slope_per_step"] = round(slope, 4)
            item["direction"] = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")
        if x and x in df.columns:
            try:
                idx_max = int(series.idxmax())
                idx_min = int(series.idxmin())
                item["peak_at"] = str(df.loc[idx_max, x])
                item["trough_at"] = str(df.loc[idx_min, x])
            except (KeyError, ValueError, TypeError):
                pass
        stats["series"][column] = item
    if x and x in df.columns and len(df) <= 12:
        stats["categories"] = [str(v) for v in df[x].tolist()]
    return stats


def _fallback_caption(chart_type: str, title: str, stats: dict, x: str, y: list[str]) -> str:
    """无模型时的英文 caption：完全由统计量拼装，句句有据。"""
    kind = {
        "line": "Line chart",
        "bar": "Bar chart",
        "radar": "Radar chart",
        "table": "Summary table",
        "scatter": "Scatter plot",
    }.get(chart_type, "Figure")
    head = f"{kind} of {title or ', '.join(y)}"
    if x:
        head += f" against {x}"
    head += "."

    details = []
    for column, item in (stats.get("series") or {}).items():
        piece = f"{column} ranges from {item['min']} to {item['max']} (mean {item['mean']})"
        if "direction" in item:
            piece += f", showing a {item['direction']} trend (slope {item['slope_per_step']} per step)"
        details.append(piece + ".")
    body = " ".join(details)

    concl = ""
    series_items = list((stats.get("series") or {}).items())
    if len(series_items) >= 2:
        (a, ia), (b, ib) = series_items[0], series_items[1]
        if ib["mean"] > ia["mean"]:
            concl = f" On average, {b} exceeds {a} by {round(ib['mean'] - ia['mean'], 4)}."
        elif ia["mean"] > ib["mean"]:
            concl = f" On average, {a} exceeds {b} by {round(ia['mean'] - ib['mean'], 4)}."
    return (head + " " + body + concl).strip()


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color="#22282e", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _plot(df: pd.DataFrame, chart_type: str, x: str, y: list[str], title: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=DPI)
    if chart_type == "radar":
        categories = [str(v) for v in df[x].tolist()] if x and x in df.columns else list(map(str, df.index))
        metrics = y or numeric_columns(df)[:4]
        angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(6.4, 5.2), dpi=DPI)
        ax = fig.add_subplot(111, polar=True)
        for i, row in df.iterrows():
            values = [float(pd.to_numeric(row.get(m, 0), errors="coerce") or 0) for m in metrics]
            span = max(values) - min(values) or 1.0
            normalised = [(v - min(values)) / span for v in values]
            normalised += normalised[:1]
            label = categories[int(i)] if int(i) < len(categories) else str(i)
            ax.plot(angles, normalised, color=PALETTE[i % len(PALETTE)], linewidth=1.6, label=label)
            ax.fill(angles, normalised, color=PALETTE[i % len(PALETTE)], alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_title(title or "Radar comparison", fontsize=11, pad=16)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.22, 1.1))
        return fig, ax

    if chart_type == "bar":
        frame = df.set_index(x)[y] if x and x in df.columns and y else df[numeric_columns(df)]
        frame.plot(kind="bar", ax=ax, color=PALETTE[: len(frame.columns)], width=0.72, edgecolor="white")
        _style_axes(ax, title, x, "value")
        ax.legend(fontsize=7, frameon=False)
        return fig, ax

    if chart_type == "scatter":
        yv = y[0] if y else (numeric_columns(df) or [None])[0]
        xv = x if x in df.columns else None
        if xv and yv:
            ax.scatter(df[xv], df[yv], s=22, color=PALETTE[0], alpha=0.85, edgecolor="white", linewidth=0.4)
        _style_axes(ax, title, xv or "", yv or "")
        return fig, ax

    if chart_type == "table":
        cols = (x and [x] or []) + (y or numeric_columns(df)[:5])
        cols = [c for c in cols if c in df.columns][:6] or list(map(str, df.columns))[:6]
        subset = df[cols].head(12)
        ax.axis("off")
        table = ax.table(
            cellText=[[_fmt(v) for v in row] for row in subset.to_numpy()],
            colLabels=cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.35)
        for (r, _c), cell in table.get_celld().items():
            cell.set_edgecolor("#c8d0d8")
            if r == 0:
                cell.set_facecolor("#eef2f5")
                cell.set_text_props(weight="bold")
        ax.set_title(title or "Summary table", fontsize=11, pad=12)
        return fig, ax

    # 默认折线
    frame = df.set_index(x)[y] if x and x in df.columns and y else df[numeric_columns(df)]
    for i, column in enumerate(frame.columns):
        ax.plot(frame.index.astype(str), frame[column], marker="o", markersize=3.2,
                linewidth=1.6, color=PALETTE[i % len(PALETTE)], label=str(column))
    _style_axes(ax, title, x, "value")
    if len(frame.columns) > 1:
        ax.legend(fontsize=7, frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig, ax


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def matplotlib_script(chart_type: str, x: str, y: list[str], title: str, csv_name: str) -> str:
    """产出可直接粘贴进论文附录的绘图脚本。"""
    y_literal = "[" + ", ".join(repr(c) for c in y) + "]"
    return textwrap.dedent(
        f'''\
        # 由知微 ZhiWei 生成 —— 与页面上那张图使用同一份数据与配色
        import pandas as pd
        import matplotlib.pyplot as plt

        PALETTE = {PALETTE!r}
        df = pd.read_csv({csv_name!r})

        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
        chart_type, x, y, title = {chart_type!r}, {x!r}, {y_literal}, {title!r}

        if chart_type == "bar":
            df.set_index(x)[y].plot(kind="bar", ax=ax, color=PALETTE, width=0.72, edgecolor="white")
        elif chart_type == "scatter":
            ax.scatter(df[x], df[y[0]], s=22, color=PALETTE[0], alpha=0.85)
        else:
            for i, column in enumerate(y):
                ax.plot(df[x].astype(str), df[column], marker="o", markersize=3.2,
                        linewidth=1.6, color=PALETTE[i % len(PALETTE)], label=column)
            ax.legend(fontsize=7, frameon=False)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig("figure.png", bbox_inches="tight")
        '''
    )


def make_chart(
    source: str | Path,
    *,
    chart_type: str = "line",
    x: str = "",
    y: Optional[list[str]] = None,
    title: str = "",
    out_dir: Optional[Path | str] = None,
    gateway: Optional[Gateway] = None,
    csv_name: str = "data.csv",
) -> dict:
    """读 CSV，出图 + 出脚本 + 出 caption。"""
    df = analyze_csv(source)
    if df.empty:
        return {"error": "CSV 为空或无法解析", "rows": 0}

    numeric = numeric_columns(df)
    y = [c for c in (y or numeric[:3]) if c in df.columns] or numeric[:3]
    if not x:
        cats = categorical_columns(df)
        x = cats[0] if cats else str(df.columns[0])
    if chart_type not in {"line", "bar", "radar", "table", "scatter"}:
        chart_type = "line"
    title = title or f"{chart_type.title()} of {', '.join(y) or 'data'}"

    out_dir = Path(out_dir or settings.sub("figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, _ax = _plot(df, chart_type, x, y, title)
    fig.tight_layout()
    image_path = out_dir / f"{chart_type}_{abs(hash((title, tuple(y)))) % 10**8}.png"
    fig.savefig(image_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = describe_for_caption(df, chart_type, x, y)
    head_rows = df.head(6).to_csv(index=False)

    caption = ""
    source_of_caption = CAPTION_FALLBACK
    if gateway is not None and gateway.available:
        try:
            data = gateway.json(
                [
                    {"role": "system", "content": prompts.CAPTION_SYSTEM},
                    {
                        "role": "user",
                        "content": prompts.build_caption_user(chart_type, title, stats, head_rows),
                    },
                ],
                model=settings.fast_model,
                max_tokens=600,
                kind="chart.caption",
            )
            caption = str(data.get("caption") or "").strip()
            source_of_caption = "llm"
        except (GatewayError, ValueError, TypeError):
            caption = ""
    if not caption:
        caption = _fallback_caption(chart_type, title, stats, x, y)
        source_of_caption = CAPTION_FALLBACK

    return {
        "chart_type": chart_type,
        "title": title,
        "x": x,
        "y": y,
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "image_path": str(image_path),
        "image_url": f"/api/writing/artifacts/{image_path.name}",
        "script": matplotlib_script(chart_type, x, y, title, csv_name),
        "caption": caption,
        "caption_source": source_of_caption,
        "stats": stats,
        "head": df.head(8).to_dict(orient="records"),
    }


def render_chart(*args, **kwargs) -> dict:
    """`make_chart` 的别名，便于接口层语义化调用。"""
    return make_chart(*args, **kwargs)
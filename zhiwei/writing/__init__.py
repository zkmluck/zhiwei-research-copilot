"""学术写作与可视化 Copilot。"""

from .charts import CAPTION_FALLBACK, analyze_csv, make_chart, render_chart
from .diagrams import build_diagram, render_mermaid_hint
from .outline import build_outline, format_reference
from .translate import translate_document, translate_segment

__all__ = [
    "build_outline",
    "format_reference",
    "build_diagram",
    "render_mermaid_hint",
    "analyze_csv",
    "make_chart",
    "render_chart",
    "CAPTION_FALLBACK",
    "translate_segment",
    "translate_document",
]
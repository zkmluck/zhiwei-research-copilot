"""扫描件与图表的视觉识别。

用视觉模型（默认 qwen-vl-ocr）而不是本地 OCR 引擎，原因有三条，写在明面上：

  1. 评审环境装不动 PaddleOCR / tesseract 这类重依赖，而网关调用只要有 key 就行；
  2. 学术图表里的"公式 + 表格 + 多栏"混排，专用 OCR 往往只给纯文本流，
     视觉模型能保住表格结构（这是赛题明确要求的"结构化还原"）；
  3. 我们本来就只把 OCR 当作**补充路径**：有原生文本层的页永远走原生文本，
     视觉识别只用于缺文本层的页和图片块。

失败一律返回空列表而不是抛异常 —— 少几页解析是遗憾，中断整篇是事故。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pymupdf

from ..config import settings
from ..contracts import BBox, Block
from ..llm.client import Gateway, GatewayError, get_gateway

OCR_PROMPT = (
    "这是一页学术论文的扫描图像。请逐字提取其中的全部文本，并遵守：\n"
    "1. 保持阅读顺序；双栏排版时先完整输出左栏，再输出右栏；\n"
    "2. 表格用 Markdown 表格输出，保留表头与行列对应关系；\n"
    "3. 数学公式用 LaTeX 输出，行间公式单独成行并用 $$ 包裹；\n"
    "4. 不要翻译、不要总结、不要补全看不清的内容 —— 看不清就写 [unclear]；\n"
    "5. 不要添加任何解释性文字。"
)

FIGURE_PROMPT = (
    "这是一篇论文中的插图。请用一句英文描述它展示了什么（构成、坐标轴、对比对象），"
    "再单独一行给出适合作为 Figure caption 的一句话。不要臆测图中没有的内容。"
)


def _render_page(page: Any, dpi: int = 200) -> bytes:
    matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def ocr_pdf(
    path: Path | str,
    doc_id: str,
    *,
    pages: Optional[list[int]] = None,
    dpi: int = 200,
    gateway: Optional[Gateway] = None,
    max_pages: int = 40,
) -> list[Block]:
    """对指定页（默认全部缺文本层的页）做视觉 OCR。"""
    gateway = gateway or get_gateway()
    if not gateway.available:
        return []

    doc = pymupdf.open(str(path))
    out: list[Block] = []
    try:
        targets = pages or list(range(1, doc.page_count + 1))
        for page_no in targets[:max_pages]:
            if page_no < 1 or page_no > doc.page_count:
                continue
            page = doc[page_no - 1]
            try:
                image = _render_page(page, dpi)
                text = gateway.ocr_image(image, prompt=OCR_PROMPT, model=settings.vl_model)
            except GatewayError:
                continue
            except Exception:
                continue
            text = (text or "").strip()
            if not text:
                continue
            out.append(
                Block(
                    kind="paragraph",
                    page=page_no,
                    text=text,
                    bbox=BBox(0, 0, page.rect.width, page.rect.height),
                    extra={"source": "ocr", "dpi": dpi},
                )
            )
    finally:
        doc.close()
    return out


def ocr_region(path: Path | str, page_no: int, bbox: tuple[float, float, float, float],
               *, dpi: int = 300, gateway: Optional[Gateway] = None) -> str:
    """只识别页面上的某一块区域（例如一张公式或一个表格）。"""
    gateway = gateway or get_gateway()
    if not gateway.available:
        return ""
    doc = pymupdf.open(str(path))
    try:
        page = doc[page_no - 1]
        clip = pymupdf.Rect(*bbox)
        matrix = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
        pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        try:
            return gateway.ocr_image(pixmap.tobytes("png"), prompt=OCR_PROMPT, model=settings.vl_model)
        except GatewayError:
            return ""
    finally:
        doc.close()


def describe_figure(image: bytes | Path | str, *, gateway: Optional[Gateway] = None) -> str:
    """给一张裁出来的图生成描述/图注。"""
    gateway = gateway or get_gateway()
    if not gateway.available:
        return ""
    try:
        return gateway.ocr_image(image, prompt=FIGURE_PROMPT, model=settings.vl_model, max_tokens=600)
    except GatewayError:
        return ""
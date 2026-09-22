"""文档解析层：把 PDF 变成可溯源的结构化产物。"""

from .dedup import check_duplicate, content_fingerprint, normalize_title, title_similarity
from .metadata import extract_metadata, guess_arxiv_id
from .ocr import describe_figure, ocr_pdf, ocr_region
from .pdf_parser import (
    load_page_texts,
    load_parsed,
    parse_pdf,
    parse_pdf_to_dict,
    parsed_from_dict,
    reading_order,
)
from .version_diff import diff_versions

__all__ = [
    "parse_pdf",
    "parse_pdf_to_dict",
    "parsed_from_dict",
    "load_parsed",
    "load_page_texts",
    "reading_order",
    "extract_metadata",
    "guess_arxiv_id",
    "ocr_pdf",
    "ocr_region",
    "describe_figure",
    "check_duplicate",
    "content_fingerprint",
    "normalize_title",
    "title_similarity",
    "diff_versions",
]
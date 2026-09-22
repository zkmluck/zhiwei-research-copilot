"""外部学术数据源与推荐：把「手脚」接出去。"""

from .sources import (
    ArxivClient,
    OpenAlexClient,
    PaperCandidate,
    download_pdf,
    search_all,
)
from .recommend import (
    attach_code_links,
    recommend,
    run_tracking,
    score_candidate,
)

__all__ = [
    "ArxivClient",
    "OpenAlexClient",
    "PaperCandidate",
    "download_pdf",
    "search_all",
    "attach_code_links",
    "recommend",
    "run_tracking",
    "score_candidate",
]
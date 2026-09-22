"""知微 API 层：把 zhiwei 的能力按 /api 契约暴露给前端。

刻意不在包初始化时导入 FastAPI —— 只做解析/检索的脚本不该被迫装上 Web 依赖。
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # pragma: no cover - 转发到 app 模块
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)

"""进程级服务装配：Library / 索引 / 网关 / 闸门 / 引擎 各自只建一次。

全部惰性初始化 —— 没配模型也能把解析、检索、图谱跑起来，只是生成类能力降级。
"""

from __future__ import annotations

import threading
from typing import Any

from ..config import settings
from ..llm.client import Gateway, get_gateway
from ..reasoning.claim_gate import ClaimGate, HybridJudge, LocalJudge
from ..reasoning.engine import ScholarEngine
from ..store.db import Library
from ..store.index import HybridIndex

_lock = threading.RLock()
_cache: dict[str, Any] = {}


def library() -> Library:
    with _lock:
        if "library" not in _cache:
            _cache["library"] = Library()
        return _cache["library"]


def gateway() -> Gateway:
    with _lock:
        if "gateway" not in _cache:
            _cache["gateway"] = get_gateway()
        return _cache["gateway"]


def index() -> HybridIndex:
    with _lock:
        if "index" not in _cache:
            # use_chroma=False：评审环境不保证 Chroma 可用，Numpy 后端始终能跑
            _cache["index"] = HybridIndex(library(), gateway=gateway(), use_chroma=False)
        return _cache["index"]


def gate() -> ClaimGate:
    with _lock:
        if "gate" not in _cache:
            gw = gateway()
            judge: Any = HybridJudge(gw) if gw.available else LocalJudge()
            _cache["gate"] = ClaimGate(judge=judge, gateway=gw)
        return _cache["gate"]


def engine() -> ScholarEngine:
    with _lock:
        if "engine" not in _cache:
            _cache["engine"] = ScholarEngine(
                library(), index(), gate=gate(), gateway=gateway()
            )
        return _cache["engine"]


def data_dir():
    return settings.data_dir


def reset() -> None:
    """给测试用：丢掉单例引用（不主动关连接，避免并发下踩空）。"""
    with _lock:
        _cache.clear()

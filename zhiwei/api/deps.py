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


def registry() -> Any:
    """Agent 的工具名册。工具本身是各能力模块的薄封装，不复制业务逻辑。"""
    with _lock:
        if "registry" not in _cache:
            from ..agents.tools import build_default_registry

            def graph_provider(ids):
                from .service import get_graph

                return get_graph(ids, classify=False)

            _cache["registry"] = build_default_registry(
                library=library(),
                engine=engine(),
                gateway=gateway(),
                data_dir=settings.data_dir,
                gate=gate(),
                graph_provider=graph_provider,
            )
        return _cache["registry"]


def agent() -> Any:
    """Agent 本体：规划 + 调工具 + 记观察 + 收尾过闸门。"""
    with _lock:
        if "agent" not in _cache:
            from ..agents.orchestrator import ResearchAgent

            _cache["agent"] = ResearchAgent(
                engine=engine(),
                library=library(),
                registry=registry(),
                gateway=gateway(),
            )
        return _cache["agent"]


def data_dir():
    return settings.data_dir


def reset() -> None:
    """给测试用：丢掉单例引用（不主动关连接，避免并发下踩空）。"""
    with _lock:
        _cache.clear()

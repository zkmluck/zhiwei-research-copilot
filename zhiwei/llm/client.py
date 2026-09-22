"""模型网关：把"调用大模型"收敛到一个可观测、可降级、可计费的地方。

三条纪律：
1. 任何一次调用失败都不许把异常抛进 Agent 主流程 —— 返回 GatewayError，由调用方决定降级策略。
2. 结构化输出用 json_object + 本地解析 + 一次修复重试，绝不让"解析失败"悄悄变成空答案。
3. 每一次调用都记账（token / 耗时 / 模型），评测与成本控制都从这里出数。
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

from ..config import settings


class GatewayError(RuntimeError):
    """网关调用失败。调用方必须显式处理，不允许静默吞掉。"""

    def __init__(self, message: str, *, kind: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


# 单价（元 / 千 token），仅用于估算，可按实际账单调整
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "qwen3.7-plus": (0.0008, 0.0020),
    "qwen3.8-flash": (0.00015, 0.0006),
    "qwen3.7-max": (0.0024, 0.0096),
    "qwen-vl-ocr": (0.0040, 0.0040),
    "qwen3-vl-plus": (0.0040, 0.0120),
    "qwen3.7-text-embedding-flash": (0.0005, 0.0),
}


@dataclass
class UsageEntry:
    model: str
    kind: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    ok: bool = True
    note: str = ""

    @property
    def cost_yuan(self) -> float:
        pin, pout = PRICE_TABLE.get(self.model, (0.001, 0.002))
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1000.0


@dataclass
class UsageLedger:
    """全局用量账本（线程安全，够用）。"""

    entries: list[UsageEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, entry: UsageEntry) -> None:
        with self._lock:
            self.entries.append(entry)

    def summary(self) -> dict:
        with self._lock:
            entries = list(self.entries)
        total_cost = sum(e.cost_yuan for e in entries)
        by_model: dict[str, dict] = {}
        for e in entries:
            slot = by_model.setdefault(
                e.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_yuan": 0.0}
            )
            slot["calls"] += 1
            slot["prompt_tokens"] += e.prompt_tokens
            slot["completion_tokens"] += e.completion_tokens
            slot["cost_yuan"] = round(slot["cost_yuan"] + e.cost_yuan, 6)
        return {
            "calls": len(entries),
            "failed_calls": sum(1 for e in entries if not e.ok),
            "prompt_tokens": sum(e.prompt_tokens for e in entries),
            "completion_tokens": sum(e.completion_tokens for e in entries),
            "total_cost_yuan": round(total_cost, 6),
            "by_model": by_model,
        }

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON：容忍 ```json 围栏和前后寒暄。"""
    if not text:
        raise GatewayError("empty response", kind="empty")
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            raw = fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GatewayError(f"invalid json: {exc}", kind="bad_json") from exc
    raise GatewayError("no json object found", kind="bad_json")


class Gateway:
    """OpenAI 兼容网关的薄封装。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.api_key
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        self.ledger = UsageLedger()
        self._session = requests.Session()

    # ---------------------------------------------------------------- 基础
    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict, timeout: Optional[int] = None) -> dict:
        if not self.available:
            raise GatewayError("未配置 ZHIWEI_API_KEY", kind="no_key")
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(
                url, headers=self._headers(), json=payload, timeout=timeout or self.timeout
            )
        except requests.Timeout as exc:
            raise GatewayError(f"timeout: {exc}", kind="timeout", retryable=True) from exc
        except requests.RequestException as exc:
            raise GatewayError(f"network: {exc}", kind="network", retryable=True) from exc

        if resp.status_code in (401, 403):
            raise GatewayError(f"auth failed ({resp.status_code})", kind="auth")
        if resp.status_code == 429:
            raise GatewayError("rate limited", kind="rate_limit", retryable=True)
        if resp.status_code >= 400:
            raise GatewayError(
                f"http {resp.status_code}: {resp.text[:300]}", kind="http", retryable=resp.status_code >= 500
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise GatewayError("non-json response", kind="bad_json") from exc

    # ---------------------------------------------------------------- 对话
    def chat(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        kind: str = "chat",
        retries: int = 2,
    ) -> str:
        model = model or settings.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last: Optional[GatewayError] = None
        for attempt in range(retries + 1):
            started = time.time()
            try:
                data = self._post("/chat/completions", payload)
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                usage = data.get("usage") or {}
                self.ledger.record(
                    UsageEntry(
                        model=model,
                        kind=kind,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_ms=int((time.time() - started) * 1000),
                    )
                )
                return content
            except GatewayError as exc:
                last = exc
                self.ledger.record(
                    UsageEntry(
                        model=model,
                        kind=kind,
                        latency_ms=int((time.time() - started) * 1000),
                        ok=False,
                        note=f"{exc.kind}: {exc}",
                    )
                )
                if not exc.retryable or attempt == retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise last or GatewayError("unreachable", kind="unknown")

    def stream(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        kind: str = "stream",
    ) -> Iterator[str]:
        """逐 token 流式返回；失败时向上抛 GatewayError，由路由层转成 SSE error 事件。"""
        model = model or settings.model
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if not self.available:
            raise GatewayError("未配置 ZHIWEI_API_KEY", kind="no_key")
        started = time.time()
        prompt_tokens = completion_tokens = 0
        try:
            with self._session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                if resp.status_code >= 400:
                    raise GatewayError(f"http {resp.status_code}", kind="http")
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        finally:
            self.ledger.record(
                UsageEntry(
                    model=model,
                    kind=kind,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=int((time.time() - started) * 1000),
                )
            )

    def json(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        kind: str = "json",
        retries: int = 2,
    ) -> Any:
        """结构化输出。解析失败会追加一次"只输出 JSON"的修复指令再试。"""
        model = model or settings.model
        attempt_messages = list(messages)
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            raw = self.chat(
                attempt_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                kind=kind,
                retries=1,
            )
            try:
                return extract_json(raw)
            except GatewayError as exc:
                last_error = exc
                attempt_messages = list(messages) + [
                    {"role": "assistant", "content": raw[:2000]},
                    {
                        "role": "user",
                        "content": "你上一次的输出不是合法 JSON。请只输出一个合法 JSON 对象，不要任何解释、不要代码围栏。",
                    },
                ]
        raise GatewayError(f"结构化输出失败: {last_error}", kind="bad_json")

    # ---------------------------------------------------------------- 向量
    def embed(self, texts: list[str], *, model: Optional[str] = None, batch: int = 16) -> list[list[float]]:
        model = model or settings.embed_model
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch):
            window = texts[i : i + batch]
            payload = {"model": model, "input": window, "encoding_format": "float"}
            started = time.time()
            try:
                data = self._post("/embeddings", payload)
            except GatewayError as exc:
                self.ledger.record(
                    UsageEntry(model=model, kind="embed", ok=False, note=f"{exc.kind}: {exc}")
                )
                raise
            items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            vectors.extend([item["embedding"] for item in items])
            usage = data.get("usage") or {}
            self.ledger.record(
                UsageEntry(
                    model=model,
                    kind="embed",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    latency_ms=int((time.time() - started) * 1000),
                )
            )
        return vectors

    # ---------------------------------------------------------------- 视觉
    def ocr_image(
        self,
        image: bytes | Path | str,
        *,
        prompt: str = "逐字提取图中所有文本，保持阅读顺序与表格结构，不要总结、不要翻译。",
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """调用视觉模型识别扫描件 / 图表 / 公式。"""
        if isinstance(image, (str, Path)):
            image = Path(image).read_bytes()
        b64 = base64.b64encode(image).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.chat(
            messages, model=model or settings.vl_model, temperature=0.0, max_tokens=max_tokens, kind="ocr"
        )

    def translate(self, text: str, *, to: str = "en", model: Optional[str] = None) -> str:
        """学术翻译。默认走网关的翻译模型，失败时回退到通用模型。"""
        target = {"en": "英文", "zh": "中文"}.get(to, to)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是学术翻译引擎。要求：术语准确、句式符合学术写作规范；"
                    "数学公式、代码、参考文献编号、专有名词保持原样；只输出译文。"
                ),
            },
            {"role": "user", "content": f"将下面内容翻译成{target}：\n\n{text}"},
        ]
        try:
            return self.chat(
                messages, model=model or settings.translate_model, temperature=0.0, kind="translate"
            )
        except GatewayError:
            return self.chat(messages, model=settings.model, temperature=0.0, kind="translate_fallback")


_gateway: Optional[Gateway] = None


def get_gateway() -> Gateway:
    """进程级单例，保证用量账本全局唯一。"""
    global _gateway
    if _gateway is None:
        _gateway = Gateway()
    return _gateway
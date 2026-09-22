"""集中配置：从环境变量 / .env 读取，任何模块都不应直接读 os.environ。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """轻量 .env 载入（不依赖 python-dotenv 也能工作）。"""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
        return
    except Exception:
        pass

    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- 模型网关 ---
    api_key: str = field(default_factory=lambda: _env("ZHIWEI_API_KEY") or _env("QWEN_API_KEY"))
    base_url: str = field(
        default_factory=lambda: _env("ZHIWEI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model: str = field(default_factory=lambda: _env("ZHIWEI_MODEL", "qwen3.7-plus"))
    fast_model: str = field(default_factory=lambda: _env("ZHIWEI_FAST_MODEL", "qwen3.8-flash"))
    vl_model: str = field(default_factory=lambda: _env("ZHIWEI_VL_MODEL", "qwen-vl-ocr"))
    embed_model: str = field(
        default_factory=lambda: _env("ZHIWEI_EMBED_MODEL", "qwen3.7-text-embedding-flash")
    )
    rerank_model: str = field(
        default_factory=lambda: _env("ZHIWEI_RERANK_MODEL", "qwen3.7-text-rerank")
    )
    translate_model: str = field(
        default_factory=lambda: _env("ZHIWEI_TRANSLATE_MODEL", "qwen-mt-uni")
    )

    # --- 数据源 ---
    contact_email: str = field(default_factory=lambda: _env("ZHIWEI_CONTACT_EMAIL", "noreply@example.com"))
    semantic_scholar_key: str = field(default_factory=lambda: _env("SEMANTIC_SCHOLAR_API_KEY"))
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))

    # --- 闸门 ---
    jev_api_key: str = field(default_factory=lambda: _env("JEV_API_KEY"))
    reflex_client: str = field(default_factory=lambda: _env("REFLEX_CLIENT", "local"))

    # --- 运行 ---
    data_dir: Path = field(
        default_factory=lambda: Path(_env("ZHIWEI_DATA_DIR", "./data")).expanduser()
    )
    host: str = field(default_factory=lambda: _env("ZHIWEI_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("ZHIWEI_PORT", 8000))
    long_doc_page_threshold: int = field(
        default_factory=lambda: _env_int("ZHIWEI_LONG_DOC_PAGE_THRESHOLD", 100)
    )
    top_k: int = field(default_factory=lambda: _env_int("ZHIWEI_TOP_K", 8))
    request_timeout: int = field(default_factory=lambda: _env_int("ZHIWEI_TIMEOUT", 120))

    @property
    def has_llm(self) -> bool:
        return bool(self.api_key)

    def sub(self, *parts: str) -> Path:
        """取 data 目录下的子路径，并确保目录存在。"""
        path = self.data_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def describe(self) -> dict:
        """可安全打日志的配置摘要（不含密钥）。"""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "fast_model": self.fast_model,
            "vl_model": self.vl_model,
            "embed_model": self.embed_model,
            "llm_configured": self.has_llm,
            "key_tail": ("*" * 6 + self.api_key[-4:]) if self.api_key else "",
            "data_dir": str(self.data_dir),
        }


settings = Settings()

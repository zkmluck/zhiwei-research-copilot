"""下载演示用的两版论文（Attention Is All You Need v1 / v7）。

演示数据的体积不适合塞进仓库，所以由这个脚本按需拉取。
用法：python tools/fetch_demo_papers.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_testdata"

PAPERS = {
    "1706.03762v1.pdf": "https://arxiv.org/pdf/1706.03762v1",
    "1706.03762v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
}


def fetch(name: str, url: str) -> None:
    target = OUT / name
    if target.exists() and target.stat().st_size > 100_000:
        print(f"[skip] {name} 已存在（{target.stat().st_size // 1024} KB）")
        return
    print(f"[get ] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "zhiwei-demo/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise SystemExit(f"{url} 返回的不是 PDF，可能是被重定向到了摘要页")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"[ok  ] {name} -> {len(data) // 1024} KB")


def main() -> None:
    for name, url in PAPERS.items():
        try:
            fetch(name, url)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name} 下载失败：{exc}")
            print("       （没网也不影响：把任意 PDF 放进 _testdata/ 就能跑离线演示）")
    print(f"\n演示数据目录：{OUT}")


if __name__ == "__main__":
    sys.exit(main())

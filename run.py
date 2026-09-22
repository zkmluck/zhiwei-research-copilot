"""知微 ZhiWei 启动入口：`python run.py` 起后端，浏览器打开 http://127.0.0.1:8000/。

环境变量（全部可选，见 .env.example）：
  ZHIWEI_API_KEY      模型网关密钥，不给也能跑解析/检索/图谱，只是生成类能力降级
  ZHIWEI_BASE_URL     兼容 OpenAI 协议的网关地址
  ZHIWEI_DATA_DIR     数据落盘目录，默认 ./data
"""

from __future__ import annotations

import argparse

from zhiwei.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="知微 ZhiWei · 溯源型科研助手 Agent")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "zhiwei.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

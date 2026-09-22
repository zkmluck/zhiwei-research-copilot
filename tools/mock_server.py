"""纯标准库的假后端：让前端在没有真后端（或没有模型密钥）时也能完整演示。

设计目的：评审环境一条命令就能看到界面 ——
    python tools/mock_server.py --port 8010
假数据全部写在本文件里，不依赖任何第三方包，也不发外网请求。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

PAPERS = [
    {
        "doc_id": "mock-attention-v7",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "affiliations": ["Google Brain", "Google Research"],
        "abstract": "The dominant sequence transduction models are based on complex recurrent networks...",
        "year": 2017,
        "venue": "NeurIPS",
        "arxiv_id": "1706.03762",
        "version": 7,
        "page_count": 15,
        "file_name": "1706.03762v7.pdf",
        "family_id": "1706.03762",
        "is_latest": True,
        "status": "active",
        "needs_ocr": False,
        "ingested_at": "2026-09-22 10:02:11",
    },
    {
        "doc_id": "mock-attention-v1",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "affiliations": ["Google Brain"],
        "abstract": "早期预印本版本。",
        "year": 2017,
        "venue": "",
        "arxiv_id": "1706.03762",
        "version": 1,
        "page_count": 11,
        "file_name": "1706.03762v1.pdf",
        "family_id": "1706.03762",
        "is_latest": False,
        "status": "superseded",
        "needs_ocr": False,
        "ingested_at": "2026-09-22 10:01:40",
    },
    {
        "doc_id": "mock-bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Jacob Devlin", "Ming-Wei Chang"],
        "affiliations": ["Google AI Language"],
        "abstract": "We introduce a new language representation model called BERT...",
        "year": 2019,
        "venue": "NAACL",
        "arxiv_id": "1810.04805",
        "version": 2,
        "page_count": 16,
        "file_name": "1810.04805v2.pdf",
        "family_id": "1810.04805",
        "is_latest": True,
        "status": "active",
        "needs_ocr": True,
        "ingested_at": "2026-09-22 10:05:03",
    },
]

REFERENCE_TEXTS = [
    "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.",
    "Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.",
    "Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.",
    "Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.",
]

TIMELINE = [
    {"year": 2014, "title": "Neural machine translation by jointly learning to align and translate", "role": "基石工作"},
    {"year": 2017, "title": "Attention Is All You Need", "role": "桥接工作"},
    {"year": 2019, "title": "BERT: Pre-training of Deep Bidirectional Transformers", "role": "边缘衍生"},
]

SURVEY_MD = """# Transformer 架构的演进 · 领域综述（自动生成，逐句可溯源）

> 材料范围：3 篇文献，21 条候选结论，经溯源闸门放行 17 条、待核实 3 条、拦下 1 条。

## 一、时间线

| 年份 | 文献 | 图谱角色 |
| --- | --- | --- |
| 2014 | Neural machine translation by jointly learning to align and translate | 基石工作 |
| 2017 | Attention Is All You Need | 桥接工作 |
| 2019 | BERT: Pre-training of Deep Bidirectional Transformers | 边缘衍生 |

## 二、演进脉络

序列转换模型长期以循环或卷积网络为主干 [1]。Bahdanau 等人引入注意力机制缓解长距离依赖 [2]。

2017 年提出的 Transformer 完全以注意力构建编码器与解码器，摒弃循环与卷积 [3]。该工作在 WMT 2014 英德任务上取得 28.4 BLEU [3]。

## 三、基石工作

- **Layer normalization**：PageRank 0.183 位居前 25%，被 2 篇文献引用，处在网络核心层。
- **Neural machine translation by jointly learning to align and translate**：被 3 篇文献实质引用。
"""

FUTURE_ITEMS = [
    {
        "doc_id": "mock-attention-v7",
        "doc_title": "Attention Is All You Need",
        "text": "计划把基于注意力的模型扩展到图像、音频与视频等非文本模态。",
        "status": "open",
        "evidence": "2021 年后的后续工作中检索到 3 篇相关尝试，但均未覆盖音频与视频联合建模。",
        "resolved_by": [],
    },
    {
        "doc_id": "mock-attention-v7",
        "doc_title": "Attention Is All You Need",
        "text": "研究局部受限注意力机制，以更高效地处理超长序列。",
        "status": "resolved",
        "evidence": "Longformer、BigBird 等后续工作已给出成熟的稀疏注意力方案。",
        "resolved_by": ["Longformer: The Long-Document Transformer", "Big Bird: Transformers for Longer Sequences"],
    },
]

CANDIDATES = [
    {
        "source": "arxiv",
        "identifier": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
        "year": 2020,
        "abstract": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation...",
        "url": "https://arxiv.org/abs/2005.11401",
        "citation_count": 4213,
        "has_code": True,
        "code_url": "https://github.com/facebookresearch/DPR",
    },
    {
        "source": "arxiv",
        "identifier": "2004.04906",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": ["Vladimir Karpukhin", "Barlas Oguz"],
        "year": 2020,
        "abstract": "Open-domain question answering relies on efficient passage retrieval...",
        "url": "https://arxiv.org/abs/2004.04906",
        "citation_count": 2810,
        "has_code": True,
        "code_url": "https://github.com/facebookresearch/DPR",
    },
]

CLAIMS = [
    {
        "text": "训练使用了 8 张 NVIDIA P100 GPU。",
        "kind": "number",
        "evidence": [{
            "doc_id": "mock-attention-v7",
            "doc_title": "Attention Is All You Need",
            "source": "text",
            "score": 0.91,
            "anchor": {
                "page": 7,
                "quote": "We trained our models on one machine with 8 NVIDIA P100 GPUs.",
                "char_start": 1204, "char_end": 1250,
                "bbox": {"x0": 96.0, "y0": 320.5, "x1": 470.0, "y1": 336.0},
                "block_kind": "text",
            },
        }],
        "decision": {
            "verdict": "allow", "confidence": 0.89,
            "reasons": ["原文片段直接陈述了 8 张 NVIDIA P100 GPU", "术语覆盖充分"],
            "checks": {"evidence_count": 1, "located": True, "numbers_ok": True},
        },
    },
    {
        "text": "基础模型训练了 12 小时（共 100,000 步）。",
        "kind": "number",
        "evidence": [{
            "doc_id": "mock-attention-v7",
            "doc_title": "Attention Is All You Need",
            "source": "text",
            "score": 0.87,
            "anchor": {
                "page": 7,
                "quote": "We trained the base models for a total of 100,000 steps or 12 hours.",
                "char_start": 1500, "char_end": 1560,
                "bbox": {"x0": 96.0, "y0": 352.5, "x1": 470.0, "y1": 368.0},
                "block_kind": "text",
            },
        }],
        "decision": {
            "verdict": "allow", "confidence": 1.0,
            "reasons": ["原文同时给出步数与时长，数字可逐一对上"],
            "checks": {"evidence_count": 1, "located": True, "numbers_ok": True},
        },
    },
    {
        "text": "作者认为该架构可以完全替代卷积网络。",
        "kind": "inference",
        "evidence": [],
        "decision": {
            "verdict": "block", "confidence": 0.12,
            "reasons": ["检索到的原文中没有支持该断言的材料", "属于推断而非原文陈述"],
            "checks": {"evidence_count": 0, "located": False, "numbers_ok": True},
        },
    },
]

QA_TEXT = (
    "训练使用了 8 张 NVIDIA P100 GPU。\n"
    "基础模型训练了 12 小时（共 100,000 步）。\n"
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Agent 演示数据：假后端也要如实演示「规划 → 调工具 → 记观察 → 收尾」的事件形状
# ---------------------------------------------------------------------------

MOCK_TOOLS = [
    {"name": "library.stats", "description": "看清本地文献库里有什么：篇数、页数、切块数、版本情况。", "params": {}, "network": False, "cost": "cheap"},
    {"name": "library.search", "description": "在本地文献库里做混合检索（BM25 + 向量 + RRF），返回带页码锚点的原文证据。", "params": {"query": "检索词", "top_k": "返回条数"}, "network": False, "cost": "model"},
    {"name": "graph.build", "description": "构建引用图谱：节点、边、四种角色（基石 / 桥接 / 衍生 / 孤岛）。", "params": {"doc_ids": "只看这几篇"}, "network": False, "cost": "cheap"},
    {"name": "graph.key_nodes", "description": "挖出最核心的节点，并给出「为什么它是核心」的人话理由。", "params": {"limit": "返回条数"}, "network": False, "cost": "cheap"},
    {"name": "future_work.scan", "description": "抽取 Future Work 并判定是否已被后续工作解决。", "params": {"max_items": "每篇最多几条"}, "network": False, "cost": "model"},
    {"name": "survey.generate", "description": "沿引用图谱生成领域综述草稿，每一句都要过主张闸门。", "params": {"topic": "综述主题"}, "network": False, "cost": "model"},
    {"name": "writing.outline", "description": "从 Idea 生成论文框架，References 只能来自本地库。", "params": {"idea": "论文想法"}, "network": False, "cost": "model"},
    {"name": "discover.search", "description": "去外部学术源检索新论文。需要外网。", "params": {"query": "检索词", "limit": "返回条数"}, "network": True, "cost": "network"},
]

MOCK_PLAN = {
    "source": "offline",
    "stop_when": "拿到足以支撑结论的原文证据即可收尾",
    "note": "mock 模式演示确定性计划的形状",
    "steps": [
        {"index": 0, "tool": "library.stats", "args": {}, "why": "先知道库里有什么，才知道能回答到什么程度", "label": "先看清库里有什么", "source": "offline"},
        {"index": 1, "tool": "library.search", "args": {"query": "训练的硬件与时长", "top_k": 8}, "why": "调研目标本身就是最好的检索式", "label": "检索证据", "source": "offline"},
        {"index": 2, "tool": "graph.build", "args": {}, "why": "目标涉及脉络与关系，需要引用图谱", "label": "构建引用图谱", "source": "offline"},
        {"index": 3, "tool": "future_work.scan", "args": {"max_items": 4}, "why": "目标是找空白，需要知道哪些待办已被解决", "label": "扫描 Future Work 时效性", "source": "offline"},
    ],
}

MOCK_OBSERVATIONS = {
    "library.stats": {"summary": "本地库共 2 篇文献、30 页、174 个检索切块", "ms": 60,
                      "highlights": ["Attention Is All You Need（2017，v7）"],
                      "payload": {"papers": [{"doc_id": "mock-attention-v7", "title": "Attention Is All You Need"}], "total_pages": 30, "total_chunks": 174}},
    "library.search": {"summary": "检索「训练的硬件与时长」命中 3 条带锚点的证据", "ms": 420,
                       "highlights": ["p7：We trained the base model on 8 NVIDIA P100 GPUs for 12 hours."],
                       "payload": {"evidence": [{"doc_id": "mock-attention-v7", "page": 7, "quote": "We trained the base model on 8 NVIDIA P100 GPUs for 12 hours."}]}},
    "graph.build": {"summary": "引用图谱：60 个节点、76 条边；cornerstone 5、bridge 3、derivative 8", "ms": 2100,
                    "highlights": ["Layer Normalization：PageRank 位居前 25%，被 3 篇文献引用"],
                    "payload": {"roles": {"cornerstone": ["ext:layer-normalization"], "bridge": ["mock-attention-v7"]}, "evidence": []}},
    "future_work.scan": {"summary": "扫出 4 条 Future Work，状态分布：open 2、resolved 1、partially 1", "ms": 1600,
                         "highlights": ["[resolved] 用更长的序列验证注意力机制的可扩展性"],
                         "payload": {"items": [{"text": "用更长的序列验证注意力机制的可扩展性", "status": "resolved"}]}},
}

MOCK_FINDINGS = {
    "library": {"papers": 2, "total_pages": 30, "total_chunks": 174},
    "graph_roles": {"cornerstone": ["ext:layer-normalization"], "bridge": ["mock-attention-v7"], "derivative": ["ext:xception"]},
    "key_nodes": [{"doc_id": "ext:layer-normalization", "title": "Layer Normalization", "role": "cornerstone",
                   "reason": "PageRank 位居前 25%，被 3 篇文献引用，处在网络核心层（k-core=2）"}],
    "future_work_status": {"open": 2, "resolved": 1, "partially": 1},
    "future_work": [{"text": "用更长的序列验证注意力机制的可扩展性", "status": "resolved"}],
    "evidence": [{"doc_id": "mock-attention-v7", "page": 7, "quote": "We trained the base model on 8 NVIDIA P100 GPUs for 12 hours."}],
}


class Handler(BaseHTTPRequestHandler):
    server_version = "ZhiWeiMock/0.1"

    def log_message(self, fmt, *args):  # noqa: A003 - 保持安静
        return

    # ------------------------------------------------------------ 工具
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._json({"error": {"kind": "not_found", "message": str(path)}}, 404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".mjs":
            ctype = "text/javascript"
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8" if ctype.startswith("text") or ctype.endswith("javascript") else ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------ 路由
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/agent/tools":
            return self._json({"tools": MOCK_TOOLS, "count": len(MOCK_TOOLS)})
        if path == "/api/health":
            return self._json({
                "status": "ok", "version": "0.1.0-mock", "llm_configured": False,
                "models": {"model": "mock", "base_url": "mock://local", "llm_configured": False, "data_dir": "mock"},
                "gate": {"allow_at": 0.75, "block_at": 0.35},
                "library": {"papers": len(PAPERS)},
            })
        if path == "/api/usage":
            return self._json({
                "calls": 42, "failed_calls": 1, "prompt_tokens": 18422,
                "completion_tokens": 3201, "total_cost_yuan": 0.0158,
                "by_model": {"qwen3.7-plus": {"calls": 30, "total_cost_yuan": 0.012}},
            })
        if path == "/api/papers":
            return self._json({"papers": PAPERS})
        if path.startswith("/api/papers/"):
            return self._paper_route(path)
        if path == "/api/graph":
            return self._json(_graph_payload())
        if path == "/api/graph/citations":
            return self._json({"doc_id": (query.get("doc_id") or [""])[0], "edges": _graph_payload()["edges"]})
        if path == "/api/survey":
            return self._json({"markdown": SURVEY_MD, "claims": CLAIMS, "timeline": TIMELINE,
                               "stats": {"materials": 21, "allowed": 17, "downgraded": 3, "blocked": 1}})
        if path == "/api/future-work":
            return self._json({"items": FUTURE_ITEMS, "stats": {"total": len(FUTURE_ITEMS), "open": 1, "resolved": 1}})
        if path == "/api/recommend":
            items = [{**c, "reason": "与库内 Transformer 系列高度相关，且带官方开源实现",
                      "score_parts": {"relevance": 0.82, "recency": 0.44, "code": 1.0, "impact": 0.91}} for c in CANDIDATES]
            return self._json({"query": "Attention Is All You Need", "items": items,
                               "weights": {"relevance": 0.45, "recency": 0.15, "code": 0.2, "impact": 0.2}})
        if path == "/api/track/status":
            return self._json({"last_run": _now(), "tracked_topics": ["retrieval augmented generation"], "new_items": 2})
        if path == "/api/writing/artifacts":
            return self._json({"artifacts": []})
        if path.startswith("/api/writing/artifacts/"):
            return self._json({"error": {"kind": "not_found", "message": "mock 模式没有真实图片"}}, 404)

        return self._static(path)

    def _paper_route(self, path: str):
        parts = [p for p in path.split("/") if p]
        doc_id = parts[2] if len(parts) > 2 else ""
        paper = next((p for p in PAPERS if p["doc_id"] == doc_id), None)
        if paper is None:
            return self._json({"error": {"kind": "not_found", "message": doc_id}}, 404)

        if len(parts) == 3:
            return self._json({
                "record": paper,
                "sections": _sections(),
                "outline": _outline(),
                "stats": {"pages": paper["page_count"], "chunks": 87, "tables": 9,
                          "figures": 3, "formulas": 6, "references": len(REFERENCE_TEXTS)},
                "warnings": ["第 11 页缺少原生文本层（已走 OCR 回填）"] if paper["needs_ocr"] else [],
            })
        if parts[3] == "page":
            return self._json(_page_payload(int(parts[4]) if len(parts) > 4 else 1))
        if parts[3] == "references":
            refs = []
            for index, raw in enumerate(REFERENCE_TEXTS, start=1):
                title = raw.split(". ", 1)[1] if ". " in raw else raw
                refs.append({
                    "index": index, "raw": raw, "doc_id": "", "in_library": index == 2,
                    "anchor": {"page": 9, "quote": raw[:120], "char_start": 100 * index,
                               "char_end": 100 * index + 100, "bbox": None, "block_kind": "reference"},
                    "resolved_title": title[:70], "year": 2016 + index, "url": "", "arxiv_id": "",
                })
            return self._json({"references": refs})
        if parts[3] == "xref":
            index = int(parse_qs(urlparse(self.path).query).get("index", ["1"])[0])
            return self._json({
                "in_text": [{"page": 4, "quote": "We employ a residual connection around each of the two sub-layers.",
                             "mark": f"[{index}]", "char_start": 220, "char_end": 224,
                             "bbox": {"x0": 96.0, "y0": 210.0, "x1": 470.0, "y1": 226.0}, "block_kind": "text"}],
                "reference": {"index": index, "raw": REFERENCE_TEXTS[min(index - 1, len(REFERENCE_TEXTS) - 1)]},
            })
        if parts[3] == "versions":
            family = [p for p in PAPERS if p["family_id"] == paper["family_id"]]
            return self._json({"family_id": paper["family_id"], "versions": [
                {"doc_id": p["doc_id"], "version": p["version"], "is_latest": p["is_latest"],
                 "title": p["title"], "status": p["status"], "ingested_at": p["ingested_at"],
                 "file_name": p["file_name"]} for p in family]})
        if parts[3] == "file":
            return self._json({"error": {"kind": "no_file", "message": "mock 模式不提供真实 PDF"}}, 404)
        return self._json({"error": {"kind": "not_found", "message": path}}, 404)

    def do_DELETE(self):  # noqa: N802
        return self._json({"deleted": True})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/qa/stream":
            return self._sse_qa(body)
        if path == "/api/agent/plan":
            return self._json(MOCK_PLAN)
        if path == "/api/agent/run":
            return self._sse_agent(body)
        if path == "/api/qa":
            return self._json({"question": body.get("question", ""), "text": QA_TEXT,
                               "claims": CLAIMS, "refused": False, "support_rate": 0.67, "trace": []})
        if path == "/api/extract":
            return self._json({"items": [
                {"field": "training_gpu", "value": "NVIDIA P100", "unit": "",
                 "evidence": CLAIMS[0]["evidence"], "decision": CLAIMS[0]["decision"]},
                {"field": "bleu", "value": "28.4", "unit": "BLEU",
                 "evidence": CLAIMS[1]["evidence"], "decision": CLAIMS[1]["decision"]},
            ]})
        if path == "/api/compare":
            return self._json({"aspects": body.get("aspects", []), "table": [], "summary_claims": []})
        if path == "/api/graph/build":
            return self._json({"stats": _graph_payload()["metrics"], "metrics": _graph_payload()["metrics"], "nodes": 10})
        if path == "/api/discover/search":
            return self._json({"query": body.get("query", ""), "results": CANDIDATES})
        if path == "/api/discover/ingest":
            return self._json({"doc_id": "mock-new", "status": "created", "dedup": None})
        if path == "/api/discover/references":
            return self._json({"doc_id": body.get("doc_id", ""), "seed_references": 40, "candidates": CANDIDATES})
        if path == "/api/feedback":
            return self._json({"ok": True, "stats": {"liked": 1, "disliked": 0}})
        if path == "/api/track/run":
            return self._json({"topics": body.get("topics", []), "scanned": 42,
                               "new_items": CANDIDATES, "new_count": len(CANDIDATES), "ran_at": _now()})
        if path == "/api/papers/upload":
            return self._json({"results": [{"doc_id": "mock-uploaded", "file_name": "uploaded.pdf",
                                            "status": "created", "dedup": None, "record": PAPERS[0]}]})
        if path == "/api/writing/outline":
            return self._json({
                "idea": body.get("idea", ""), "venue": body.get("venue", "generic"),
                "abstract": "我们提出一种把引用图谱先验注入检索排序的方法，在保持可溯源的前提下提升长尾问题的召回。",
                "introduction": "科研问答对确定性要求极高。现有检索增强生成在长尾问题上召回不足。[1][2]",
                "related_work": "检索增强生成 [1] 与稠密检索 [2] 是本文的两条主线。",
                "references": [{"index": 1, "raw": REFERENCE_TEXTS[1], "doc_id": "mock-attention-v1"},
                               {"index": 2, "raw": REFERENCE_TEXTS[0], "doc_id": "mock-attention-v7"}],
                "unsupported": [{"text": "该方法可以完全替代微调。", "reason": "库内没有支持该断言的原文"}],
                "stats": {"candidate_claims": 8, "allowed": 6, "blocked": 1, "references": 2},
            })
        if path == "/api/writing/diagram":
            kind = body.get("kind", "mermaid")
            code = {
                "mermaid": "flowchart LR\n  A[PDF 解析] --> B[切块与索引]\n  B --> C[混合检索]\n  C --> D[主张闸门]",
                "graphviz": "digraph G {\n  rankdir=LR;\n  A -> B -> C -> D;\n}",
                "tikz": "% 需要 pgf/tikz 宏包\n\\begin{tikzpicture}\n  \\node (a) {PDF}; \\node[right=of a] (b) {Index};\n  \\draw[->] (a) -- (b);\n\\end{tikzpicture}",
            }.get(kind, "")
            return self._json({"kind": kind, "code": code, "render_hint": "mock", "nodes": 4, "edges": 3})
        if path == "/api/writing/figure":
            return self._json({"chart_type": body.get("chart", "line"), "title": body.get("title", ""),
                               "image_url": "", "script": "import matplotlib.pyplot as plt",
                               "caption": "图 1：随训练步数增加，验证集 BLEU 稳步上升。"})
        if path == "/api/translate/segment":
            return self._json({"translated": f"【译文】{body.get('text', '')[:80]}", "terms": []})

        return self._json({"error": {"kind": "not_found", "message": path}}, 404)

    # ------------------------------------------------------------ SSE
    def _sse_agent(self, body: dict):
        """假后端也要如实演示「规划 → 调工具 → 记观察 → 收尾过闸门」的事件形状。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event: str, payload: dict, delay: float = 0.2):
            chunk = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
            time.sleep(delay)

        goal = body.get("goal", "")
        try:
            send("plan", {**MOCK_PLAN, "goal": goal, "replanned": False})
            for step in MOCK_PLAN["steps"]:
                send("step", {"step": step["index"] + 1, "tool": step["tool"], "status": "running",
                              "label": step["label"], "why": step["why"], "args": step["args"], "round": 1}, 0.15)
                send("tool", {"step": step["index"] + 1, "name": step["tool"], "detail": step["why"]}, 0.05)
                observation = MOCK_OBSERVATIONS[step["tool"]]
                send("observation", {"step": step["index"] + 1, "tool": step["tool"], "ok": True,
                                     "summary": observation["summary"], "highlights": observation["highlights"],
                                     "ms": observation["ms"], "payload": observation["payload"],
                                     "evidence_count": len(observation["payload"].get("evidence", [])),
                                     "memory": {"observations": step["index"] + 1, "folded": 0, "chars": 1200 + 400 * step["index"],
                                                "budget_chars": 8000, "tools": {}, "evidence": 3}}, 0.2)
                send("step", {"step": step["index"] + 1, "tool": step["tool"], "status": "done",
                              "label": step["label"], "ms": observation["ms"], "ok": True,
                              "summary": observation["summary"], "round": 1}, 0.12)
            for claim in CLAIMS:
                send("claim", claim, 0.3)
            send("done", {
                "goal": goal, "plan": {"rounds": 1, "sources": ["offline"]},
                "observations": [{"step": i + 1, "tool": s["tool"], "ok": True} for i, s in enumerate(MOCK_PLAN["steps"])],
                "findings": MOCK_FINDINGS, "memory": {"observations": len(MOCK_PLAN["steps"]), "folded": 0,
                                                      "compressions": 0, "chars": 3200, "budget_chars": 8000,
                                                      "tools": {}, "evidence": 3},
                "text": QA_TEXT, "refused": False, "support_rate": 0.67,
                "claim_counts": {"allow": 2, "downgrade": 0, "block": 1},
                "claims": CLAIMS, "stopped_early": False, "stop_reason": "",
                "ms": 4800, "usage": {"calls": 4, "total_cost_yuan": 0.0009},
            }, 0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse_qa(self, body: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(event: str, payload: dict, delay: float = 0.25):
            chunk = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
            time.sleep(delay)

        try:
            send("trace", {"step": "rewrite", "detail": f"把问题改写成 3 条检索式：{body.get('question', '')[:24]}", "ms": 90}, 0.2)
            send("trace", {"step": "retrieve", "detail": "混合检索（BM25 + 向量 + RRF 融合）命中 8 条证据", "ms": 210}, 0.2)
            send("trace", {"step": "gate", "detail": "逐句过主张闸门：3 条主张，放行 2、拦下 1", "ms": 480}, 0.2)
            send("evidence", {"evidence": [c["evidence"][0] for c in CLAIMS if c["evidence"]]}, 0.2)
            for claim in CLAIMS:
                send("claim", claim, 0.35)
            for sentence in QA_TEXT.split("\n"):
                if sentence.strip():
                    send("token", {"text": sentence + "\n"}, 0.25)
            send("done", {"answer_id": "mock-1", "support_rate": 0.67, "refused": False,
                          "usage": {"calls": 3, "total_cost_yuan": 0.0004}}, 0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ------------------------------------------------------------ 静态
    def _static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        target = (WEB / rel).resolve()
        if not str(target).startswith(str(WEB.resolve())):
            return self._json({"error": {"kind": "forbidden", "message": rel}}, 403)
        if target.is_dir():
            target = target / "index.html"
        return self._file(target)


def _sections():
    return [
        {"section_id": "s1", "level": 1, "title": "1 Introduction", "number": "1", "page": 2, "parent": None},
        {"section_id": "s2", "level": 1, "title": "3 Model Architecture", "number": "3", "page": 2, "parent": None},
        {"section_id": "s21", "level": 2, "title": "3.2 Attention", "number": "3.2", "page": 3, "parent": "s2"},
        {"section_id": "s3", "level": 1, "title": "5 Training", "number": "5", "page": 7, "parent": None},
        {"section_id": "s31", "level": 2, "title": "5.2 Hardware and Schedule", "number": "5.2", "page": 7, "parent": "s3"},
    ]


def _outline():
    secs = _sections()
    nodes = {s["section_id"]: {**s, "children": []} for s in secs}
    roots = []
    for s in secs:
        if s["parent"] and s["parent"] in nodes:
            nodes[s["parent"]]["children"].append(nodes[s["section_id"]])
        else:
            roots.append(nodes[s["section_id"]])
    return roots


def _page_payload(page: int) -> dict:
    blocks = [
        {"kind": "heading", "text": "3.2 Attention", "bbox": {"x0": 96.0, "y0": 96.0, "x1": 300.0, "y1": 116.0},
         "section_path": ["3 Model Architecture", "3.2 Attention"], "index": 0, "extra": {}},
        {"kind": "paragraph", "text": "An attention function can be described as mapping a query and a set of key-value pairs to an output.",
         "bbox": {"x0": 96.0, "y0": 130.0, "x1": 300.0, "y1": 210.0},
         "section_path": ["3 Model Architecture", "3.2 Attention"], "index": 1, "extra": {}},
        {"kind": "formula", "text": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V",
         "bbox": {"x0": 320.0, "y0": 130.0, "x1": 480.0, "y1": 160.0},
         "section_path": ["3 Model Architecture", "3.2 Attention"], "index": 2, "extra": {}},
        {"kind": "paragraph", "text": "We employ a residual connection around each of the two sub-layers, followed by layer normalization [11].",
         "bbox": {"x0": 320.0, "y0": 175.0, "x1": 480.0, "y1": 250.0},
         "section_path": ["3 Model Architecture", "3.2 Attention"], "index": 3, "extra": {}},
        {"kind": "table", "text": "| Layer | d_model | heads |\n| --- | --- | --- |\n| base | 512 | 8 |\n| big | 1024 | 16 |",
         "bbox": {"x0": 96.0, "y0": 270.0, "x1": 480.0, "y1": 330.0},
         "section_path": ["3 Model Architecture"], "index": 4, "extra": {}},
    ]
    return {
        "page": page, "width": 612.0, "height": 792.0, "blocks": blocks,
        "text": " ".join(b["text"] for b in blocks),
    }


def _graph_payload() -> dict:
    nodes = [
        {"node_id": "mock-attention-v7", "title": "Attention Is All You Need", "year": 2017,
         "pagerank": 0.183, "betweenness": 0.42, "in_degree": 3, "out_degree": 4, "k_core": 3,
         "role": "bridge", "reason": "中介中心性位居前 25%，连接不同引用簇", "in_library": True,
         "distance": 1, "symbolSize": 42},
        {"node_id": "mock-bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2019,
         "pagerank": 0.121, "betweenness": 0.18, "in_degree": 0, "out_degree": 4, "k_core": 2,
         "role": "derivative", "reason": "以库内基石工作为主要引用来源，自身尚未被广泛引用", "in_library": True},
        {"node_id": "ext:layer-normalization", "title": "Layer normalization", "year": 2016,
         "pagerank": 0.155, "betweenness": 0.31, "in_degree": 2, "out_degree": 0, "k_core": 3,
         "role": "cornerstone", "reason": "PageRank 位居前 25%，被 2 篇文献实质引用", "in_library": False},
        {"node_id": "ext:neural-machine-translation-by-jointly-learning", "title": "Neural machine translation by jointly learning to align and translate",
         "year": 2014, "pagerank": 0.144, "betweenness": 0.22, "in_degree": 3, "out_degree": 0, "k_core": 2,
         "role": "cornerstone", "reason": "被 3 篇文献实质引用，是注意力的源头工作", "in_library": False},
        {"node_id": "ext:xception", "title": "Xception: Deep learning with depthwise separable convolutions",
         "year": 2016, "pagerank": 0.061, "betweenness": 0.0, "in_degree": 0, "out_degree": 0, "k_core": 0,
         "role": "isolated", "reason": "在该局域网络中没有任何引用关系", "in_library": False},
    ]
    edges = [
        {"src_doc_id": "mock-attention-v7", "dst_doc_id": "ext:layer-normalization",
         "dst_title": "Layer normalization", "intent": "uses", "value_score": 0.8, "substantive": True,
         "context": "We employ a residual connection around each of the two sub-layers, followed by layer normalization [11].",
         "page": 3, "src": "mock-attention-v7", "target": "ext:layer-normalization"},
        {"src_doc_id": "mock-attention-v7", "dst_doc_id": "ext:neural-machine-translation-by-jointly-learning",
         "dst_title": "Neural machine translation by jointly learning to align and translate",
         "intent": "compares", "value_score": 0.9, "substantive": True,
         "context": "The Transformer allows for significantly more parallelization... unlike [2].", "page": 2},
        {"src_doc_id": "mock-bert", "dst_doc_id": "mock-attention-v7", "dst_title": "Attention Is All You Need",
         "intent": "extends", "value_score": 0.95, "substantive": True,
         "context": "BERT is built on the Transformer encoder [3].", "page": 3},
        {"src_doc_id": "mock-bert", "dst_doc_id": "ext:layer-normalization", "dst_title": "Layer normalization",
         "intent": "lists", "value_score": 0.2, "substantive": False,
         "context": "Many works study normalization [11, 12, 13].", "page": 4},
    ]
    return {"nodes": nodes, "edges": edges,
            "metrics": {"nodes": 18, "edges": 26, "density": 0.085, "components": 1,
                        "core_size": 3, "pagerank_p75": 0.15, "substantive_ratio": 0.62}}


def main() -> None:
    parser = argparse.ArgumentParser(description="知微 ZhiWei 前端 mock 后端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock 后端已启动：http://{args.host}:{args.port}/  （静态目录 {WEB}）")
    print("按 Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

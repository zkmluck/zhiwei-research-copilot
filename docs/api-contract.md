# 知微 ZhiWei · 前后端接口契约（冻结版 v1）

> 本文件是前端与后端的唯一接口约定。任何一方要改，先改这里。
> 所有接口前缀 `/api`，请求/响应均为 JSON（文件上传除外）。
> 错误统一返回 `{"error": {"kind": "...", "message": "..."}}`，HTTP 4xx/5xx。

## 0. 通用约定

**锚点（Anchor）** —— 全系统溯源的原子结构，前端据此在 PDF 上画高亮：

```json
{
  "page": 3,
  "quote": "We trained on 8 NVIDIA P100 GPUs for 12 hours.",
  "char_start": 1204, "char_end": 1253,
  "bbox": {"x0": 72.0, "y0": 320.5, "x1": 288.0, "y1": 336.0},
  "block_kind": "text"
}
```

**证据（Evidence）**：`{"doc_id","anchor","score","source","doc_title"}`

**主张（Claim）**：`{"claim_id","text","kind","evidence":[...],"decision":{"verdict":"allow|downgrade|block","confidence","reasons":[...],"checks":{}}}`

**用量**：`GET /api/usage` → `{"calls","failed_calls","prompt_tokens","completion_tokens","total_cost_yuan","by_model":{...}}`

---

## 1. 健康与配置

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | `{"status":"ok","version":"0.1.0","llm_configured":true,"models":{...}}` |
| GET | `/api/usage` | 用量账本 |

## 2. 文献库 CRUD 与去重（模块一）

| 方法 | 路径 | 请求 | 响应 |
| --- | --- | --- | --- |
| POST | `/api/papers/upload` | multipart `files[]` | `{"results":[{"doc_id","file_name","status":"created|duplicate|needs_decision","dedup":DedupVerdict|null,"record":PaperRecord}]}` |
| GET | `/api/papers` | — | `{"papers":[PaperRecord]}` |
| GET | `/api/papers/{doc_id}` | — | `{"record","sections":[Section],"outline":[tree],"stats":{"pages","chunks","tables","figures","formulas","references"},"warnings":[]}` |
| DELETE | `/api/papers/{doc_id}` | — | `{"deleted":true}` |
| POST | `/api/papers/{doc_id}/dedup/resolve` | `{"action":"keep_existing|replace|keep_both"}` | `{"record","status"}` |
| GET | `/api/papers/{doc_id}/versions` | — | `{"family_id","versions":[{"doc_id","version","is_latest","title","ingested_at"}]}` |
| GET | `/api/papers/{doc_id}/diff?against={doc_id}` | — | `{"summary","changes":[{"kind":"added|removed|modified","section","page","before","after","reason"}]}` |

## 3. 版式感知阅读（模块三）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/papers/{doc_id}/page/{page}` | `{"page","width","height","blocks":[{"kind","text","bbox","section_path","index"}],"text"}` |
| GET | `/api/papers/{doc_id}/file` | 原始 PDF 字节流（供 PDF.js 加载） |
| GET | `/api/papers/{doc_id}/references` | `{"references":[{"index","raw","doc_id","anchor","resolved_title","year","url","in_library":bool}]}` |
| GET | `/api/papers/{doc_id}/xref?index=11` | 正文中 `[11]` 的跳转锚点 `{"in_text":[Anchor],"reference":{...}}` |
| POST | `/api/translate/segment` | `{"text","to":"zh|en"}` → `{"translated","terms":[...]}` |
| GET | `/api/papers/{doc_id}/translate?to=zh&page=N` | 整页译文 `{"page","markdown","segments":[{"index","src","dst","bbox"}]}` |

## 4. 抗幻觉问答（模块二）

`POST /api/qa/stream`（SSE，`text/event-stream`）

请求：`{"question","doc_ids":[],"mode":"single|cross|auto","top_k":8}`

事件序列：

```
event: trace   data: {"step":"rewrite","detail":"把问题改写成 3 条检索式","ms":120}
event: evidence data: {"evidence":[Evidence,...]}
event: claim   data: Claim            # 逐句到达，含闸门裁决
event: token   data: {"text":"..."}   # 流式正文片段
event: done    data: {"answer_id","support_rate","refused":bool,"usage":{...}}
event: error   data: {"kind","message"}
```

非流式版本：`POST /api/qa` → `Answer`（含 `claims`、`support_rate`、`refused`）

细粒度抽取：`POST /api/extract` `{"doc_id","fields":["training_gpu","batch_size","bleu","affiliation"]}` → `{"items":[{"field","value","unit","evidence":[Evidence],"decision":GateDecision}]}`

跨文献对比：`POST /api/compare` `{"doc_ids":[],"aspects":[...]}` → `{"aspects":[...],"table":[{"aspect","cells":[{"doc_id","value","evidence":[Evidence],"decision"}]}],"summary_claims":[Claim]}`

## 5. 引用图谱与综述（模块五）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/graph/build` | `{"doc_ids":[],"expand_external":true}` → `{"stats":{...}}`（异步可轮询） |
| GET | `/api/graph?doc_ids=a,b` | `{"nodes":[{"doc_id","title","year","role","pagerank","betweenness","in_degree","out_degree","k_core","in_library","reason"}],"edges":[{"source","target","intent","value_score","substantive","context","page"}],"metrics":{"density","components","core_size"}}` |
| GET | `/api/graph/citations?doc_id=` | 单篇的引用上下文与含金量明细 |
| GET | `/api/survey?doc_ids=` | `{"markdown","claims":[Claim],"timeline":[{"year","count","doc_ids":[]}]}` |
| GET | `/api/future-work?doc_ids=` | `{"items":[{"doc_id","text","anchor","status":"open|resolved|stale","resolved_by":[],"evidence"}]}` |

## 6. 检索、追踪与推荐（模块四）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/discover/search` | `{"query","limit":20}` → `{"results":[{"source":"arxiv|openalex","identifier","title","authors","year","abstract","url","pdf_url","has_code","code_url","citation_count"}]}` |
| POST | `/api/discover/ingest` | `{"identifier","source":"arxiv"}` → `{"doc_id","status"}`（下钻下载并入库） |
| POST | `/api/discover/references` | `{"doc_id"}` → 解析该文参考文献并批量检索候选 |
| GET | `/api/recommend?doc_id=&days=30` | `{"items":[{"identifier","title","reason","relevance","has_code","code_url","score_parts":{...}}]}` |
| POST | `/api/feedback` | `{"doc_id","action":"like|dislike"}` → `{"ok":true}` |
| GET | `/api/track/status` | `{"last_run","tracked_topics":[],"new_items":n}` |
| POST | `/api/track/run` | `{"topics":["..."],"since_days":7}` → 立即跑一次追踪 |

## 7. 写作与可视化 Copilot（模块六）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/writing/outline` | `{"idea","doc_ids":[],"venue":"generic"}` → `{"abstract":Claim,"introduction":[Claim],"related_work":[Claim],"references":[{"index","raw","doc_id","evidence"}],"unsupported":[]}` |
| POST | `/api/writing/diagram` | `{"kind":"mermaid|graphviz|tikz","spec":"..."}` → `{"code","kind","render_hint"}` |
| POST | `/api/writing/figure` | `{"csv_text","chart":"line|bar|radar|table","title","x","y"}` → `{"image_url","script","caption","stats":{...},"claims":[Claim]}` |
| POST | `/api/writing/translate` | `{"text","to"}` → `{"translated"}`（学术中译英） |
| GET | `/api/writing/artifacts` | 已产出图表/脚本清单 |

## 8. Agent 编排可视化

`POST /api/agent/run`（SSE）`{"goal":"帮我调研 XXX 方向","doc_ids":[]}`

事件：`event: plan` / `event: node` / `event: tool` / `event: claim` / `event: done`

```json
{"node":"planner","status":"running|done|failed","label":"任务规划","detail":"拆成 4 个子目标","ms":120}
```
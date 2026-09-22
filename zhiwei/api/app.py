"""FastAPI 应用：按 docs/api-contract.md 暴露全部能力。

设计取向：路由只做参数校验与序列化，编排都在 service / 各能力模块里。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..agents import planner, stream_agent
from ..config import settings
from ..ingest import pdf_parser
from ..ingest.version_diff import diff_versions
from . import deps, service
from .serialize import error_payload, to_jsonable


def _json(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(to_jsonable(payload), status_code=status_code)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(to_jsonable(data), ensure_ascii=False)}\n\n"


def _require_doc(doc_id: str):
    record = deps.library().get_paper(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=error_payload("not_found", f"库中没有这篇文献：{doc_id}"))
    return record


def create_app() -> FastAPI:
    app = FastAPI(title="知微 ZhiWei API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def _http_error(_request, exc: HTTPException):  # pragma: no cover - 框架回调
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(detail, status_code=exc.status_code)
        return JSONResponse(error_payload("http_error", str(detail)), status_code=exc.status_code)

    # ---------------------------------------------------------------- 健康
    @app.get("/api/health")
    def health() -> JSONResponse:
        gw = deps.gateway()
        return _json(
            {
                "status": "ok",
                "version": __version__,
                "llm_configured": bool(gw.available),
                "models": settings.describe(),
                "gate": deps.gate().config.to_dict(),
                "library": {"papers": len(service.library_view())},
            }
        )

    @app.get("/api/usage")
    def usage() -> JSONResponse:
        return _json(deps.gateway().ledger.summary())

    # ---------------------------------------------------------------- 模块一
    @app.post("/api/papers/upload")
    async def upload(files: list[UploadFile] = File(...)) -> JSONResponse:
        results: list[dict] = []
        for item in files:
            filename = item.filename or "upload.pdf"
            try:
                data = await item.read()
                staged = service.stage_upload(filename, data)
                results.append(service.ingest_path(staged, file_name=filename))
            except Exception as exc:  # noqa: BLE001 - 单篇失败不该拖垮整批
                results.append(
                    {
                        "doc_id": "",
                        "file_name": filename,
                        "status": "failed",
                        "dedup": None,
                        "record": None,
                        "error": {"kind": type(exc).__name__, "message": str(exc)},
                    }
                )
        service.invalidate_graph()
        return _json({"results": results})

    @app.get("/api/papers")
    def list_papers() -> JSONResponse:
        return _json({"papers": service.library_view()})

    @app.get("/api/papers/{doc_id}")
    def paper_detail(doc_id: str) -> JSONResponse:
        detail = service.paper_detail(doc_id)
        if detail is None:
            raise HTTPException(404, error_payload("not_found", doc_id))
        return _json(detail)

    @app.delete("/api/papers/{doc_id}")
    def delete_paper(doc_id: str) -> JSONResponse:
        deleted = service.delete_paper(doc_id)
        service.invalidate_graph()
        return _json({"deleted": deleted})

    @app.post("/api/papers/{doc_id}/dedup/resolve")
    def resolve_dedup(doc_id: str, body: dict = Body(...)) -> JSONResponse:
        action = str(body.get("action") or "")
        if action not in {"keep_existing", "replace", "keep_both"}:
            raise HTTPException(400, error_payload("bad_action", f"不认识的动作：{action}"))
        try:
            result = service.resolve_dedup(doc_id, action)
        except KeyError:
            raise HTTPException(404, error_payload("not_found", doc_id)) from None
        service.invalidate_graph()
        return _json(result)

    @app.get("/api/papers/{doc_id}/versions")
    def versions(doc_id: str) -> JSONResponse:
        record = _require_doc(doc_id)
        family = deps.library().find_by_family(record.family_id) if record.family_id else [record]
        family = sorted(family, key=lambda p: (p.version or 0))
        return _json(
            {
                "family_id": record.family_id,
                "versions": [
                    {
                        "doc_id": p.doc_id,
                        "version": p.version,
                        "is_latest": bool(p.is_latest),
                        "title": p.title,
                        "status": p.status.value if hasattr(p.status, "value") else str(p.status),
                        "ingested_at": p.ingested_at,
                        "file_name": p.file_name,
                    }
                    for p in family
                ],
            }
        )

    @app.get("/api/papers/{doc_id}/diff")
    def version_diff(doc_id: str, against: str = Query(...)) -> JSONResponse:
        _require_doc(doc_id)
        _require_doc(against)
        old = pdf_parser.load_parsed(settings.data_dir, against)
        new = pdf_parser.load_parsed(settings.data_dir, doc_id)
        if not old or not new:
            raise HTTPException(409, error_payload("no_cache", "缺少解析缓存，无法比对"))
        return _json(diff_versions(old, new))

    # ---------------------------------------------------------------- 模块三
    @app.get("/api/papers/{doc_id}/page/{page}")
    def page_view(doc_id: str, page: int) -> JSONResponse:
        payload = service.page_view(doc_id, page)
        if payload is None:
            raise HTTPException(404, error_payload("not_found", doc_id))
        return _json(payload)

    @app.get("/api/papers/{doc_id}/file")
    def paper_file(doc_id: str):
        _require_doc(doc_id)
        path = service.source_path(doc_id)
        if not path.exists():
            raise HTTPException(404, error_payload("no_file", "原始的 PDF 不在本地"))
        return FileResponse(path, media_type="application/pdf", filename=f"{doc_id}.pdf")

    @app.get("/api/papers/{doc_id}/figures/{name}")
    def paper_figure(doc_id: str, name: str):
        _require_doc(doc_id)
        base = (settings.data_dir / doc_id / "figures").resolve()
        target = (base / Path(name).name).resolve()
        if not str(target).startswith(str(base)) or not target.exists():
            raise HTTPException(404, error_payload("not_found", name))
        return FileResponse(target)

    @app.get("/api/papers/{doc_id}/references")
    def references(doc_id: str) -> JSONResponse:
        payload = service.references_view(doc_id)
        if payload is None:
            raise HTTPException(404, error_payload("not_found", doc_id))
        return _json(payload)

    @app.get("/api/papers/{doc_id}/xref")
    def xref(doc_id: str, index: int = Query(...)) -> JSONResponse:
        payload = service.xref_view(doc_id, index)
        if payload is None:
            raise HTTPException(404, error_payload("not_found", doc_id))
        return _json(payload)

    @app.post("/api/translate/segment")
    def translate_segment_route(body: dict = Body(...)) -> JSONResponse:
        from ..writing.translate import translate_segment

        text = str(body.get("text") or "")
        to = str(body.get("to") or "zh")
        if not text.strip():
            raise HTTPException(400, error_payload("empty_text", "没有要翻译的内容"))
        result = translate_segment(text, to=to, gateway=deps.gateway())
        return _json({"translated": result.get("dst") or "", "terms": result.get("terms") or []})

    @app.get("/api/papers/{doc_id}/translate")
    def translate_page(doc_id: str, to: str = "zh", page: int = Query(1)) -> JSONResponse:
        payload = service.translate_page(doc_id, page, to)
        if payload is None:
            raise HTTPException(404, error_payload("not_found", doc_id))
        return _json(payload)

    # ---------------------------------------------------------------- 模块二
    @app.post("/api/qa/stream")
    def qa_stream(body: dict = Body(...)) -> StreamingResponse:
        question = str(body.get("question") or "").strip()
        doc_ids = [str(d) for d in (body.get("doc_ids") or []) if d]
        mode = str(body.get("mode") or "auto")
        top_k = body.get("top_k")
        top_k = int(top_k) if top_k else None

        def gen() -> Iterator[str]:
            if not question:
                yield _sse("error", {"kind": "empty_question", "message": "问题不能为空"})
                return
            try:
                for event, data in deps.engine().stream(
                    question, doc_ids=doc_ids or None, mode=mode, top_k=top_k
                ):
                    yield _sse(event, data)
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"kind": type(exc).__name__, "message": str(exc)})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/qa")
    def qa(body: dict = Body(...)) -> JSONResponse:
        question = str(body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, error_payload("empty_question", "问题不能为空"))
        answer = deps.engine().answer(
            question,
            doc_ids=[str(d) for d in (body.get("doc_ids") or []) if d] or None,
            mode=str(body.get("mode") or "auto"),
            top_k=int(body["top_k"]) if body.get("top_k") else None,
        )
        return _json(answer)

    @app.post("/api/extract")
    def extract(body: dict = Body(...)) -> JSONResponse:
        doc_id = str(body.get("doc_id") or "")
        _require_doc(doc_id)
        fields = [str(f) for f in (body.get("fields") or [])]
        if not fields:
            raise HTTPException(400, error_payload("no_fields", "至少给一个待抽取字段"))
        return _json(deps.engine().extract(fields, doc_id=doc_id, question_hint=str(body.get("hint") or "")))

    @app.post("/api/compare")
    def compare(body: dict = Body(...)) -> JSONResponse:
        doc_ids = [str(d) for d in (body.get("doc_ids") or []) if d]
        aspects = [str(a) for a in (body.get("aspects") or [])]
        if len(doc_ids) < 2 or not aspects:
            raise HTTPException(400, error_payload("bad_request", "跨文献对比至少需要 2 篇文献和 1 个对比维度"))
        return _json(deps.engine().compare(aspects, doc_ids=doc_ids))

    # ---------------------------------------------------------------- 模块五
    @app.post("/api/graph/build")
    def graph_build(body: dict = Body(default={})) -> JSONResponse:
        doc_ids = [str(d) for d in (body.get("doc_ids") or []) if d] or [
            p.doc_id for p in deps.library().list_papers()
        ]
        if not doc_ids:
            raise HTTPException(400, error_payload("empty_library", "库里还没有文献"))
        # 默认走离线启发式分类：秒级出图，且不依赖网络与密钥。
        # 需要让模型精判引用意图时，显式传 classify=true（会慢，几十次模型往返）。
        classify = bool(body.get("classify"))
        collected: list[dict] = []
        for doc_id in doc_ids:
            library = deps.library()
            edges = [e for e in library.load_citations([doc_id])]
            collected.append({"doc_id": doc_id, "edges": len(edges)})
        network, insights, metrics = service.build_graph(doc_ids, classify=classify)
        return _json(
            {
                "stats": network.to_dict().get("stats", {}),
                "metrics": metrics,
                "nodes": len(insights),
                "per_doc": collected,
                "classified_by": "llm" if classify else "offline",
            }
        )

    @app.get("/api/graph")
    def graph(doc_ids: str = Query("")) -> JSONResponse:
        ids = [d for d in doc_ids.split(",") if d] or [p.doc_id for p in deps.library().list_papers()]
        if not ids:
            return _json({"nodes": [], "edges": [], "metrics": {}})
        network, insights, metrics = service.get_graph(ids)
        data = network.to_dict()
        reasons = {i.doc_id: (i.role, i.reason) for i in insights}
        nodes = []
        for node in data.get("nodes", []):
            node_id = node.get("node_id") or node.get("doc_id") or ""
            role, reason = reasons.get(node_id, ("peripheral", ""))
            nodes.append({**node, "role": role, "reason": reason})
        return _json({"nodes": nodes, "edges": data.get("edges", []), "metrics": metrics})

    @app.get("/api/graph/citations")
    def graph_citations(doc_id: str = Query(...)) -> JSONResponse:
        _require_doc(doc_id)
        edges = deps.library().load_citations([doc_id])
        return _json({"doc_id": doc_id, "edges": [e.to_dict() for e in edges]})

    @app.get("/api/survey")
    def survey(doc_ids: str = Query(""), topic: str = Query("")) -> JSONResponse:
        from ..graph.survey import generate_survey

        ids = [d for d in doc_ids.split(",") if d] or [p.doc_id for p in deps.library().list_papers()]
        if not ids:
            raise HTTPException(400, error_payload("empty_library", "库里还没有文献"))
        _, insights, _metrics = service.get_graph(ids, classify=bool(deps.gateway().available))
        payload = generate_survey(
            topic or "文献综述",
            ids,
            library=deps.library(),
            insights=insights,
            data_dir=settings.data_dir,
            gateway=deps.gateway(),
            engine=deps.engine(),
            gate=deps.gate(),
        )
        return _json(payload)

    @app.get("/api/future-work")
    def future_work(doc_ids: str = Query(""), check: bool = Query(True)) -> JSONResponse:
        from ..graph.future_work import scan_future_work

        ids = [d for d in doc_ids.split(",") if d] or [p.doc_id for p in deps.library().list_papers()]
        if not ids:
            raise HTTPException(400, error_payload("empty_library", "库里还没有文献"))
        payload = scan_future_work(
            ids,
            library=deps.library(),
            data_dir=settings.data_dir,
            gateway=deps.gateway(),
            check_timeliness=bool(check and deps.gateway().available),
        )
        return _json(payload)

    # ---------------------------------------------------------------- 模块四
    @app.post("/api/discover/search")
    def discover_search(body: dict = Body(...)) -> JSONResponse:
        from ..discover.sources import search_all

        query = str(body.get("query") or "").strip()
        if not query:
            raise HTTPException(400, error_payload("empty_query", "检索词不能为空"))
        limit = int(body.get("limit") or 20)
        results = search_all(query, limit=limit, since_year=body.get("since_year"))
        return _json({"query": query, "results": [to_jsonable(c) for c in results]})

    @app.post("/api/discover/ingest")
    def discover_ingest(body: dict = Body(...)) -> JSONResponse:
        from ..discover.sources import ArxivClient, download_pdf

        identifier = str(body.get("identifier") or "").strip()
        if not identifier:
            raise HTTPException(400, error_payload("no_identifier", "缺少 arXiv 号"))
        client = ArxivClient()
        candidate = client.fetch_by_id(identifier)
        if candidate is None:
            raise HTTPException(404, error_payload("not_found", identifier))
        dest = settings.sub("downloads")
        path = download_pdf(candidate, dest)
        if path is None:
            raise HTTPException(502, error_payload("download_failed", "PDF 下载失败"))
        result = service.ingest_path(path, file_name=path.name)
        service.invalidate_graph()
        return _json({"doc_id": result["doc_id"], "status": result["status"], "dedup": result["dedup"]})

    @app.post("/api/discover/references")
    def discover_references(body: dict = Body(...)) -> JSONResponse:
        from ..discover.sources import search_semantic_scholar

        doc_id = str(body.get("doc_id") or "")
        _require_doc(doc_id)
        refs = (service.references_view(doc_id) or {}).get("references", [])
        candidates: list[dict] = []
        for ref in refs[:8]:
            title = ref.get("resolved_title") or ref.get("raw", "")[:80]
            if not title:
                continue
            try:
                hits = search_semantic_scholar(title, limit=2)
            except Exception:  # noqa: BLE001 - 外部源失败就跳过这条
                hits = []
            for hit in hits[:1]:
                candidates.append({"seed": ref["index"], **to_jsonable(hit)})
        return _json({"doc_id": doc_id, "seed_references": len(refs), "candidates": candidates})

    @app.get("/api/recommend")
    def recommend(doc_id: str = Query(""), days: int = Query(30)) -> JSONResponse:
        from ..discover.recommend import recommend as run_recommend

        query = ""
        if doc_id:
            record = _require_doc(doc_id)
            query = record.title or ""
        payload = run_recommend(
            deps.library(),
            query=query,
            days=days,
            gateway=deps.gateway(),
            use_github=bool(settings.github_token),
        )
        return _json(payload)

    @app.post("/api/feedback")
    def feedback(body: dict = Body(...)) -> JSONResponse:
        doc_id = str(body.get("doc_id") or "")
        action = str(body.get("action") or "")
        if action not in {"like", "dislike"}:
            raise HTTPException(400, error_payload("bad_action", action))
        deps.library().save_feedback(doc_id, action)
        return _json({"ok": True, "stats": deps.library().feedback_stats()})

    @app.get("/api/track/status")
    def track_status() -> JSONResponse:
        topics = [t for t in (deps.library().get_meta("track_topics", "") or "").split(",") if t]
        new_items = deps.library().get_meta("track_new_items", "0")
        return _json(
            {
                "last_run": deps.library().get_meta("track_last_run", ""),
                "tracked_topics": topics,
                "new_items": int(new_items or 0),
            }
        )

    @app.post("/api/track/run")
    def track_run(body: dict = Body(default={})) -> JSONResponse:
        from ..discover.recommend import run_tracking

        topics = [str(t) for t in (body.get("topics") or []) if str(t).strip()]
        days = int(body.get("since_days") or 7)
        payload = run_tracking(deps.library(), topics=topics or None, days=days, gateway=deps.gateway())
        library = deps.library()
        library.set_meta("track_topics", ",".join(payload.get("topics") or []))
        library.set_meta("track_new_items", str(payload.get("new_count") or 0))
        library.set_meta("track_last_run", str(payload.get("ran_at") or ""))
        return _json(payload)

    # ---------------------------------------------------------------- 模块六
    @app.post("/api/writing/outline")
    def writing_outline(body: dict = Body(...)) -> JSONResponse:
        from ..writing.outline import build_outline

        idea = str(body.get("idea") or "").strip()
        if not idea:
            raise HTTPException(400, error_payload("empty_idea", "先给一句话想法"))
        doc_ids = [str(d) for d in (body.get("doc_ids") or []) if d]
        return _json(
            build_outline(
                idea,
                doc_ids=doc_ids,
                library=deps.library(),
                gateway=deps.gateway(),
                gate=deps.gate(),
                engine=deps.engine(),
                venue=str(body.get("venue") or "generic"),
            )
        )

    @app.post("/api/writing/diagram")
    def writing_diagram(body: dict = Body(...)) -> JSONResponse:
        from ..writing.diagrams import build_diagram, diagram_from_network

        spec = body.get("spec")
        if not spec and body.get("from_graph"):
            ids = [str(d) for d in (body.get("doc_ids") or []) if d] or [
                p.doc_id for p in deps.library().list_papers()
            ]
            network, _insights, _metrics = service.get_graph(ids)
            spec = diagram_from_network(network)["spec"]
        return _json(build_diagram(str(body.get("kind") or "mermaid"), spec))

    @app.post("/api/writing/figure")
    def writing_figure(body: dict = Body(...)) -> JSONResponse:
        from ..writing.charts import make_chart

        csv_text = str(body.get("csv_text") or "").strip()
        if not csv_text:
            raise HTTPException(400, error_payload("empty_csv", "没有数据"))
        out_dir = settings.sub("figures")
        payload = make_chart(
            csv_text,
            chart_type=str(body.get("chart") or "line"),
            x=str(body.get("x") or ""),
            y=body.get("y") or None,
            title=str(body.get("title") or ""),
            out_dir=out_dir,
            gateway=deps.gateway(),
        )
        if payload.get("image_path"):
            deps.library().save_artifact(
                Path(payload["image_path"]).name, payload, kind="figure"
            )
        return _json(payload)

    @app.post("/api/writing/translate")
    def writing_translate(body: dict = Body(...)) -> JSONResponse:
        from ..writing.translate import translate_segment

        text = str(body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, error_payload("empty_text", "没有要翻译的内容"))
        result = translate_segment(text, to=str(body.get("to") or "en"), gateway=deps.gateway())
        return _json({"translated": result.get("dst") or "", "terms": result.get("terms") or []})

    @app.get("/api/writing/artifacts")
    def artifacts() -> JSONResponse:
        return _json({"artifacts": deps.library().list_artifacts()})

    @app.get("/api/writing/artifacts/{name}")
    def artifact_file(name: str):
        base = (settings.data_dir / "figures").resolve()
        target = (base / Path(name).name).resolve()
        if not str(target).startswith(str(base)) or not target.exists():
            raise HTTPException(404, error_payload("not_found", name))
        return FileResponse(target)

    # ---------------------------------------------------------------- 模块八
    @app.post("/api/agent/run")
    def agent_run(body: dict = Body(...)) -> StreamingResponse:
        goal = str(body.get("goal") or "").strip()
        doc_ids = [str(d) for d in (body.get("doc_ids") or []) if d]
        mode = str(body.get("mode") or "auto")

        def gen() -> Iterator[str]:
            try:
                for event, payload in stream_agent(
                    deps.agent(), goal, doc_ids=doc_ids, mode=mode
                ):
                    yield _sse(event, payload)
            except Exception as exc:  # noqa: BLE001
                yield _sse("error", {"kind": type(exc).__name__, "message": str(exc)})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/agent/tools")
    def agent_tools() -> JSONResponse:
        """Agent 的工具名册。规划器只能从这份名册里选，界面上也照实展示。"""
        registry = deps.registry()
        return _json({"tools": registry.catalog(), "count": len(registry.names())})

    @app.post("/api/agent/plan")
    def agent_plan(body: dict = Body(...)) -> JSONResponse:
        """只出计划、不执行 —— 让人先看清 Agent 打算怎么做。"""
        goal = str(body.get("goal") or "").strip()
        if not goal:
            raise HTTPException(400, error_payload("empty_goal", "先说清楚要调研什么"))
        doc_ids = [str(d) for d in (body.get("doc_ids") or []) if d]
        planned = planner.plan(
            goal,
            library=deps.library(),
            registry=deps.registry(),
            gateway=deps.gateway(),
            doc_ids=doc_ids,
        )
        return _json(planned.to_dict())

    # ---------------------------------------------------------------- 静态前端
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app

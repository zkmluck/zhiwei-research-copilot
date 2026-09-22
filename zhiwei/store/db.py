"""SQLite 档案库。

为什么用 SQLite 而不是直接丢 JSON 文件：
评审环境要跑「多文档并发分析」，并发写 JSON 一定会互相覆盖；
SQLite 的 WAL 模式能扛住一个 Web 服务级别的并发，而且零部署成本。

所有写操作走同一把可重入锁，读操作也给短事务 —— 简单、可预期、不会死锁。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from ..config import settings
from ..contracts import (
    Anchor,
    BBox,
    Chunk,
    CitationEdge,
    CitationIntent,
    DocStatus,
    PaperRecord,
)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    doc_id        TEXT PRIMARY KEY,
    title         TEXT DEFAULT '',
    authors       TEXT DEFAULT '[]',
    affiliations  TEXT DEFAULT '[]',
    abstract      TEXT DEFAULT '',
    year          INTEGER,
    venue         TEXT DEFAULT '',
    arxiv_id      TEXT DEFAULT '',
    version       INTEGER DEFAULT 1,
    doi           TEXT DEFAULT '',
    url           TEXT DEFAULT '',
    page_count    INTEGER DEFAULT 0,
    file_name     TEXT DEFAULT '',
    sha256        TEXT DEFAULT '',
    content_hash  TEXT DEFAULT '',
    family_id     TEXT DEFAULT '',
    is_latest     INTEGER DEFAULT 1,
    status        TEXT DEFAULT 'active',
    needs_ocr     INTEGER DEFAULT 0,
    ingested_at   TEXT DEFAULT '',
    extra         TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_papers_family ON papers(family_id);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    text          TEXT DEFAULT '',
    page          INTEGER DEFAULT 0,
    kind          TEXT DEFAULT 'text',
    section_path  TEXT DEFAULT '[]',
    char_start    INTEGER DEFAULT -1,
    char_end      INTEGER DEFAULT -1,
    bbox          TEXT DEFAULT '',
    tokens        INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS page_texts (
    doc_id  TEXT NOT NULL,
    page    INTEGER NOT NULL,
    text    TEXT DEFAULT '',
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE IF NOT EXISTS citations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    src_doc_id    TEXT NOT NULL,
    dst_ref       TEXT DEFAULT '',
    dst_doc_id    TEXT DEFAULT '',
    dst_title     TEXT DEFAULT '',
    dst_year      INTEGER,
    context       TEXT DEFAULT '',
    page          INTEGER DEFAULT 0,
    intent        TEXT DEFAULT 'unknown',
    value_score   REAL DEFAULT 0,
    substantive   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_citations_src ON citations(src_doc_id);

CREATE TABLE IF NOT EXISTS claims (
    claim_id   TEXT PRIMARY KEY,
    doc_id     TEXT DEFAULT '',
    payload    TEXT DEFAULT '{}',
    created_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(doc_id);

CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     TEXT NOT NULL,
    action     TEXT NOT NULL,
    created_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_feedback_doc ON feedback(doc_id);

CREATE TABLE IF NOT EXISTS artifacts (
    name       TEXT PRIMARY KEY,
    kind       TEXT DEFAULT '',
    payload    TEXT DEFAULT '{}',
    created_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT DEFAULT ''
);
"""


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S")


class Library:
    """文献档案库。所有上层模块都通过它读写元数据、切块与每页纯文本。"""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.path = Path(db_path or (settings.data_dir / "zhiwei.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ 基础
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cursor

    def _query(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)).fetchall())

    # ---------------------------------------------------------------- papers
    def upsert_paper(self, record: PaperRecord) -> None:
        data = record.to_dict()
        self._execute(
            """
            INSERT INTO papers (doc_id, title, authors, affiliations, abstract, year, venue,
                arxiv_id, version, doi, url, page_count, file_name, sha256, content_hash,
                family_id, is_latest, status, needs_ocr, ingested_at, extra)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title=excluded.title, authors=excluded.authors, affiliations=excluded.affiliations,
                abstract=excluded.abstract, year=excluded.year, venue=excluded.venue,
                arxiv_id=excluded.arxiv_id, version=excluded.version, doi=excluded.doi,
                url=excluded.url, page_count=excluded.page_count, file_name=excluded.file_name,
                sha256=excluded.sha256, content_hash=excluded.content_hash,
                family_id=excluded.family_id, is_latest=excluded.is_latest, status=excluded.status,
                needs_ocr=excluded.needs_ocr, ingested_at=excluded.ingested_at, extra=excluded.extra
            """,
            (
                record.doc_id,
                record.title,
                json.dumps(record.authors, ensure_ascii=False),
                json.dumps(record.affiliations, ensure_ascii=False),
                record.abstract,
                record.year,
                record.venue,
                record.arxiv_id,
                record.version,
                record.doi,
                record.url,
                record.page_count,
                record.file_name,
                record.sha256,
                record.content_hash,
                record.family_id,
                1 if record.is_latest else 0,
                record.status.value if isinstance(record.status, DocStatus) else str(record.status),
                1 if record.needs_ocr else 0,
                record.ingested_at or _now(),
                json.dumps(record.extra, ensure_ascii=False),
            ),
        )

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> PaperRecord:
        def _load(value: str, default: Any) -> Any:
            try:
                return json.loads(value) if value else default
            except json.JSONDecodeError:
                return default

        return PaperRecord(
            doc_id=row["doc_id"],
            title=row["title"] or "",
            authors=_load(row["authors"], []),
            affiliations=_load(row["affiliations"], []),
            abstract=row["abstract"] or "",
            year=row["year"],
            venue=row["venue"] or "",
            arxiv_id=row["arxiv_id"] or "",
            version=row["version"] or 1,
            doi=row["doi"] or "",
            url=row["url"] or "",
            page_count=row["page_count"] or 0,
            file_name=row["file_name"] or "",
            sha256=row["sha256"] or "",
            content_hash=row["content_hash"] or "",
            family_id=row["family_id"] or "",
            is_latest=bool(row["is_latest"]),
            status=DocStatus(row["status"] or "active"),
            needs_ocr=bool(row["needs_ocr"]),
            ingested_at=row["ingested_at"] or "",
            extra=_load(row["extra"], {}),
        )

    def get_paper(self, doc_id: str) -> Optional[PaperRecord]:
        rows = self._query("SELECT * FROM papers WHERE doc_id = ?", (doc_id,))
        return self._row_to_paper(rows[0]) if rows else None

    def list_papers(self, *, include_superseded: bool = False) -> list[PaperRecord]:
        sql = "SELECT * FROM papers"
        if not include_superseded:
            sql += " WHERE status != 'duplicate'"
        sql += " ORDER BY ingested_at DESC"
        return [self._row_to_paper(r) for r in self._query(sql)]

    def find_by_family(self, family_id: str) -> list[PaperRecord]:
        if not family_id:
            return []
        rows = self._query(
            "SELECT * FROM papers WHERE family_id = ? ORDER BY version ASC", (family_id,)
        )
        return [self._row_to_paper(r) for r in rows]

    def find_by_arxiv(self, arxiv_id: str, version: Optional[int] = None) -> list[PaperRecord]:
        if not arxiv_id:
            return []
        if version is None:
            rows = self._query(
                "SELECT * FROM papers WHERE arxiv_id = ? ORDER BY version ASC", (arxiv_id,)
            )
        else:
            rows = self._query(
                "SELECT * FROM papers WHERE arxiv_id = ? AND version = ?", (arxiv_id, version)
            )
        return [self._row_to_paper(r) for r in rows]

    def set_status(self, doc_id: str, status: DocStatus | str) -> None:
        value = status.value if isinstance(status, DocStatus) else str(status)
        self._execute("UPDATE papers SET status = ? WHERE doc_id = ?", (value, doc_id))

    def set_latest(self, doc_id: str, is_latest: bool) -> None:
        self._execute("UPDATE papers SET is_latest = ? WHERE doc_id = ?", (1 if is_latest else 0, doc_id))

    def delete_paper(self, doc_id: str) -> None:
        self._execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._execute("DELETE FROM page_texts WHERE doc_id = ?", (doc_id,))
        self._execute("DELETE FROM citations WHERE src_doc_id = ?", (doc_id,))
        self._execute("DELETE FROM claims WHERE doc_id = ?", (doc_id,))
        self._execute("DELETE FROM papers WHERE doc_id = ?", (doc_id,))

    # ---------------------------------------------------------------- chunks
    def save_chunks(self, doc_id: str, chunks: list[Chunk]) -> None:
        self.delete_chunks(doc_id)
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, page, kind, section_path,
                    char_start, char_end, bbox, tokens)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        c.chunk_id,
                        doc_id,
                        c.text,
                        c.page,
                        c.kind,
                        json.dumps(c.section_path, ensure_ascii=False),
                        c.char_start,
                        c.char_end,
                        json.dumps(c.bbox.to_dict()) if c.bbox else "",
                        c.tokens,
                    )
                    for c in chunks
                ],
            )
            self._conn.commit()

    def delete_chunks(self, doc_id: str) -> None:
        self._execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def load_chunks(self, doc_id: str) -> list[Chunk]:
        out: list[Chunk] = []
        for row in self._query("SELECT * FROM chunks WHERE doc_id = ?", (doc_id,)):
            bbox = None
            if row["bbox"]:
                try:
                    raw = json.loads(row["bbox"])
                    bbox = BBox(**raw)
                except (json.JSONDecodeError, TypeError):
                    bbox = None
            out.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"] or "",
                    page=row["page"] or 0,
                    kind=row["kind"] or "text",
                    section_path=json.loads(row["section_path"] or "[]"),
                    char_start=row["char_start"],
                    char_end=row["char_end"],
                    bbox=bbox,
                    tokens=row["tokens"] or 0,
                )
            )
        return out

    # ------------------------------------------------------------ page_texts
    def save_page_texts(self, doc_id: str, pages: dict[int, str]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO page_texts (doc_id, page, text) VALUES (?,?,?)",
                [(doc_id, int(p), t or "") for p, t in (pages or {}).items()],
            )
            self._conn.commit()

    def load_page_texts(self, doc_id: str) -> dict[int, str]:
        rows = self._query("SELECT page, text FROM page_texts WHERE doc_id = ?", (doc_id,))
        return {int(r["page"]): (r["text"] or "") for r in rows}

    # ------------------------------------------------------------- citations
    def save_citations(self, edges: list[CitationEdge]) -> None:
        if not edges:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO citations (src_doc_id, dst_ref, dst_doc_id, dst_title, dst_year,
                    context, page, intent, value_score, substantive)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        e.src_doc_id,
                        e.dst_ref,
                        e.dst_doc_id,
                        e.dst_title,
                        e.dst_year,
                        e.context,
                        e.page,
                        e.intent.value if isinstance(e.intent, CitationIntent) else str(e.intent),
                        e.value_score,
                        1 if e.substantive else 0,
                    )
                    for e in edges
                ],
            )
            self._conn.commit()

    def clear_citations(self, doc_ids: Iterable[str]) -> None:
        with self._lock:
            self._conn.executemany(
                "DELETE FROM citations WHERE src_doc_id = ?", [(d,) for d in doc_ids]
            )
            self._conn.commit()

    def load_citations(self, doc_ids: Iterable[str]) -> list[CitationEdge]:
        ids = list(doc_ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        out: list[CitationEdge] = []
        for row in self._query(f"SELECT * FROM citations WHERE src_doc_id IN ({marks})", ids):
            out.append(
                CitationEdge(
                    src_doc_id=row["src_doc_id"],
                    dst_ref=row["dst_ref"] or "",
                    dst_doc_id=row["dst_doc_id"] or "",
                    dst_title=row["dst_title"] or "",
                    dst_year=row["dst_year"],
                    context=row["context"] or "",
                    page=row["page"] or 0,
                    intent=CitationIntent(row["intent"] or "unknown"),
                    value_score=row["value_score"] or 0.0,
                    substantive=bool(row["substantive"]),
                )
            )
        return out

    # ---------------------------------------------------------------- claims
    def save_claims(self, doc_id: str, claims: list[Any]) -> None:
        if not claims:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO claims (claim_id, doc_id, payload, created_at) VALUES (?,?,?,?)",
                [
                    (
                        getattr(c, "claim_id", "") or f"{doc_id}-{i}",
                        doc_id,
                        json.dumps(c.to_dict() if hasattr(c, "to_dict") else c, ensure_ascii=False),
                        _now(),
                    )
                    for i, c in enumerate(claims)
                ],
            )
            self._conn.commit()

    def load_claims(self, doc_id: str) -> list[dict]:
        rows = self._query(
            "SELECT payload FROM claims WHERE doc_id = ? ORDER BY created_at DESC LIMIT 200", (doc_id,)
        )
        out = []
        for row in rows:
            try:
                out.append(json.loads(row["payload"]))
            except json.JSONDecodeError:
                continue
        return out

    def create_claim_table_alias(self) -> None:  # pragma: no cover - 兼容占位
        return None

    # -------------------------------------------------------------- feedback
    def save_feedback(self, doc_id: str, action: str) -> None:
        self._execute(
            "INSERT INTO feedback (doc_id, action, created_at) VALUES (?,?,?)",
            (doc_id, action, _now()),
        )

    def feedback_stats(self) -> dict:
        rows = self._query("SELECT doc_id, action FROM feedback")
        liked, disliked = [], []
        for row in rows:
            (liked if row["action"] == "like" else disliked).append(row["doc_id"])
        return {
            "total": len(rows),
            "liked_ids": sorted(set(liked)),
            "disliked_ids": sorted(set(disliked)),
            "likes": len(liked),
            "dislikes": len(disliked),
        }

    # ------------------------------------------------------------- artifacts
    def save_artifact(self, name: str, payload: dict, kind: str = "") -> None:
        self._execute(
            "INSERT OR REPLACE INTO artifacts (name, kind, payload, created_at) VALUES (?,?,?,?)",
            (name, kind, json.dumps(payload, ensure_ascii=False), _now()),
        )

    def list_artifacts(self) -> list[dict]:
        out = []
        for row in self._query("SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 200"):
            try:
                payload = json.loads(row["payload"])
            except json.JSONDecodeError:
                payload = {}
            out.append({"name": row["name"], "kind": row["kind"], "created_at": row["created_at"], **payload})
        return out

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: str = "") -> str:
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else default

    def set_meta(self, key: str, value: str) -> None:
        self._execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, str(value)))
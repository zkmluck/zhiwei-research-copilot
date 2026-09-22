"""存储层与混合检索的离线测试。全部不依赖网络与 API key。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zhiwei.contracts import (
    Anchor,
    BBox,
    Chunk,
    CitationEdge,
    CitationIntent,
    DocStatus,
    PaperRecord,
)
from zhiwei.store import Library, LocalHashEmbedder, locate_quote, tokenize
from zhiwei.store.index import BM25, HybridIndex

PAGE_ONE = (
    "We trained the base model on 8 NVIDIA P100 GPUs for 12 hours. "
    "The Transformer uses multi-head attention with h = 8 heads and d_model = 512."
)
PAGE_TWO = "On the WMT 2014 English-to-German task our big model achieves a BLEU score of 28.4."


def make_paper(doc_id: str = "p1", version: int = 1, family: str = "fam1") -> PaperRecord:
    return PaperRecord(
        doc_id=doc_id,
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        arxiv_id="1706.03762",
        version=version,
        family_id=family,
        page_count=15,
        file_name=f"1706.03762v{version}.pdf",
        content_hash="deadbeef",
    )


class TestLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = Library(Path(self.tmp.name) / "lib.sqlite3")

    def tearDown(self):
        self.lib.close()
        self.tmp.cleanup()

    def test_schema_is_idempotent(self):
        # 重复打开同一路径不应该报错，也不应该丢数据
        again = Library(self.lib.path)
        again.upsert_paper(make_paper())
        self.assertEqual(len(again.list_papers()), 1)
        again.close()

    def test_paper_crud_and_versions(self):
        self.lib.upsert_paper(make_paper("v7", version=7))
        self.lib.upsert_paper(make_paper("v1", version=1))
        self.assertEqual(len(self.lib.list_papers()), 2)
        family = self.lib.find_by_family("fam1")
        self.assertEqual([p.version for p in family], [1, 7])
        self.assertEqual(len(self.lib.find_by_arxiv("1706.03762", 7)), 1)
        # 被取代的版本标成 duplicate 后，默认列表里不再出现
        self.lib.set_status("v1", DocStatus.DUPLICATE)
        self.assertEqual(len(self.lib.list_papers()), 1)
        self.assertEqual(len(self.lib.list_papers(include_superseded=True)), 2)

    def test_chunks_and_page_texts_roundtrip(self):
        self.lib.upsert_paper(make_paper())
        chunks = [
            Chunk("c1", "p1", "hello world", 1, "text", ["Intro"], 0, 11, BBox(0, 0, 10, 10), 2),
            Chunk("c2", "p1", "second chunk", 2, "text", ["Method"], 12, 24),
        ]
        self.lib.save_chunks("p1", chunks)
        loaded = self.lib.load_chunks("p1")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].bbox.x1, 10)
        self.lib.save_page_texts("p1", {1: PAGE_ONE, 2: PAGE_TWO})
        pages = self.lib.load_page_texts("p1")
        self.assertEqual(set(pages), {1, 2})
        self.assertIn("P100", pages[1])

    def test_delete_paper_cascades(self):
        self.lib.upsert_paper(make_paper())
        self.lib.save_chunks("p1", [Chunk("c1", "p1", "x", 1)])
        self.lib.save_page_texts("p1", {1: "x"})
        self.lib.delete_paper("p1")
        self.assertIsNone(self.lib.get_paper("p1"))
        self.assertEqual(self.lib.load_chunks("p1"), [])
        self.assertEqual(self.lib.load_page_texts("p1"), {})

    def test_citations_and_feedback(self):
        self.lib.save_citations(
            [
                CitationEdge("p1", "[1] Some ref", dst_doc_id="p2", context="we build on [1]",
                             intent=CitationIntent.EXTENDS, value_score=0.9, substantive=True),
                CitationEdge("p1", "[2] Another", context="many works [1,2,3]",
                             intent=CitationIntent.LISTS, value_score=0.1, substantive=False),
            ]
        )
        edges = self.lib.load_citations(["p1"])
        self.assertEqual(len(edges), 2)
        self.assertIs(edges[0].intent, CitationIntent.EXTENDS)
        self.lib.clear_citations(["p1"])
        self.assertEqual(self.lib.load_citations(["p1"]), [])

        self.lib.save_feedback("p1", "like")
        self.lib.save_feedback("p2", "dislike")
        stats = self.lib.feedback_stats()
        self.assertEqual(stats["likes"], 1)
        self.assertIn("p1", stats["liked_ids"])
        self.assertIn("p2", stats["disliked_ids"])

    def test_artifacts(self):
        self.lib.save_artifact("figure-1", {"image_path": "a.png"}, kind="figure")
        items = self.lib.list_artifacts()
        self.assertEqual(items[0]["name"], "figure-1")
        self.assertEqual(items[0]["image_path"], "a.png")


class TestTokenize(unittest.TestCase):
    def test_mixed_language(self):
        tokens = tokenize("多注意力头 Multi-Head Attention 有8个头")
        self.assertIn("multi-head", tokens)
        self.assertIn("注意", tokens)  # 中文按字符 bigram 切词

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestBM25(unittest.TestCase):
    def test_ranks_relevant_higher(self):
        corpus = [tokenize(t) for t in [
            "the transformer trains on 8 gpus",
            "protein folding prediction model",
            "gpu training time for transformers",
        ]]
        bm25 = BM25(corpus)
        scores = bm25.scores(tokenize("transformer gpu training"))
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[0], scores[1])


class TestLocalHashEmbedder(unittest.TestCase):
    def test_deterministic_and_normalised(self):
        emb = LocalHashEmbedder(dim=128)
        a = emb.encode(["attention is all you need"])[0]
        b = emb.encode(["attention is all you need"])[0]
        self.assertEqual(a, b)
        self.assertEqual(emb.dim, 128)
        norm = sum(x * x for x in a) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similar_text_scores_higher(self):
        emb = LocalHashEmbedder(dim=256)
        base, close, far = emb.encode([
            "multi head attention in the transformer encoder",
            "the transformer encoder uses multi head attention",
            "protein structure prediction with folding networks",
        ])
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))
        self.assertGreater(dot(base, close), dot(base, far))


class TestLocateQuote(unittest.TestCase):
    def setUp(self):
        self.pages = {1: PAGE_ONE, 2: PAGE_TWO}

    def test_exact_match(self):
        anchor = locate_quote(self.pages, "8 NVIDIA P100 GPUs for 12 hours")
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.page, 1)
        self.assertGreaterEqual(anchor.char_start, 0)

    def test_whitespace_and_newline_differences(self):
        anchor = locate_quote(self.pages, "8 NVIDIA P100 GPUs\n   for 12 hours")
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.page, 1)

    def test_case_insensitive(self):
        anchor = locate_quote(self.pages, "we trained the BASE model on 8 nvidia p100 gpus")
        self.assertIsNotNone(anchor)

    def test_fabricated_quote_returns_none(self):
        # 最关键的一条：编造的引文必须返回 None，否则闸门就失去意义
        anchor = locate_quote(
            self.pages, "we trained the model on 64 TPU v4 chips for three weeks"
        )
        self.assertIsNone(anchor)

    def test_prefix_match_when_model_omits_middle(self):
        anchor = locate_quote(self.pages, "The Transformer uses multi-head attention with h = 8")
        self.assertIsNotNone(anchor)
        self.assertLessEqual(anchor.char_end - anchor.char_start, 64)

    def test_maps_to_bbox_when_blocks_given(self):
        blocks = {1: [{"text": PAGE_ONE, "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 130}}]}
        anchor = locate_quote(self.pages, "d_model = 512", blocks)
        self.assertIsNotNone(anchor)
        self.assertIsNotNone(anchor.bbox)
        self.assertEqual(anchor.bbox.x1, 540)


class TestHybridIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = Library(Path(self.tmp.name) / "lib.sqlite3")
        self.lib.upsert_paper(make_paper())
        self.lib.save_page_texts("p1", {1: PAGE_ONE, 2: PAGE_TWO})
        parsed = {
            "record": {"doc_id": "p1", "title": "Attention Is All You Need"},
            "chunks": [
                {"chunk_id": "p1:0", "text": PAGE_ONE, "page": 1, "kind": "text",
                 "section_path": ["3 Model Architecture"]},
                {"chunk_id": "p1:1", "text": PAGE_TWO, "page": 2, "kind": "text",
                 "section_path": ["6 Results"]},
                {"chunk_id": "p0:0", "text": "Protein folding is a different topic entirely.",
                 "page": 1, "kind": "text", "section_path": []},
            ],
        }
        self.index = HybridIndex(
            self.lib, data_dir=Path(self.tmp.name) / "data", use_chroma=False,
            embedder=LocalHashEmbedder(),
        )
        self.count = self.index.index_document(parsed)

    def tearDown(self):
        self.lib.close()
        self.tmp.cleanup()

    def test_index_counts_chunks(self):
        self.assertEqual(self.count, 3)

    def test_reindex_is_idempotent(self):
        again = self.index.index_document(
            {
                "record": {"doc_id": "p1"},
                "chunks": [{"chunk_id": "p1:0", "text": PAGE_ONE, "page": 1, "kind": "text"}],
            }
        )
        self.assertEqual(again, 1)
        # 重新索引同一篇文档应替换而不是累加（幂等）\n        self.assertEqual(self.index.stats()["chunks"], 1)\n        self.assertEqual(self.index.stats()["documents"], 1)

    def test_search_returns_evidence_with_anchor(self):
        hits = self.index.search("how many GPUs were used for training", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "p1")
        self.assertGreater(hits[0].anchor.page, 0)
        self.assertTrue(hits[0].chunk_id)

    def test_doc_id_filter(self):
        hits = self.index.search("protein folding", top_k=3, doc_ids=["p1"])
        self.assertTrue(all(h.doc_id == "p1" for h in hits))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.index.search(""), [])

    def test_delete_document(self):
        self.index.delete_document("p1")
        hits = self.index.search("GPUs", top_k=3, doc_ids=["p1"])
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
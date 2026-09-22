"""解析层与去重层的离线测试。

全部用手工构造的数据：不读 PDF、不联网、不依赖模型密钥。
理由很直接 —— 这几条逻辑是「抗幻觉」的地基，地基必须能在任何环境下被验证。
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhiwei.contracts import (  # noqa: E402
    BBox,
    Block,
    DedupRelation,
    DocStatus,
    ParsedDocument,
    PaperRecord,
)
from zhiwei.ingest import dedup as dedup_mod  # noqa: E402
from zhiwei.ingest import pdf_parser  # noqa: E402


def raw(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1, size: float = 10.0):
    return pdf_parser._RawBlock(page=page, text=text, bbox=BBox(x0, y0, x1, y1), size=size)


def para(text: str, page: int = 1, path=None) -> Block:
    return Block(kind="paragraph", page=page, text=text, section_path=list(path or []))


def heading(text: str, page: int = 1, path=None) -> Block:
    return Block(kind="heading", page=page, text=text, section_path=list(path or []))


def parsed_of(doc_id: str, body: str, *, title: str = "", arxiv: str = "", version: int = 1):
    record = PaperRecord(doc_id=doc_id, title=title, arxiv_id=arxiv, version=version,
                         status=DocStatus.ACTIVE)
    blocks = [para(line) for line in body.split("\n") if line.strip()]
    document = ParsedDocument(record=record, blocks=blocks)
    record.content_hash = dedup_mod.content_fingerprint(document)
    return document


class TestColumnDetection(unittest.TestCase):
    def test_two_column_page(self):
        blocks = [
            raw("L1", 60, 100, 290, 130),
            raw("L2", 60, 140, 290, 170),
            raw("L3", 60, 180, 290, 210),
            raw("R1", 320, 100, 550, 130),
            raw("R2", 320, 140, 550, 170),
            raw("R3", 320, 180, 550, 210),
        ]
        self.assertEqual(pdf_parser.detect_columns(blocks, 612.0), 2)

    def test_single_column_page(self):
        blocks = [raw(f"P{i}", 60, 100 + i * 40, 550, 130 + i * 40) for i in range(5)]
        self.assertEqual(pdf_parser.detect_columns(blocks, 612.0), 1)

    def test_reading_order_keeps_columns_separate(self):
        left = [raw("L1", 60, 100, 290, 130), raw("L2", 60, 200, 290, 230)]
        right = [raw("R1", 320, 100, 550, 130), raw("R2", 320, 200, 550, 230)]
        ordered = pdf_parser.reading_order(left + right, 612.0, 2)
        texts = [b.text for b in ordered]
        self.assertEqual(sorted(texts), ["L1", "L2", "R1", "R2"])
        # 左栏四块应当整体排在右栏之前，而不是按 y 坐标交错
        self.assertLess(texts.index("L2"), texts.index("R1"))

    def test_reading_order_single_column_is_top_down(self):
        blocks = [raw("B", 60, 300, 550, 330), raw("A", 60, 100, 550, 130)]
        ordered = pdf_parser.reading_order(blocks, 612.0, 1)
        self.assertEqual([b.text for b in ordered], ["A", "B"])


class TestReferenceSplitting(unittest.TestCase):
    def test_numbered_references(self):
        blocks = [
            Block(kind="reference", page=9, text="[1] Author One. Title one. arXiv:1706.03762, 2017."),
            Block(kind="reference", page=9, text="[2] Author Two. Title two. NeurIPS, 2018."),
        ]
        entries = pdf_parser.split_references(blocks)
        self.assertEqual(len(entries), 2)
        self.assertIn("Title one", entries[0])
        self.assertIn("Title two", entries[1])

    def test_unnumbered_falls_back_to_blocks(self):
        first = "Ba, J. Layer normalization and its applications in deep sequence models."
        second = "Chollet, F. Xception: deep learning with depthwise separable convolutions."
        entries = pdf_parser.split_references([
            Block(kind="paragraph", page=9, text=first),
            Block(kind="paragraph", page=9, text=second),
        ])
        self.assertEqual(len(entries), 2)


class TestChunking(unittest.TestCase):
    def test_chunks_do_not_cross_sections(self):
        blocks = [
            heading("1 Introduction", path=["1 Introduction"]),
            para("Intro paragraph one about motivation and background.", path=["1 Introduction"]),
            para("Intro paragraph two about the problem statement.", path=["1 Introduction"]),
            heading("2 Method", path=["2 Method"]),
            para("Method paragraph about the proposed architecture.", path=["2 Method"]),
        ]
        chunks = pdf_parser.build_chunks(blocks, "doc-1", {1: " ".join(b.text for b in blocks)})
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertFalse(
                "Intro" in chunk.text and "Method" in chunk.text,
                msg=f"切块跨越了章节边界：{chunk.text!r}",
            )
        paths = {tuple(c.section_path) for c in chunks}
        self.assertGreaterEqual(len(paths), 2)

    def test_chunks_do_not_cross_pages(self):
        blocks = [
            para("Page one paragraph with enough length to be a chunk on its own.", page=1),
            para("Page two paragraph with enough length to be a chunk on its own.", page=2),
        ]
        chunks = pdf_parser.build_chunks(blocks, "doc-1", {1: "", 2: ""})
        pages = [c.page for c in chunks]
        self.assertEqual(pages, sorted(set(pages)))

    def test_heading_only_section_is_dropped(self):
        blocks = [heading("5 Training"), heading("5.1 Optimizer")]
        chunks = pdf_parser.build_chunks(blocks, "doc-1", {1: ""})
        self.assertEqual(chunks, [])


class TestDedup(unittest.TestCase):
    def test_empty_library_is_distinct(self):
        incoming = parsed_of("d1", "Attention is all you need.")
        verdict = dedup_mod.check_duplicate(incoming.record, incoming, existing=[], use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.DISTINCT)
        self.assertIn("empty_library", verdict.detected_by)

    def test_same_version(self):
        v7 = parsed_of("d7", "the same body text about transformers", arxiv="1706.03762", version=7)
        incoming = parsed_of("dup", "the same body text about transformers", arxiv="1706.03762", version=7)
        verdict = dedup_mod.check_duplicate(incoming.record, incoming, existing=[v7.record],
                                             use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.SAME_VERSION)
        self.assertEqual(verdict.recommendation, "keep_existing")
        self.assertIn("arxiv_base", verdict.detected_by)

    def test_newer_version_suggests_replace(self):
        v1 = parsed_of("d1", "early preprint body", arxiv="1706.03762", version=1)
        v7 = parsed_of("d7", "final camera ready body", arxiv="1706.03762", version=7)
        verdict = dedup_mod.check_duplicate(v7.record, v7, existing=[v1.record], use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.NEWER_VERSION)
        self.assertEqual(verdict.recommendation, "replace")
        self.assertEqual(verdict.existing_version, 1)

    def test_older_version_suggests_keep_existing(self):
        v1 = parsed_of("d1", "early preprint body", arxiv="1706.03762", version=1)
        v7 = parsed_of("d7", "final camera ready body", arxiv="1706.03762", version=7)
        verdict = dedup_mod.check_duplicate(v1.record, v1, existing=[v7.record], use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.OLDER_VERSION)
        self.assertEqual(verdict.recommendation, "keep_existing")
        self.assertIn("较早的", verdict.message)

    def test_near_duplicate_by_content(self):
        body = "identical body content used to trigger the content fingerprint match\n" * 3
        first = parsed_of("a", body, title="A Study of Something")
        second = parsed_of("b", body, title="A Study of Something Else")
        verdict = dedup_mod.check_duplicate(second.record, second, existing=[first.record],
                                            use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.NEAR_DUPLICATE)
        self.assertIn("content_hash", verdict.detected_by)

    def test_unrelated_papers_are_distinct(self):
        first = parsed_of("a", "quantum chemistry simulation of molecular orbitals",
                          title="Quantum Chemistry Notes")
        second = parsed_of("b", "retrieval augmented generation for open domain question answering",
                           title="Retrieval Augmented Generation")
        verdict = dedup_mod.check_duplicate(second.record, second, existing=[first.record],
                                            use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.DISTINCT)
        self.assertEqual(verdict.similarity, 0.0)

    def test_same_doc_id_is_never_reported_as_duplicate_of_itself(self):
        one = parsed_of("same", "body text that would otherwise match", arxiv="1234.5678", version=1)
        verdict = dedup_mod.check_duplicate(one.record, one, existing=[one.record], use_embedding=False)
        self.assertIs(verdict.relation, DedupRelation.DISTINCT)


if __name__ == "__main__":
    unittest.main()

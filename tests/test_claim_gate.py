"""主张闸门的离线测试。

这些用例的意义：**证明闸门真的会拦**。一个只会放行的闸门等于没有闸门，
所以每个"应该被拦下"的场景都必须有对应用例。
全部离线，不依赖网络与 API key。
"""

from __future__ import annotations

import unittest

from zhiwei.contracts import Anchor, Claim, Evidence, Verdict
from zhiwei.reasoning.claim_gate import ClaimGate, GateConfig, LocalJudge
from zhiwei.reasoning.evidence import (
    GateContext,
    attach_anchors,
    lexical_overlap,
    numbers_in,
    numeric_support,
    split_sentences,
)

PAGE_TEXT = (
    "We trained the base model on 8 NVIDIA P100 GPUs for 12 hours. "
    "The big model was trained for 3.5 days on the same hardware. "
    "On the WMT 2014 English-to-German translation task, our big model achieves "
    "a BLEU score of 28.4, outperforming the previous best reported models. "
    "The Transformer uses multi-head attention with h = 8 heads and d_model = 512."
)

BLOCKS_BY_PAGE = {
    1: [
        {
            "index": 0,
            "text": PAGE_TEXT[:120],
            "bbox": (72.0, 300.0, 540.0, 320.0),
        }
    ]
}


def make_gate() -> ClaimGate:
    # 强制用本地裁判，保证离线可复现
    return ClaimGate(judge=LocalJudge(), config=GateConfig())


def make_context() -> GateContext:
    return GateContext(
        page_texts={"p1": {1: PAGE_TEXT}},
        blocks_by_page={"p1": BLOCKS_BY_PAGE},
        doc_titles={"p1": "Attention Is All You Need"},
    )


def evidence_with_quote(quote: str, page: int = 1, located: bool = False) -> Evidence:
    anchor = (
        Anchor(page=page, quote=quote, char_start=10, char_end=10 + len(quote))
        if located
        else Anchor(page=page, quote=quote)
    )
    return Evidence(doc_id="p1", anchor=anchor, score=1.0, doc_title="Attention Is All You Need")


class TestLexicalHelpers(unittest.TestCase):
    def test_numbers_in_extracts_metric_and_value(self):
        found = numbers_in("BLEU 27.3 on 8 P100 GPUs, lr=1e-4")
        self.assertIn("27.3", found)
        self.assertIn("8", found)
        self.assertIn("1e-4", found)

    def test_numbers_in_handles_fullwidth_digits(self):
        # 全角数字不能让模型绕过核验
        self.assertEqual(numbers_in("ＢＬＥＵ ２８.４"), numbers_in("BLEU 28.4"))

    def test_numeric_support_detects_missing(self):
        ok, missing = numeric_support("BLEU 达到 41.8", [PAGE_TEXT])
        self.assertFalse(ok)
        self.assertIn("41.8", missing)

    def test_numeric_support_passes_on_exact_match(self):
        ok, missing = numeric_support("BLEU 达到 28.4", [PAGE_TEXT])
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_lexical_overlap_orders_correctly(self):
        high = lexical_overlap("在 WMT 2014 英德翻译任务上达到 28.4 BLEU", [PAGE_TEXT])
        low = lexical_overlap("该方法在蛋白质折叠预测上取得突破", [PAGE_TEXT])
        self.assertGreater(high, low)

    def test_split_sentences_bilingual(self):
        parts = split_sentences("第一句。第二句！Third one. Fourth one.")
        self.assertGreaterEqual(len(parts), 4)


class TestClaimGateBlocks(unittest.TestCase):
    """闸门必须拦下的四类东西。"""

    def setUp(self):
        self.gate = make_gate()
        self.ctx = make_context()

    def test_block_when_no_evidence(self):
        decision = self.gate.vet(Claim(text="Transformer 使用 8 个注意力头。"), self.ctx)
        self.assertIs(decision.verdict, Verdict.BLOCK)
        self.assertIn("没有附带任何原文出处", decision.reasons[0])

    def test_block_when_quote_is_fabricated(self):
        # 引文看起来很像真的，但原文里根本没有 —— 必须拦下
        claim = Claim(
            text="论文报告在 EN-DE 上达到 41.8 BLEU。",
            evidence=[evidence_with_quote("our model achieves a BLEU score of 41.8 on WMT 2014")],
        )
        decision = self.gate.vet(claim, self.ctx)
        self.assertIs(decision.verdict, Verdict.BLOCK)
        self.assertIn("定位不到", decision.reasons[0])

    def test_block_when_number_not_in_source(self):
        # 引文真实存在，但主张里的数字被改了 —— 这是最典型的幻觉
        claim = Claim(
            text="模型使用了 64 张 V100 显卡训练。",
            evidence=[evidence_with_quote("We trained the base model on 8 NVIDIA P100 GPUs")],
        )
        decision = self.gate.vet(claim, self.ctx)
        self.assertIs(decision.verdict, Verdict.BLOCK)
        self.assertFalse(decision.checks["numbers_ok"])
        self.assertIn("64", decision.checks["missing_numbers"])

    def test_block_when_evidence_is_off_topic(self):
        claim = Claim(
            text="该文提出了蛋白质结构预测的新范式。",
            evidence=[evidence_with_quote("We trained the base model on 8 NVIDIA P100 GPUs")],
        )
        decision = self.gate.vet(claim, self.ctx)
        self.assertIsNot(decision.verdict, Verdict.ALLOW)


class TestClaimGateAllows(unittest.TestCase):
    """有出处、数字一致的主张必须能过 —— 闸门不能只会拦。"""

    def setUp(self):
        self.gate = make_gate()
        self.ctx = make_context()

    def test_allow_fully_supported_claim(self):
        claim = Claim(
            text="基础模型在 8 张 NVIDIA P100 上训练了 12 小时。",
            evidence=[evidence_with_quote("We trained the base model on 8 NVIDIA P100 GPUs for 12 hours.")],
        )
        decision = self.gate.vet(claim, self.ctx)
        self.assertIs(decision.verdict, Verdict.ALLOW)
        self.assertGreaterEqual(decision.confidence, 0.75)
        self.assertEqual(decision.checks["judge"], "local")

    def test_anchor_is_located_automatically(self):
        claim = Claim(
            text="模型使用 8 个注意力头，d_model 为 512。",
            evidence=[evidence_with_quote("The Transformer uses multi-head attention with h = 8 heads and d_model = 512")],
        )
        decision = self.gate.vet(claim, self.ctx)
        self.assertIs(decision.verdict, Verdict.ALLOW)
        self.assertTrue(claim.evidence[0].anchor.is_located())
        self.assertGreaterEqual(claim.evidence[0].anchor.char_start, 0)

    def test_collect_filters_blocked_claims(self):
        good = Claim(
            text="基础模型用 8 张 P100 训练 12 小时。",
            evidence=[evidence_with_quote("We trained the base model on 8 NVIDIA P100 GPUs for 12 hours.")],
        )
        bad = Claim(text="该文提出了蛋白质结构预测范式。")
        self.gate.vet_all([good, bad], self.ctx)
        kept = ClaimGate.collect([good, bad])
        self.assertEqual([c.text for c in kept], [good.text])
        summary = ClaimGate.summarize([good, bad])
        self.assertEqual(summary["by_verdict"]["block"], 1)


class TestAttachAnchors(unittest.TestCase):
    def test_attach_anchors_marks_unlocatable_quote(self):
        ctx = make_context()
        claims = attach_anchors(
            [
                {
                    "text": "基础模型训练用了 8 张 P100。",
                    "kind": "fact",
                    "quotes": ["We trained the base model on 8 NVIDIA P100 GPUs for 12 hours."],
                    "doc_id": "p1",
                },
                {
                    "text": "作者称模型参数量为 10 亿。",
                    "kind": "number",
                    "quotes": ["the model has one billion parameters according to the authors"],
                    "doc_id": "p1",
                },
            ],
            ctx,
            fallback_doc_id="p1",
        )
        self.assertTrue(claims[0].evidence[0].anchor.is_located())
        # 找不到的引文不许被"凑合"成某个位置
        self.assertFalse(claims[1].evidence[0].anchor.is_located())


if __name__ == "__main__":
    unittest.main()
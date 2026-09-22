"""主张闸门（Claim Gate）—— 本项目的核心机制。

一句话：**把「抗幻觉」从提示词里的请求，变成系统里的不变量。**

思路来自反射弧：语言模型擅长「写出像样的话」，不擅长「保证这话有出处」。
所以不让模型自己保证，而是在它说完之后，逐句问四个**有确定答案**的问题：

  1. 这句主张有没有带证据？（没有 -> 拦下）
  2. 它引的那段原文，真的存在于那篇论文的那一页吗？（找不到 -> 拦下，因为引文是编的）
  3. 主张里出现的数字，在它引的原文里能找到吗？（找不到 -> 拦下，这是最常见的幻觉形态）
  4. 这段原文是否**足以**支撑这句话？（分档给概率，弱 -> 降级为「待核实」）

前三问是硬检查，纯词法可判，不依赖模型，可离线复现。
第四问是模糊判断，交给一个可替换的裁判（本地启发式 / 大模型 / 真 Jev），
用两个阈值分流 —— 和反射弧的 0.35 / 0.75 同一套路数，阈值可被评测集标定。

结果只有三种：放行 / 降级 / 拦下。任何异常一律降级，绝不静默放行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..config import settings
from ..contracts import Claim, Evidence, GateDecision, Verdict
from ..llm import prompts
from ..llm.client import Gateway, GatewayError, get_gateway
from .evidence import (
    GateContext,
    cross_lingual_support,
    lexical_overlap,
    locate_evidence,
    numbers_in,
    numeric_support,
)


# --------------------------------------------------------------------------
# 配置：阈值就是立场
# --------------------------------------------------------------------------


@dataclass
class GateConfig:
    """闸门参数。默认值由评测集标定，见 evals/。"""

    allow_at: float = 0.75        # 综合分 >= 此值才放行
    block_at: float = 0.35        # 综合分 < 此值直接拦下
    strict_location: bool = True  # 引文定位不到 => 拦下（而不是降级）
    min_evidence: int = 1
    min_located_ratio: float = 0.5
    block_on_missing_number: bool = True

    # 综合分权重
    w_entailment: float = 0.55
    w_overlap: float = 0.25
    w_location: float = 0.20

    def to_dict(self) -> dict:
        return {
            "allow_at": self.allow_at,
            "block_at": self.block_at,
            "strict_location": self.strict_location,
            "weights": {
                "entailment": self.w_entailment,
                "overlap": self.w_overlap,
                "location": self.w_location,
            },
        }


# --------------------------------------------------------------------------
# 裁判：第四问交给谁
# --------------------------------------------------------------------------


@dataclass
class JudgeResult:
    score: float
    reason: str = ""
    missing: list[str] = field(default_factory=list)
    judge: str = "local"


class Judge(Protocol):
    name: str

    def entail(self, claim_text: str, quotes: list[str]) -> JudgeResult:  # pragma: no cover - 协议
        ...


class LocalJudge:
    """离线启发式裁判：不需要网络，也不需要 key。

    它判不出深层语义蕴含，但能抓住两类最要命的问题：数字对不上、以及证据与主张
    在术语层面几乎不相关。**判不了的时候它会说自己判不了**（`local_undecided`），
    而不是硬给一个低分把好主张冤枉掉。
    """

    name = "local"

    def entail(self, claim_text: str, quotes: list[str]) -> JudgeResult:
        if not quotes:
            return JudgeResult(0.0, "没有任何原文片段", ["原文片段"], self.name)

        ok, missing = numeric_support(claim_text, quotes)
        claim_numbers = numbers_in(claim_text)

        if not ok:
            # 数字缺失是最强的负面信号：宁可判死
            return JudgeResult(
                0.2,
                f"主张中的数字 {missing} 在原文片段中找不到",
                [f"数字 {m}" for m in missing],
                self.name,
            )

        score, method = cross_lingual_support(claim_text, quotes)

        if score is None:
            # 跨语言、又没有可对齐的术语 —— 离线裁判在这一层没有判断力
            if claim_numbers:
                return JudgeResult(
                    0.80,
                    "主张中的数字在原文中全部命中；语义层面离线不判",
                    [],
                    self.name,
                )
            return JudgeResult(
                0.50,
                "离线裁判无法跨语言判定语义蕴含，需模型裁判复核",
                [],
                "local_undecided",
            )

        if claim_numbers:
            return JudgeResult(
                min(1.0, 0.55 + 0.45 * min(1.0, score / 0.6)),
                "数字与原文一致，术语覆盖充分",
                [],
                self.name,
            )
        value = min(1.0, score / 0.7)
        reason = (
            "术语覆盖充分"
            if value >= 0.75
            else f"术语覆盖不足（{method}={score:.2f}），证据与主张可能不相关"
        )
        return JudgeResult(round(value, 4), reason, [], self.name)


class LLMJudge:
    """用大模型做蕴含判断。失败一律回落到 LocalJudge，绝不把异常放出去。"""

    name = "llm"

    def __init__(self, gateway: Optional[Gateway] = None, model: Optional[str] = None) -> None:
        self.gateway = gateway or get_gateway()
        self.model = model or settings.fast_model
        self._local = LocalJudge()

    def entail(self, claim_text: str, quotes: list[str]) -> JudgeResult:
        if not quotes:
            return JudgeResult(0.0, "没有任何原文片段", ["原文片段"], self.name)
        try:
            data = self.gateway.json(
                [
                    {"role": "system", "content": prompts.ENTAILMENT_SYSTEM},
                    {"role": "user", "content": prompts.build_entailment_user(claim_text, quotes)},
                ],
                model=self.model,
                max_tokens=400,
                kind="gate.entail",
            )
            score = float(data.get("supports", 0.0))
            return JudgeResult(
                max(0.0, min(1.0, score)),
                str(data.get("reason", ""))[:300],
                [str(m) for m in (data.get("missing") or [])][:6],
                self.name,
            )
        except (GatewayError, ValueError, TypeError, KeyError) as exc:
            fallback = self._local.entail(claim_text, quotes)
            fallback.reason = f"{fallback.reason}（模型裁判不可用，已降级：{type(exc).__name__}）"
            fallback.judge = "local_fallback"
            return fallback


class HybridJudge:
    """先本地判硬伤，再让模型判语义；两者取保守值。"""

    name = "hybrid"

    def __init__(self, gateway: Optional[Gateway] = None, model: Optional[str] = None) -> None:
        self.local = LocalJudge()
        self.llm = LLMJudge(gateway, model)

    def entail(self, claim_text: str, quotes: list[str]) -> JudgeResult:
        local = self.local.entail(claim_text, quotes)
        if local.score < 0.35:
            # 本地已判出硬伤（如数字对不上），不必再花钱问模型
            return local
        llm = self.llm.entail(claim_text, quotes)
        if llm.judge == "local_fallback":
            return local
        if local.judge == "local_undecided":
            return llm
        score = min(local.score, llm.score)
        reasons = [r for r in (llm.reason, local.reason) if r]
        return JudgeResult(score, " | ".join(reasons)[:400], llm.missing, "hybrid")


# --------------------------------------------------------------------------
# 闸门本体
# --------------------------------------------------------------------------


class ClaimGate:
    """逐句裁决。用法：`gate.vet_all(claims, context)`。"""

    def __init__(
        self,
        judge: Optional[Judge] = None,
        config: Optional[GateConfig] = None,
        gateway: Optional[Gateway] = None,
    ) -> None:
        self.config = config or GateConfig()
        if judge is not None:
            self.judge = judge
        else:
            gateway = gateway or get_gateway()
            self.judge = HybridJudge(gateway) if gateway.available else LocalJudge()
        self.stats = {"checked": 0, "allow": 0, "downgrade": 0, "block": 0}

    # ------------------------------------------------------------------ 单条
    def vet(self, claim: Claim, context: Optional[GateContext] = None) -> GateDecision:
        cfg = self.config
        context = context or GateContext()
        checks: dict = {}
        reasons: list[str] = []

        evidence: list[Evidence] = list(claim.evidence or [])
        checks["evidence_count"] = len(evidence)

        # --- 硬检查 1：有没有证据 -------------------------------------------
        if len(evidence) < cfg.min_evidence:
            decision = GateDecision(
                Verdict.BLOCK,
                0.99,
                [f"这句主张没有附带任何原文出处（需要至少 {cfg.min_evidence} 条）"],
                checks,
            )
            self._count(decision)
            claim.decision = decision
            return decision

        # --- 硬检查 2：引文能不能在原文里定位到 -----------------------------
        located = 0
        for ev in evidence:
            if ev.anchor.is_located():
                located += 1
                continue
            pages = context.pages_of(ev.doc_id)
            if pages:
                found = locate_evidence(ev, pages, (context.blocks_by_page or {}).get(ev.doc_id))
                if found is not None:
                    ev.anchor = found
                    ev.score = max(ev.score, 0.8)
                    located += 1
        located_ratio = located / len(evidence)
        checks["located_ratio"] = round(located_ratio, 4)
        checks["located"] = located
        checks["evidence_total"] = len(evidence)

        if cfg.strict_location and located == 0 and context.page_texts:
            decision = GateDecision(
                Verdict.BLOCK,
                0.96,
                ["引文在原文中定位不到 —— 属于疑似编造的出处"],
                checks,
            )
            self._count(decision)
            claim.decision = decision
            return decision
        if located_ratio < cfg.min_located_ratio:
            reasons.append(f"仅 {located}/{len(evidence)} 条引文能在原文中定位")

        quotes = [ev.quote for ev in evidence if ev.quote]

        # --- 硬检查 3：数字一致性 -------------------------------------------
        numbers_ok, missing_numbers = numeric_support(claim.text, quotes)
        checks["numbers_ok"] = numbers_ok
        checks["missing_numbers"] = missing_numbers
        checks["claim_numbers"] = sorted(numbers_in(claim.text))
        if not numbers_ok and cfg.block_on_missing_number:
            decision = GateDecision(
                Verdict.BLOCK,
                0.93,
                [f"主张中的数字 {', '.join(missing_numbers)} 在给出的原文片段中找不到"],
                checks,
            )
            self._count(decision)
            claim.decision = decision
            return decision

        # --- 软检查：词面支撑度 ---------------------------------------------
        overlap = lexical_overlap(claim.text, quotes)
        checks["lexical_overlap"] = round(overlap, 4)

        # --- 软检查：证据是否足以支撑（裁判） -------------------------------
        result = self.judge.entail(claim.text, quotes)
        checks["entailment"] = round(result.score, 4)
        checks["judge"] = result.judge
        if result.missing:
            checks["missing"] = result.missing
        if result.reason:
            reasons.append(result.reason)

        score = (
            cfg.w_entailment * result.score
            + cfg.w_overlap * overlap
            + cfg.w_location * located_ratio
        )
        checks["score"] = round(score, 4)

        if score >= cfg.allow_at:
            verdict = Verdict.ALLOW
        elif score >= cfg.block_at:
            verdict = Verdict.DOWNGRADE
            reasons.append(f"综合置信度 {score:.2f} 未达放行线 {cfg.allow_at}")
        else:
            verdict = Verdict.BLOCK
            reasons.append(f"综合置信度 {score:.2f} 低于拦下线 {cfg.block_at}")

        decision = GateDecision(verdict, score, reasons, checks)
        self._count(decision)
        claim.decision = decision
        return decision

    # ------------------------------------------------------------------ 批量
    def vet_all(
        self, claims: list[Claim], context: Optional[GateContext] = None
    ) -> list[Claim]:
        for claim in claims:
            self.vet(claim, context)
        return claims

    def _count(self, decision: GateDecision) -> None:
        self.stats["checked"] += 1
        self.stats[decision.verdict.value] += 1

    # ------------------------------------------------------------------ 收口
    @staticmethod
    def collect(claims: list[Claim], *, include_downgrade: bool = False) -> list[Claim]:
        """按闸门结论挑出可以进入最终稿的主张。"""
        allow = {Verdict.ALLOW, Verdict.DOWNGRADE} if include_downgrade else {Verdict.ALLOW}
        return [c for c in claims if c.decision and c.decision.verdict in allow]

    @staticmethod
    def summarize(claims: list[Claim]) -> dict:
        """给评测与前端用的统计。"""
        by_verdict = {"allow": 0, "downgrade": 0, "block": 0}
        for c in claims:
            if c.decision:
                by_verdict[c.decision.verdict.value] += 1
        total = len(claims)
        return {
            "total": total,
            "by_verdict": by_verdict,
            "pass_rate": round(by_verdict["allow"] / total, 4) if total else 0.0,
        }
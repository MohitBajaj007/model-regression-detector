"""
Diffing: the core value of the system.

Compares a new run against a baseline run and answers:
    * Did the overall pass rate go up or down, and by how much?
    * Which categories got better or worse?
    * Which exact cases flipped pass -> fail (regressions) or fail -> pass?
    * Is the change big enough to care about (thresholds)?
    * Is it likely real, or could it be noise (McNemar test)?
"""
from __future__ import annotations

from math import comb
from typing import Optional

from pydantic import BaseModel

from .config import Thresholds
from .schemas import CaseResult, EvalRun, Status


class CaseFlip(BaseModel):
    case_id: str
    difficulty: str
    expected_category: str
    old: CaseResult
    new: CaseResult


class Comparison(BaseModel):
    status: Status
    reasons: list[str]
    baseline_run_id: Optional[str]
    baseline_prompt_version: Optional[str]
    pass_rate_old: Optional[float]
    pass_rate_new: float
    pass_rate_delta_pct: float            # percentage points, negative = worse
    category_accuracy_delta_pct: float
    summary_score_delta: float
    latency_delta_ms: float
    token_delta: int
    per_category_delta_pct: dict[str, float]
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    mcnemar_p_value: Optional[float]
    significant: Optional[bool]
    dataset_mismatch: bool = False

    @property
    def headline(self) -> str:
        if self.pass_rate_old is None:
            return f"First run: pass rate {self.pass_rate_new:.0%} (no baseline yet)"
        return (f"{len(self.regressions)} regression(s), {len(self.improvements)} improvement(s); "
                f"pass rate {self.pass_rate_old:.0%} -> {self.pass_rate_new:.0%} "
                f"({self.pass_rate_delta_pct:+.1f} pts)")


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact McNemar test on paired results.

    b = cases that went pass -> fail, c = cases that went fail -> pass.
    If the prompt change made no real difference, flips should go either way
    like coin tosses. The p-value is the chance of a split at least this
    lopsided happening by luck alone. Small p (< 0.05) = probably real.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare_runs(new: EvalRun, baseline: Optional[EvalRun], thresholds: Thresholds) -> Comparison:
    s = new.summary
    if baseline is None:
        return Comparison(
            status="pass", reasons=["No baseline to compare against; this run becomes the reference."],
            baseline_run_id=None, baseline_prompt_version=None, pass_rate_old=None,
            pass_rate_new=s.pass_rate, pass_rate_delta_pct=0.0, category_accuracy_delta_pct=0.0,
            summary_score_delta=0.0, latency_delta_ms=0.0, token_delta=0,
            per_category_delta_pct={}, regressions=[], improvements=[],
            mcnemar_p_value=None, significant=None,
        )

    b = baseline.summary
    old_by_id = {c.case_id: c for c in baseline.cases}
    regressions, improvements = [], []
    for case in new.cases:
        old = old_by_id.get(case.case_id)
        if old is None:
            continue  # case was added after the baseline; nothing to diff
        flip = CaseFlip(case_id=case.case_id, difficulty=case.difficulty.value,
                        expected_category=case.expected_category.value, old=old, new=case)
        if old.passed and not case.passed:
            regressions.append(flip)
        elif not old.passed and case.passed:
            improvements.append(flip)

    pass_delta = (s.pass_rate - b.pass_rate) * 100
    per_cat = {cat: (acc - b.per_category_accuracy.get(cat, acc)) * 100
               for cat, acc in s.per_category_accuracy.items()}
    p_value = mcnemar_exact(len(regressions), len(improvements))
    significant = p_value < thresholds.significance_alpha

    status: Status = "pass"
    reasons: list[str] = []
    drop = -pass_delta
    if drop >= thresholds.critical_delta_pct:
        status = "fail"
        reasons.append(f"CRITICAL: pass rate dropped {drop:.1f} pts "
                       f"(critical threshold {thresholds.critical_delta_pct:.1f}).")
    elif drop >= thresholds.warn_delta_pct:
        status = "warn"
        reasons.append(f"WARNING: pass rate dropped {drop:.1f} pts "
                       f"(warn threshold {thresholds.warn_delta_pct:.1f}).")

    for cat, d in per_cat.items():
        if -d >= thresholds.per_category_warn_pct:
            if status == "pass":
                status = "warn"
            reasons.append(f"WARNING: '{cat}' accuracy dropped {-d:.1f} pts.")

    if s.error_count > b.error_count:
        if status == "pass":
            status = "warn"
        reasons.append(f"WARNING: errors rose from {b.error_count} to {s.error_count}.")

    if regressions:
        noise = "likely real" if significant else "could be noise"
        reasons.append(f"{len(regressions)} case(s) flipped pass->fail, "
                       f"{len(improvements)} fail->pass (McNemar p={p_value:.3f}, {noise}).")
    if status == "pass" and not reasons:
        reasons.append("No regressions beyond thresholds.")

    mismatch = s.dataset_version != b.dataset_version
    if mismatch:
        reasons.append(f"Note: dataset changed ({b.dataset_version} -> {s.dataset_version}); "
                       "some movement may come from new or edited cases.")

    return Comparison(
        status=status, reasons=reasons, baseline_run_id=b.run_id,
        baseline_prompt_version=b.prompt_version, pass_rate_old=b.pass_rate,
        pass_rate_new=s.pass_rate, pass_rate_delta_pct=round(pass_delta, 2),
        category_accuracy_delta_pct=round((s.category_accuracy - b.category_accuracy) * 100, 2),
        summary_score_delta=round(s.avg_summary_score - b.avg_summary_score, 2),
        latency_delta_ms=round(s.avg_latency_ms - b.avg_latency_ms, 1),
        token_delta=s.total_tokens - b.total_tokens,
        per_category_delta_pct={k: round(v, 2) for k, v in per_cat.items()},
        regressions=regressions, improvements=improvements,
        mcnemar_p_value=round(p_value, 4), significant=significant, dataset_mismatch=mismatch,
    )

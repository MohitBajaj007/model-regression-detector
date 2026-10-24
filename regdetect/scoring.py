"""
Multi-dimensional scoring.

For every test case we record four dimensions:
    1. category_correct  - binary exact match (ambiguous cases accept 2 labels)
    2. summary_score     - 1-5, rated by an LLM-as-judge
    3. latency_ms        - how long the call took
    4. tokens            - prompt + completion tokens (cost proxy)

A case PASSES only if the category is right AND the summary scores at least
`summary_pass_score` (default 3).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .llm import LLMProvider
from .schemas import CaseResult, LLMCallResult, TestCase

JUDGE_SYSTEM = """You are a strict evaluator of customer-support email summaries.
Compare a CANDIDATE summary with an IDEAL summary written by a human expert.

Score from 1 to 5:
5 = captures every key fact in the ideal summary, nothing wrong added
4 = captures the main request, misses a minor detail
3 = roughly right but vague or missing an important detail
2 = partially related, misses the main request
1 = wrong, irrelevant, or invented facts

Wording does not matter, meaning does.
Respond with JSON only: {"score": <1-5>, "reason": "<short reason>"}"""


class JudgeVerdict(BaseModel):
    score: int = Field(..., ge=1, le=5)
    reason: str = ""


async def judge_summary(email: str, ideal: str, candidate: str,
                        provider: LLMProvider, model: str) -> JudgeVerdict:
    user = f"EMAIL:\n{email}\nIDEAL SUMMARY: {ideal}\nCANDIDATE SUMMARY: {candidate}"
    resp = await provider.complete_json(system=JUDGE_SYSTEM, user=user, model=model,
                                        temperature=0.0, task="judge")
    return JudgeVerdict(**resp.data)


async def score_case(case: TestCase, call: LLMCallResult, provider: LLMProvider,
                     judge_model: str, summary_pass_score: int) -> CaseResult:
    if call.output is None:
        # The classifier failed (bad JSON, timeout...). That is an automatic fail.
        return CaseResult(
            case_id=case.id, expected_category=case.expected.category,
            predicted_category=None, predicted_summary=None,
            category_correct=False, summary_score=0, passed=False,
            latency_ms=call.latency_ms, prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            difficulty=case.expected_difficulty, error=call.error,
        )

    correct = case.expected.accepts(call.output.category)
    try:
        verdict = await judge_summary(case.input.text, case.expected.summary,
                                      call.output.summary, provider, judge_model)
        summary_score, error = verdict.score, None
    except Exception as exc:
        summary_score, error = 0, f"judge failed: {exc}"

    return CaseResult(
        case_id=case.id, expected_category=case.expected.category,
        predicted_category=call.output.category, predicted_summary=call.output.summary,
        category_correct=correct, summary_score=summary_score,
        passed=correct and summary_score >= summary_pass_score,
        latency_ms=call.latency_ms, prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        difficulty=case.expected_difficulty, error=error,
    )

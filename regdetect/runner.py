"""
The test runner: PromptConfig + GoldenDataset -> EvalRun.

Uses asyncio so that many requests are in flight at once (fast), with a
semaphore capping concurrency (stays under rate limits, controls cost spikes).
"""
from __future__ import annotations

import asyncio
import os
import statistics
import subprocess
import uuid
from datetime import datetime, timezone

from .classifier import classify_email
from .llm import LLMProvider
from .schemas import CaseResult, EvalRun, GoldenDataset, PromptConfig, RunSummary, TestCase
from .scoring import score_case


async def _run_one(case: TestCase, prompt: PromptConfig, provider: LLMProvider,
                   judge_model: str, pass_score: int, sem: asyncio.Semaphore) -> CaseResult:
    async with sem:  # waits here if `concurrency` calls are already running
        call = await classify_email(case.input.text, prompt, provider)
        return await score_case(case, call, provider, judge_model, pass_score)


async def run_eval(prompt: PromptConfig, dataset: GoldenDataset, provider: LLMProvider,
                   *, judge_model: str = "gpt-4o-mini", concurrency: int = 8,
                   summary_pass_score: int = 3, progress: bool = True) -> EvalRun:
    sem = asyncio.Semaphore(concurrency)
    tasks = [_run_one(c, prompt, provider, judge_model, summary_pass_score, sem)
             for c in dataset.cases]

    results: list[CaseResult] = []
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await fut)
        if progress and (i % 10 == 0 or i == len(tasks)):
            print(f"  evaluated {i}/{len(tasks)} cases", flush=True)

    order = {c.id: i for i, c in enumerate(dataset.cases)}
    results.sort(key=lambda r: order[r.case_id])  # as_completed scrambles order
    return EvalRun(summary=summarise(results, prompt, dataset, provider.name), cases=results)


def _git_sha() -> str | None:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def summarise(results: list[CaseResult], prompt: PromptConfig, dataset: GoldenDataset,
              provider_name: str) -> RunSummary:
    n = len(results)
    per_cat: dict[str, list[bool]] = {}
    for r in results:
        per_cat.setdefault(r.expected_category.value, []).append(r.category_correct)

    latencies = sorted(r.latency_ms for r in results)
    p95 = latencies[min(n - 1, int(round(0.95 * (n - 1))))] if n else 0.0
    scored = [r.summary_score for r in results if r.summary_score > 0]
    now = datetime.now(timezone.utc)

    return RunSummary(
        run_id=f"{now:%Y%m%d-%H%M%S}-v{prompt.version}-{uuid.uuid4().hex[:4]}",
        created_at=now,
        prompt_version=prompt.version,
        model=prompt.model,
        dataset_version=dataset.dataset_version,
        provider=provider_name,
        git_sha=_git_sha(),
        total_cases=n,
        pass_rate=sum(r.passed for r in results) / n if n else 0.0,
        category_accuracy=sum(r.category_correct for r in results) / n if n else 0.0,
        avg_summary_score=statistics.fmean(scored) if scored else 0.0,
        per_category_accuracy={k: sum(v) / len(v) for k, v in sorted(per_cat.items())},
        avg_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_ms=p95,
        total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in results),
        error_count=sum(1 for r in results if r.error),
    )

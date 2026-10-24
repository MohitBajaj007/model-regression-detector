"""Tests run fully offline with the mock provider. `pytest -q`"""
import asyncio
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from regdetect.compare import compare_runs, mcnemar_exact
from regdetect.config import DriftConfig, Thresholds, load_settings
from regdetect.dataset import load_dataset
from regdetect.drift import detect_drift, moving_average
from regdetect.llm import MockProvider
from regdetect.prompts import list_prompts, load_prompt, render_system_prompt
from regdetect.classifier import classify_email
from regdetect.runner import run_eval
from regdetect.schemas import CaseResult, ClassificationOutput, EvalRun, RunSummary
from regdetect.storage import RunStore

SETTINGS = load_settings()
DATASET = load_dataset(SETTINGS.path(SETTINGS.dataset))


# ---------- contracts ----------
def test_output_contract_rejects_unknown_category():
    with pytest.raises(ValidationError):
        ClassificationOutput(category="shipping", summary="x")


def test_dataset_is_valid_and_balanced():
    cats = {c.expected.category.value for c in DATASET.cases}
    assert cats == {"billing", "technical", "account", "general"}
    assert 50 <= len(DATASET.cases) <= 100
    assert all(c.notes for c in DATASET.cases), "every case needs a notes field"


def test_all_prompts_load_and_render():
    for p in list_prompts(SETTINGS.path("prompts")):
        text = render_system_prompt(p)
        assert "{categories}" not in text and "- billing:" in text


# ---------- feature under test ----------
def test_classifier_returns_structured_output():
    prompt = load_prompt(SETTINGS.path("prompts/classifier_v1.1.0.yaml"))
    res = asyncio.run(classify_email("I was charged twice, refund please", prompt, MockProvider()))
    assert res.error is None
    assert res.output.category.value == "billing"
    assert res.latency_ms > 0 and res.prompt_tokens > 0


# ---------- statistics ----------
def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0          # balanced flips = pure noise
    assert mcnemar_exact(10, 0) < 0.01         # one-sided flips = real signal
    assert 0.05 < mcnemar_exact(4, 1) < 1.0


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


# ---------- comparison ----------
def _run(pass_flags, run_id="r", provider="mock") -> EvalRun:
    cases = [CaseResult(case_id=f"c{i}", expected_category="billing",
                        predicted_category="billing" if ok else "general",
                        predicted_summary="s", category_correct=ok, summary_score=4,
                        passed=ok, latency_ms=100, prompt_tokens=10, completion_tokens=5,
                        difficulty="easy") for i, ok in enumerate(pass_flags)]
    n = len(cases)
    rate = sum(pass_flags) / n
    summary = RunSummary(run_id=run_id, created_at=datetime.now(timezone.utc),
                         prompt_version="1.0.0", model="m", dataset_version="1.0.0",
                         provider=provider, total_cases=n, pass_rate=rate,
                         category_accuracy=rate, avg_summary_score=4,
                         per_category_accuracy={"billing": rate}, avg_latency_ms=100,
                         p95_latency_ms=100, total_tokens=15 * n, error_count=0)
    return EvalRun(summary=summary, cases=cases)


def test_compare_detects_flips_and_levels():
    t = Thresholds()
    base = _run([True] * 100, "base")
    same = compare_runs(_run([True] * 100, "new"), base, t)
    assert same.status == "pass" and not same.regressions

    warn = compare_runs(_run([False] * 5 + [True] * 95, "new"), base, t)
    assert warn.status == "warn" and len(warn.regressions) == 5

    fail = compare_runs(_run([False] * 10 + [True] * 90, "new"), base, t)
    assert fail.status == "fail" and fail.significant


def test_compare_without_baseline_passes():
    assert compare_runs(_run([True, False]), None, Thresholds()).status == "pass"


# ---------- drift ----------
def test_slow_drift_caught_even_when_each_step_is_small():
    rates = [0.95, 0.94, 0.93, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83]
    history = []
    for i, r in enumerate(rates):
        n_pass = round(r * 100)
        history.append(_run([True] * n_pass + [False] * (100 - n_pass), f"r{i}").summary)
    # every consecutive step is <= 1.5 pts, so per-run checks would never warn
    assert max(abs(a - b) for a, b in zip(rates, rates[1:])) * 100 < Thresholds().warn_delta_pct
    assert detect_drift(history, DriftConfig()).drifting


# ---------- storage ----------
def test_store_roundtrip(tmp_path):
    store = RunStore(tmp_path / "e.db")
    run = _run([True, False, True], "abc")
    store.save(run)
    loaded = store.load_run("abc")
    assert loaded.summary.pass_rate == run.summary.pass_rate
    assert [c.case_id for c in loaded.cases] == ["c0", "c1", "c2"]


# ---------- end to end ----------
def test_bad_prompt_is_flagged_critical():
    provider = MockProvider()
    good = asyncio.run(run_eval(load_prompt(SETTINGS.path("prompts/classifier_v1.1.0.yaml")),
                                DATASET, provider, progress=False))
    bad = asyncio.run(run_eval(load_prompt(SETTINGS.path("prompts/classifier_v1.2.0.yaml")),
                               DATASET, provider, progress=False))
    cmp = compare_runs(bad, good, Thresholds())
    assert cmp.status == "fail"
    assert len(cmp.regressions) > len(cmp.improvements)

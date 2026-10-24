"""
Command-line entry point.

    python -m regdetect run                     # eval the active prompt vs baseline
    python -m regdetect run --prompt prompts/classifier_v1.2.0.yaml
    python -m regdetect promote                 # make the latest run the new baseline
    python -m regdetect history                 # list past runs
    python -m regdetect classify "my card was charged twice"
    python -m regdetect validate                # check dataset + prompt files

Exit codes: 0 = pass or warn, 1 = critical regression (CI blocks the merge),
2 = configuration problem.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from .alerts import build_slack_payload, send_slack, should_alert
from .classifier import classify_email_sync
from .compare import compare_runs
from .config import load_settings
from .dataset import load_dataset
from .drift import detect_drift
from .llm import make_provider
from .prompts import list_prompts, load_prompt
from .report import render_markdown, write_html
from .runner import run_eval
from .storage import RunStore, load_json, save_json


def _provider(settings, override: str | None):
    name = override or settings.provider
    return make_provider(name, settings.openai_api_key, settings.request_timeout_s,
                         settings.max_retries)


def cmd_run(args, settings) -> int:
    prompt_path = settings.path(args.prompt or settings.active_prompt)
    prompt = load_prompt(prompt_path)
    dataset = load_dataset(settings.path(settings.dataset))
    provider = _provider(settings, args.provider)
    print(f"Evaluating prompt v{prompt.version} ({prompt_path.name}) on "
          f"{len(dataset.cases)} cases with provider '{provider.name}'...")

    run = asyncio.run(run_eval(
        prompt, dataset, provider, judge_model=settings.judge_model,
        concurrency=settings.concurrency, summary_pass_score=settings.summary_pass_score,
    ))

    store = RunStore(settings.path(settings.db_path))
    # Baseline: the committed known-good run (default), or the previous run in history.
    if args.against == "previous":
        baseline = store.previous_run(run.summary.run_id, provider=provider.name)
    else:
        baseline_file = settings.path(args.baseline) if args.baseline else settings.baseline_for(provider.name)
        baseline = load_json(baseline_file)
        if baseline and baseline.summary.provider != provider.name:
            print(f"! Baseline was produced by provider '{baseline.summary.provider}', this run "
                  f"uses '{provider.name}'. Comparison is not apples to apples.")

    store.save(run)
    save_json(run, settings.path(f"results/runs/{run.summary.run_id}.json"))

    cmp = compare_runs(run, baseline, settings.thresholds)
    history = store.recent_summaries(limit=max(settings.trend_runs, 2 * settings.drift.window),
                                     provider=provider.name)
    drift = detect_drift(history, settings.drift)
    chart_history = history[-settings.trend_runs:]

    reports = settings.path(settings.reports_dir)
    report_name = f"{run.summary.run_id}.html"
    write_html(reports / report_name, run=run, cmp=cmp, drift=drift, history=chart_history,
               drift_floor=settings.drift.min_pass_rate, baseline=baseline)
    shutil.copy(reports / report_name, reports / "latest.html")

    report_url = settings.report_base_url or None
    md = render_markdown(run, cmp, drift, baseline=baseline, report_url=report_url)
    (reports / "latest.md").write_text(md, encoding="utf-8")
    (reports / "latest.json").write_text(json.dumps({
        "status": cmp.status, "headline": cmp.headline, "drifting": drift.drifting,
        "summary": run.summary.model_dump(mode="json"),
        "comparison": cmp.model_dump(mode="json", exclude={"regressions", "improvements"}),
        "regressed_case_ids": [f.case_id for f in cmp.regressions],
    }, indent=2), encoding="utf-8")

    print()
    print(f"  Status:   {cmp.status.upper()}")
    print(f"  Headline: {cmp.headline}")
    for r in cmp.reasons:
        print(f"  - {r}")
    print(f"  - {drift.message}")
    print(f"\n  Report: {reports / report_name}")

    if not args.no_slack and should_alert(cmp, drift, always=args.always_alert):
        send_slack(build_slack_payload(run, cmp, drift, report_url), settings.slack_webhook_url)

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:  # lets later workflow steps read the result
        with open(gh_out, "a") as fh:
            fh.write(f"status={cmp.status}\nrun_id={run.summary.run_id}\n")

    if args.promote and cmp.status != "fail":
        save_json(run, settings.baseline_for(provider.name))
        print(f"  Promoted {run.summary.run_id} to baseline.")

    return 1 if cmp.status == "fail" and not args.no_fail else 0


def cmd_promote(args, settings) -> int:
    store = RunStore(settings.path(settings.db_path))
    if args.run_id:
        run = store.load_run(args.run_id)
    else:
        latest = store.recent_summaries(limit=1)
        run = store.load_run(latest[-1].run_id) if latest else None
    if not run:
        print("No run found to promote. Run `python -m regdetect run` first.")
        return 2
    target = settings.baseline_for(run.summary.provider)
    save_json(run, target)
    print(f"Baseline is now {run.summary.run_id} (prompt v{run.summary.prompt_version}, "
          f"pass rate {run.summary.pass_rate:.1%}). Commit {target.relative_to(settings.path('.'))}.")
    return 0


def cmd_history(args, settings) -> int:
    store = RunStore(settings.path(settings.db_path))
    rows = store.recent_summaries(limit=args.limit)
    if not rows:
        print("No runs yet.")
        return 0
    print(f"{'run_id':40} {'prompt':8} {'provider':8} {'pass':>6} {'cat acc':>8} {'summ':>5}")
    for s in rows:
        print(f"{s.run_id:40} v{s.prompt_version:7} {s.provider:8} {s.pass_rate:6.1%} "
              f"{s.category_accuracy:8.1%} {s.avg_summary_score:5.2f}")
    return 0


def cmd_classify(args, settings) -> int:
    prompt = load_prompt(settings.path(args.prompt or settings.active_prompt))
    result = classify_email_sync(args.email, prompt, _provider(settings, args.provider))
    print(result.model_dump_json(indent=2))
    return 0 if result.output else 1


def cmd_validate(args, settings) -> int:
    ds = load_dataset(settings.path(settings.dataset))
    prompts = list_prompts(settings.path("prompts"))
    by_cat: dict[str, int] = {}
    for c in ds.cases:
        by_cat[c.expected.category.value] = by_cat.get(c.expected.category.value, 0) + 1
    print(f"Dataset v{ds.dataset_version}: {len(ds.cases)} cases {by_cat}")
    print("Prompts: " + ", ".join(f"v{p.version}" for p in prompts))
    load_prompt(settings.path(settings.active_prompt))
    print(f"Active prompt OK: {settings.active_prompt}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="regdetect", description="LLM regression detection")
    parser.add_argument("--config", help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="evaluate a prompt and compare with the baseline")
    p.add_argument("--prompt", help="prompt YAML (default: active_prompt in config)")
    p.add_argument("--provider", choices=["openai", "mock"])
    p.add_argument("--against", choices=["baseline", "previous"], default="baseline")
    p.add_argument("--baseline", help="baseline JSON path override")
    p.add_argument("--no-slack", action="store_true")
    p.add_argument("--always-alert", action="store_true", help="alert even on pass")
    p.add_argument("--no-fail", action="store_true", help="exit 0 even on critical")
    p.add_argument("--promote", action="store_true", help="save as baseline if not failing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("promote", help="make a run the new baseline")
    p.add_argument("--run-id")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("history", help="list past runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("classify", help="classify one email")
    p.add_argument("email")
    p.add_argument("--prompt")
    p.add_argument("--provider", choices=["openai", "mock"])
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("validate", help="check dataset and prompt files")
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        return args.func(args, settings)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

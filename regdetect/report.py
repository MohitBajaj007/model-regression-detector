"""
Reporting: a self-contained HTML diff report and a Markdown PR comment.

The HTML file has no external dependencies (the trend chart is inline SVG), so
it opens offline and can be uploaded as a CI artifact as-is.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .compare import Comparison
from .drift import DriftResult
from .schemas import EvalRun, RunSummary

TEMPLATES = Path(__file__).parent / "templates"
STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌"}


def trend_svg(history: list[RunSummary], drift: DriftResult, floor: float,
              width: int = 760, height: int = 240) -> str:
    """Line chart of pass rate (solid) and moving average (dashed) per run."""
    if not history:
        return "<p>No history yet.</p>"
    pad_l, pad_r, pad_t, pad_b = 44, 16, 16, 34
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(history)
    lo = max(0.0, min(min(r.pass_rate for r in history), floor) - 0.05)
    hi = 1.0

    def x(i: int) -> float:
        return pad_l + (w * i / (n - 1) if n > 1 else w / 2)

    def y(v: float) -> float:
        return pad_t + h * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
             f'aria-label="Pass rate trend over the last {n} runs">']
    for tick in [lo + (hi - lo) * k / 4 for k in range(5)]:
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{y(tick) + 4:.1f}" class="axis" text-anchor="end">{tick:.0%}</text>')
    parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(floor):.1f}" y2="{y(floor):.1f}" class="floor"/>'
                 f'<text x="{width - pad_r}" y="{y(floor) - 5:.1f}" class="axis" text-anchor="end">drift floor {floor:.0%}</text>')

    pts = " ".join(f"{x(i):.1f},{y(r.pass_rate):.1f}" for i, r in enumerate(history))
    parts.append(f'<polyline points="{pts}" class="line"/>')
    ma = [(i, v) for i, v in enumerate(drift.series[-n:]) if v is not None]
    if len(ma) > 1:
        parts.append('<polyline points="' + " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in ma)
                     + '" class="ma"/>')
    for i, r in enumerate(history):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(r.pass_rate):.1f}" r="3.5" class="dot">'
                     f'<title>{html.escape(r.run_id)}: {r.pass_rate:.1%} (prompt v{r.prompt_version})</title></circle>')
        if n <= 12 or i in (0, n - 1):
            parts.append(f'<text x="{x(i):.1f}" y="{height - 12}" class="axis" text-anchor="middle">v{r.prompt_version}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(run: EvalRun, cmp: Comparison, drift: DriftResult, history: list[RunSummary],
                drift_floor: float, baseline: Optional[EvalRun]) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    tmpl = env.get_template("report.html.j2")
    return tmpl.render(run=run, s=run.summary, cmp=cmp, drift=drift, baseline=baseline,
                       chart=trend_svg(history, drift, drift_floor), icon=STATUS_ICON,
                       trend_n=len(history))


def write_html(path: str | Path, **kwargs) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(**kwargs), encoding="utf-8")
    return p


def render_markdown(run: EvalRun, cmp: Comparison, drift: DriftResult,
                    baseline: Optional[EvalRun] = None, report_url: Optional[str] = None) -> str:
    """Short summary posted as a GitHub PR comment."""
    s = run.summary
    lines = [
        f"## {STATUS_ICON[cmp.status]} LLM eval: **{cmp.status.upper()}**",
        "",
        f"**{cmp.headline}**",
        "",
        f"Prompt `v{s.prompt_version}` · model `{s.model}` · dataset `v{s.dataset_version}` "
        f"· provider `{s.provider}` · run `{s.run_id}`",
        "",
        "| Metric | Baseline | This run | Δ |",
        "|---|---|---|---|",
    ]
    if baseline is not None:
        b = baseline.summary
        lines += [
            f"| Pass rate | {b.pass_rate:.1%} | {s.pass_rate:.1%} | {cmp.pass_rate_delta_pct:+.1f} pts |",
            f"| Category accuracy | {b.category_accuracy:.1%} | {s.category_accuracy:.1%} | {cmp.category_accuracy_delta_pct:+.1f} pts |",
            f"| Avg summary score (1-5) | {b.avg_summary_score:.2f} | {s.avg_summary_score:.2f} | {cmp.summary_score_delta:+.2f} |",
            f"| Avg latency | {b.avg_latency_ms:.0f} ms | {s.avg_latency_ms:.0f} ms | {cmp.latency_delta_ms:+.0f} ms |",
            f"| Total tokens | {b.total_tokens:,} | {s.total_tokens:,} | {cmp.token_delta:+,} |",
        ]
    else:
        lines.append(f"| Pass rate | – | {s.pass_rate:.1%} | – |")
    lines += ["", "**Why:**"] + [f"- {r}" for r in cmp.reasons]
    lines += [f"- {drift.message}"]
    if cmp.regressions:
        lines += ["", "<details><summary>Regressed cases</summary>", "",
                  "| Case | Difficulty | Expected | Was | Now |", "|---|---|---|---|---|"]
        for f in cmp.regressions[:25]:
            was = f.old.predicted_category.value if f.old.predicted_category else "error"
            now = f.new.predicted_category.value if f.new.predicted_category else "error"
            lines.append(f"| `{f.case_id}` | {f.difficulty} | {f.expected_category} | {was} | {now} |")
        lines += ["", "</details>"]
    if report_url:
        lines += ["", f"📄 [Full HTML diff report]({report_url})"]
    if cmp.status == "fail":
        lines += ["", "> Merge is blocked until the critical regression is fixed "
                      "or the baseline is deliberately updated."]
    return "\n".join(lines)

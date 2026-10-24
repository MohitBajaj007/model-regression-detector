"""
Slow-drift detection.

Per-run diffs catch sudden drops. They miss the boiling-frog case: each run is
only 1-2 points worse than the last, so no single run crosses the 3% warning
line, but after ten runs you have lost 12 points.

So we also track a moving average of the pass rate over the last N runs and
warn if:
    * the average falls below an absolute floor (min_pass_rate), or
    * the average fell more than max_drop_pct vs the N runs before it.
"""
from __future__ import annotations

import statistics
from typing import Optional

from pydantic import BaseModel

from .config import DriftConfig
from .schemas import RunSummary


class DriftResult(BaseModel):
    drifting: bool
    message: str
    moving_average: Optional[float]
    previous_average: Optional[float]
    series: list[Optional[float]]  # moving average at each run, for the chart


def moving_average(values: list[float], window: int) -> list[Optional[float]]:
    return [None if i + 1 < window else statistics.fmean(values[i + 1 - window: i + 1])
            for i in range(len(values))]


def detect_drift(history: list[RunSummary], cfg: DriftConfig) -> DriftResult:
    """`history` is oldest -> newest and should include the current run."""
    rates = [h.pass_rate for h in history]
    series = moving_average(rates, cfg.window)
    if len(rates) < cfg.window:
        return DriftResult(drifting=False, series=series, moving_average=None,
                           previous_average=None,
                           message=f"Need {cfg.window} runs for drift detection (have {len(rates)}).")

    current = statistics.fmean(rates[-cfg.window:])
    prev_slice = rates[-2 * cfg.window: -cfg.window]
    previous = statistics.fmean(prev_slice) if len(prev_slice) == cfg.window else None

    problems = []
    if current < cfg.min_pass_rate:
        problems.append(f"{cfg.window}-run average {current:.1%} is below the "
                        f"{cfg.min_pass_rate:.0%} floor")
    if previous is not None and (previous - current) * 100 >= cfg.max_drop_pct:
        problems.append(f"{cfg.window}-run average fell from {previous:.1%} to {current:.1%}")

    if problems:
        return DriftResult(drifting=True, series=series, moving_average=current,
                           previous_average=previous,
                           message="SLOW DRIFT: " + "; ".join(problems) + ".")
    return DriftResult(drifting=False, series=series, moving_average=current,
                       previous_average=previous,
                       message=f"No drift: {cfg.window}-run average is {current:.1%}.")

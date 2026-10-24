"""
Loads config.yaml and lets environment variables override it.

Order of precedence (highest wins):
    1. environment variables   (what Docker / GitHub Actions use)
    2. config.yaml             (what you commit to the repo)
    3. defaults in this file
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Thresholds(BaseModel):
    warn_delta_pct: float = 3.0
    critical_delta_pct: float = 8.0
    per_category_warn_pct: float = 10.0
    significance_alpha: float = 0.05


class DriftConfig(BaseModel):
    window: int = 7
    min_pass_rate: float = 0.85
    max_drop_pct: float = 4.0


class Settings(BaseModel):
    active_prompt: str = "prompts/classifier_v1.1.0.yaml"
    dataset: str = "data/golden_dataset.json"
    db_path: str = "results/evals.db"
    reports_dir: str = "reports"
    baseline_path: str = "baselines/baseline.{provider}.json"
    provider: Literal["openai", "mock"] = "openai"
    judge_model: str = "gpt-4o-mini"
    concurrency: int = Field(8, ge=1, le=64)
    request_timeout_s: float = 30
    max_retries: int = 2
    summary_pass_score: int = Field(3, ge=1, le=5)
    thresholds: Thresholds = Thresholds()
    drift: DriftConfig = DriftConfig()
    trend_runs: int = 20

    # Secrets: only ever read from the environment, never from config.yaml.
    openai_api_key: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    report_base_url: Optional[str] = None  # e.g. link to the CI artifact page

    def baseline_for(self, provider: str) -> Path:
        return self.path(self.baseline_path.format(provider=provider))

    def path(self, rel: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


# env var name -> (dotted settings key, type)
_ENV_OVERRIDES = {
    "ACTIVE_PROMPT": ("active_prompt", str),
    "EVAL_PROVIDER": ("provider", str),
    "JUDGE_MODEL": ("judge_model", str),
    "EVAL_CONCURRENCY": ("concurrency", int),
    "WARN_DELTA_PCT": ("thresholds.warn_delta_pct", float),
    "CRITICAL_DELTA_PCT": ("thresholds.critical_delta_pct", float),
    "PER_CATEGORY_WARN_PCT": ("thresholds.per_category_warn_pct", float),
    "SIGNIFICANCE_ALPHA": ("thresholds.significance_alpha", float),
    "DRIFT_WINDOW": ("drift.window", int),
    "DRIFT_MIN_PASS_RATE": ("drift.min_pass_rate", float),
    "OPENAI_API_KEY": ("openai_api_key", str),
    "SLACK_WEBHOOK_URL": ("slack_webhook_url", str),
    "REPORT_BASE_URL": ("report_base_url", str),
    "EVAL_DB_PATH": ("db_path", str),
}


def _set_dotted(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_settings(config_file: Optional[str | Path] = None) -> Settings:
    path = Path(config_file) if config_file else PROJECT_ROOT / "config.yaml"
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}

    for env_name, (key, cast) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_name)
        if val not in (None, ""):
            _set_dotted(raw, key, cast(val))

    return Settings(**raw)

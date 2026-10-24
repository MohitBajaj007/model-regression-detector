"""
Persistence.

* SQLite (results/evals.db) keeps the full history of every run on this
  machine: one row per run + one row per case. Used for trend charts and drift.
* JSON (baselines/baseline.json) is the single accepted "known good" run. It
  is committed to git so that CI, which starts from a clean machine every time,
  always has something to compare against.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schemas import CaseResult, EvalRun, RunSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    provider       TEXT NOT NULL,
    pass_rate      REAL NOT NULL,
    summary_json   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id             TEXT NOT NULL REFERENCES runs(run_id),
    case_id            TEXT NOT NULL,
    passed             INTEGER NOT NULL,
    category_correct   INTEGER NOT NULL,
    summary_score      INTEGER NOT NULL,
    latency_ms         REAL NOT NULL,
    tokens             INTEGER NOT NULL,
    result_json        TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


class RunStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def save(self, run: EvalRun) -> None:
        s = run.summary
        with self.conn:  # one transaction: all rows saved, or none
            self.conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
                (s.run_id, s.created_at.isoformat(), s.prompt_version, s.model,
                 s.provider, s.pass_rate, s.model_dump_json()),
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO case_results VALUES (?,?,?,?,?,?,?,?)",
                [(s.run_id, c.case_id, int(c.passed), int(c.category_correct),
                  c.summary_score, c.latency_ms, c.prompt_tokens + c.completion_tokens,
                  c.model_dump_json()) for c in run.cases],
            )

    def recent_summaries(self, limit: int = 20, provider: str | None = None) -> list[RunSummary]:
        """Most recent runs, returned oldest -> newest (chart order)."""
        sql, args = "SELECT summary_json FROM runs", []
        if provider:
            sql, args = sql + " WHERE provider = ?", [provider]
        rows = self.conn.execute(sql + " ORDER BY created_at DESC LIMIT ?", [*args, limit])
        return [RunSummary.model_validate_json(r[0]) for r in rows][::-1]

    def load_run(self, run_id: str) -> EvalRun | None:
        row = self.conn.execute("SELECT summary_json FROM runs WHERE run_id=?", [run_id]).fetchone()
        if not row:
            return None
        cases = self.conn.execute(
            "SELECT result_json FROM case_results WHERE run_id=? ORDER BY rowid",
            [run_id]).fetchall()
        return EvalRun(summary=RunSummary.model_validate_json(row[0]),
                       cases=[CaseResult.model_validate_json(c[0]) for c in cases])

    def previous_run(self, before_run_id: str, provider: str | None = None) -> EvalRun | None:
        runs = [s for s in self.recent_summaries(limit=200, provider=provider)
                if s.run_id != before_run_id]
        return self.load_run(runs[-1].run_id) if runs else None


def save_json(run: EvalRun, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(run.model_dump_json(indent=2), encoding="utf-8")


def load_json(path: str | Path) -> EvalRun | None:
    p = Path(path)
    return EvalRun.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None

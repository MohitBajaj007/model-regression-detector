"""
Slack alerts via an Incoming Webhook.

An incoming webhook is just a secret URL. You POST JSON to it and Slack posts
the message in the channel the webhook was created for. No bot, no OAuth.
If no webhook is configured we print the payload instead (dry run), so local
runs and forks without secrets still work.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from .compare import Comparison
from .drift import DriftResult
from .schemas import EvalRun

EMOJI = {"pass": ":white_check_mark:", "warn": ":warning:", "fail": ":rotating_light:"}


def build_slack_payload(run: EvalRun, cmp: Comparison, drift: DriftResult,
                        report_url: Optional[str] = None) -> dict:
    s = run.summary
    status = cmp.status.upper()
    if drift.drifting and cmp.status == "pass":
        status = "DRIFT WARNING"

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text",
                                    "text": f"LLM eval {status}: prompt v{s.prompt_version}"}},
        {"type": "section", "text": {"type": "mrkdwn",
                                     "text": f"{EMOJI[cmp.status]} *{cmp.headline}*"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Model*\n`{s.model}`"},
            {"type": "mrkdwn", "text": f"*Baseline*\nv{cmp.baseline_prompt_version or '—'}"},
            {"type": "mrkdwn", "text": f"*Category accuracy*\n{s.category_accuracy:.1%}"},
            {"type": "mrkdwn", "text": f"*Avg summary score*\n{s.avg_summary_score:.2f} / 5"},
        ]},
        {"type": "section", "text": {"type": "mrkdwn",
                                     "text": "\n".join(f"• {r}" for r in cmp.reasons + [drift.message])}},
    ]
    if cmp.regressions:
        ids = ", ".join(f"`{f.case_id}`" for f in cmp.regressions[:10])
        more = f" and {len(cmp.regressions) - 10} more" if len(cmp.regressions) > 10 else ""
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"Regressed: {ids}{more}"}]})
    if report_url:
        blocks.append({"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Open diff report"},
             "url": report_url}]})

    # `text` is the fallback shown in notifications and on old clients.
    return {"text": f"LLM eval {status}: {cmp.headline}", "blocks": blocks}


def send_slack(payload: dict, webhook_url: Optional[str]) -> bool:
    if not webhook_url:
        print("[slack] SLACK_WEBHOOK_URL not set, dry run. Payload would be:")
        print(json.dumps(payload, indent=2)[:1500])
        return False
    req = urllib.request.Request(webhook_url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
    except Exception as exc:  # an alert failing must never crash the eval
        print(f"[slack] failed to send alert: {exc}")
        return False
    print("[slack] alert sent" if ok else "[slack] unexpected response")
    return ok


def should_alert(cmp: Comparison, drift: DriftResult, always: bool = False) -> bool:
    """Only ping the channel when something needs a human."""
    return always or cmp.status != "pass" or drift.drifting

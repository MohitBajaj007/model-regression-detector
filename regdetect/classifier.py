"""
The LLM feature under test: a customer-support email classifier.

    classify_email(email_text, prompt, provider) -> LLMCallResult

The prompt is a parameter, not a hard-coded string. That is what lets the eval
pipeline run the exact same function against prompt v1.0.0, v1.1.0, v1.2.0...
and compare the results.
"""
from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from .llm import LLMProvider
from .prompts import render_system_prompt
from .schemas import ClassificationOutput, EmailInput, LLMCallResult, PromptConfig


async def classify_email(email_text: str, prompt: PromptConfig,
                         provider: LLMProvider) -> LLMCallResult:
    """Classify one email. Never raises: failures come back in `.error`."""
    email = EmailInput(text=email_text)  # validates the input contract
    system = render_system_prompt(prompt)
    start = time.perf_counter()
    try:
        resp = await provider.complete_json(
            system=system, user=email.text, model=prompt.model,
            temperature=prompt.temperature, task="classify",
        )
        output = ClassificationOutput(**resp.data)  # validates the output contract
        error = None
    except ValidationError as exc:
        resp, output = None, None
        error = f"LLM returned JSON that breaks the contract: {exc.errors()[0]['msg']}"
    except Exception as exc:  # network errors, timeouts, bad JSON
        resp, output = None, None
        error = f"{type(exc).__name__}: {exc}"

    wall_ms = (time.perf_counter() - start) * 1000
    latency = resp.simulated_latency_ms if resp and resp.simulated_latency_ms else wall_ms
    return LLMCallResult(
        output=output, error=error, latency_ms=round(latency, 1),
        prompt_tokens=resp.prompt_tokens if resp else 0,
        completion_tokens=resp.completion_tokens if resp else 0,
    )


def classify_email_sync(email_text: str, prompt: PromptConfig,
                        provider: LLMProvider) -> LLMCallResult:
    """Convenience wrapper for scripts and the REPL."""
    return asyncio.run(classify_email(email_text, prompt, provider))

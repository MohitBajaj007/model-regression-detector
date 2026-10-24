"""
LLM providers.

The rest of the code only talks to the small `LLMProvider` interface below, so
swapping OpenAI for another vendor means writing one new class here.

Two providers ship with the project:

* OpenAIProvider - the real thing (gpt-4o / gpt-4o-mini).
* MockProvider   - a free, offline, deterministic stand-in. It reads the
  category definitions out of the rendered prompt and does keyword matching,
  so a *worse prompt really does produce worse results*. That makes the whole
  pipeline (CI, reports, Slack) demoable without spending a rupee.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Literal, Protocol

Task = Literal["classify", "judge"]


class LLMResponse:
    def __init__(self, data: dict, prompt_tokens: int, completion_tokens: int,
                 simulated_latency_ms: float | None = None):
        self.data = data
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # Mock sets this so latency numbers look realistic; OpenAI leaves it None
        # and the caller measures wall-clock time instead.
        self.simulated_latency_ms = simulated_latency_ms


class LLMProvider(Protocol):
    name: str

    async def complete_json(self, *, system: str, user: str, model: str,
                            temperature: float, task: Task) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, timeout_s: float = 30, max_retries: int = 2):
        from openai import AsyncOpenAI  # imported lazily so mock mode needs no key

        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=max_retries)

    async def complete_json(self, *, system, user, model, temperature, task) -> LLMResponse:
        resp = await self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},  # forces valid JSON back
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content or "{}"
        usage = resp.usage
        return LLMResponse(
            data=json.loads(content),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
_STOPWORDS = set("""
a an the and or but if of to in on for with at by from is are was were be been
it this that these those i me my we our you your he she they them their not no
so as do does did have has had can could would should will just than then
there here what when where which who why how all any some very into about also
stuff things thing anything else everything other others please hi hello thanks
""".split())

_CATEGORY_LINE = re.compile(r"^- (billing|technical|account|general): (.+)$", re.M)


def _stem(word: str) -> str:
    """Tiny suffix-stripper so 'charged', 'charges', 'charging' all match 'charg'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if len(w) > 2 and w not in _STOPWORDS}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _stable_jitter(text: str, low: float, high: float) -> float:
    """Deterministic pseudo-random number derived from the text."""
    h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return low + (h / 0xFFFFFFFF) * (high - low)


class MockProvider:
    name = "mock"

    async def complete_json(self, *, system, user, model, temperature, task) -> LLMResponse:
        await asyncio.sleep(0.002)  # yield to the event loop like a real network call
        if task == "classify":
            data = self._classify(system, user)
        else:
            data = self._judge(user)
        completion = json.dumps(data)
        p_tok, c_tok = _estimate_tokens(system + user), _estimate_tokens(completion)
        latency = 250 + p_tok * 0.4 + c_tok * 12 + _stable_jitter(user, 0, 180)
        return LLMResponse(data, p_tok, c_tok, simulated_latency_ms=round(latency, 1))

    # -- classification: keyword overlap with the prompt's own definitions --
    @staticmethod
    def _classify(system: str, email: str) -> dict:
        definitions = {m.group(1): content_words(m.group(2))
                       for m in _CATEGORY_LINE.finditer(system)}
        email_words = content_words(email)
        scores = {cat: len(email_words & kws) for cat, kws in definitions.items()
                  if cat != "general"}
        best = max(scores, key=scores.get) if scores else "general"
        category = best if scores.get(best, 0) > 0 else "general"
        return {"category": category, "summary": MockProvider._summarise(email)}

    @staticmethod
    def _summarise(email: str) -> str:
        text = re.sub(r"^\s*(hi|hello|hey|dear)[^,\n]*[,\n]", "", email.strip(), flags=re.I)
        words = " ".join(text.split()).split(" ")
        return "Customer writes: " + " ".join(words[:28]).rstrip(".,!?") + "."

    # -- judge: how much of the ideal summary's meaning shows up in the candidate --
    @staticmethod
    def _judge(user: str) -> dict:
        ideal = re.search(r"IDEAL SUMMARY:\s*(.+)", user)
        cand = re.search(r"CANDIDATE SUMMARY:\s*(.+)", user)
        if not (ideal and cand):
            return {"score": 1, "reason": "could not parse judge request"}
        ideal_w = content_words(ideal.group(1)) - {"customer"}
        cand_w = content_words(cand.group(1))
        recall = len(ideal_w & cand_w) / max(1, len(ideal_w))
        score = 5 if recall >= 0.5 else 4 if recall >= 0.35 else 3 if recall >= 0.2 else 2 if recall >= 0.1 else 1
        return {"score": score, "reason": f"mock judge: {recall:.0%} of key ideas covered"}


def make_provider(name: str, api_key: str | None = None, timeout_s: float = 30,
                  max_retries: int = 2) -> LLMProvider:
    if name == "mock":
        return MockProvider()
    if name == "openai":
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it, or run with --provider mock "
                "to use the offline mock provider."
            )
        return OpenAIProvider(api_key, timeout_s, max_retries)
    raise ValueError(f"unknown provider: {name}")

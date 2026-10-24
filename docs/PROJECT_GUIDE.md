# Project Guide: Model Regression Detection System

*A plain-language walkthrough of everything in this repo, written so you can explain any part of it in a presentation or interview.*

---

## 1. The project in one minute

**The problem.** Companies put LLMs inside products, for example a bot that sorts support emails. Someone then "improves" the prompt, or OpenAI quietly updates the model, and the bot starts making mistakes. Nobody notices until customers complain.

**The idea.** Treat a prompt like code. In normal software, every code change runs automated tests before it can be merged. This project does the same for prompts:

1. Keep a set of test emails with the **correct answers** written by a human (the *golden dataset*).
2. Every time the prompt changes, run all test emails through the LLM.
3. Score the answers.
4. **Compare** with the last known-good run (the *baseline*).
5. If things got worse: comment on the pull request, alert Slack, and **block the merge**.

**The one-sentence pitch:**
> "I built CI/CD for prompts: every prompt change is tested against 60 human-labelled emails, diffed case-by-case against a baseline, and blocked from merging if quality drops."

---

## 2. Words you need to know

| Word | Simple meaning |
|---|---|
| **LLM** | Large Language Model, e.g. GPT-4o. Takes text in, gives text out. |
| **Prompt** | The instructions we give the LLM. Here it lives in a YAML file. |
| **System prompt** | The "rules" part of the prompt, sent before the user's text. |
| **Few-shot examples** | Example questions + answers put inside the prompt so the model copies the pattern. |
| **Regression** | Something that used to work and now doesn't. |
| **Golden dataset** | Test inputs with correct answers, checked by a human. "Ground truth". |
| **Baseline** | The saved results of the last run we were happy with. New runs are compared to it. |
| **Eval / evaluation** | Running the test set and scoring the results. |
| **LLM-as-judge** | Using a second LLM call to grade an answer (here: grading summaries 1–5). |
| **CI/CD** | Continuous Integration / Delivery. Automatic checks that run on every code change. |
| **GitHub Actions** | GitHub's built-in CI. Runs scripts on GitHub's machines when you push or open a PR. |
| **PR (pull request)** | A request to merge your changes into the main branch. Others review it. |
| **Pydantic** | A Python library that checks data has the right shape and types. |
| **async / asyncio** | A way to start many network requests at once instead of waiting for each one. |
| **Semaphore** | A counter that says "at most N things at once". Used to limit parallel API calls. |
| **SQLite** | A full database stored in a single file. No server needed. |
| **Webhook** | A secret URL. You POST data to it and the other service (Slack) reacts. |
| **Docker** | Packs the app and everything it needs into one "container" that runs the same everywhere. |
| **Drift** | Slow, gradual worsening over many runs. |
| **Mock** | A fake stand-in for a real service. Our mock LLM runs offline and costs nothing. |

---

## 3. The feature we are testing

A **customer-support email classifier**. You give it an email, it returns JSON:

```json
{"category": "billing", "summary": "Customer was charged twice and wants a refund."}
```

Four allowed categories:

- **billing**: money (charges, refunds, invoices, prices)
- **technical**: something is broken (crash, error, slow)
- **account**: getting in or changing account details (password, login, email change)
- **general**: everything else (feedback, office hours, partnerships)

This is just the *thing being tested*. The real project is the testing system around it. You could swap in any LLM feature.

---

## 4. Folder map

```
model-regression-detector/
├── regdetect/              ← all the Python code (the "package")
│   ├── schemas.py          ← shapes of all data
│   ├── config.py           ← reads settings
│   ├── prompts.py          ← loads prompt YAML files
│   ├── llm.py              ← talks to OpenAI (or the mock)
│   ├── classifier.py       ← THE feature: classify one email
│   ├── scoring.py          ← grades each answer
│   ├── runner.py           ← runs all test cases in parallel
│   ├── storage.py          ← saves results (SQLite + JSON)
│   ├── compare.py          ← diffs new run vs baseline
│   ├── drift.py            ← slow-drift detection
│   ├── report.py           ← builds the HTML report + PR comment
│   ├── alerts.py           ← sends Slack messages
│   ├── cli.py              ← the commands you type
│   └── templates/report.html.j2  ← HTML report template
├── prompts/                ← versioned prompts (v1.0.0, v1.1.0, v1.2.0)
├── data/golden_dataset.json← 60 test emails with correct answers
├── baselines/              ← the saved "known good" run
├── tests/test_pipeline.py  ← automated tests for the code itself
├── .github/workflows/      ← GitHub Actions (CI) files
├── Dockerfile              ← container recipe
├── config.yaml             ← all settings in one place
└── requirements.txt        ← Python libraries to install
```

---

## 5. How one run flows, start to finish

When you type `python -m regdetect run`:

```
1. cli.py         reads config.yaml + env vars           (config.py)
2. cli.py         loads the prompt YAML                  (prompts.py)
3. cli.py         loads the 60 test cases                (dataset.py)
4. runner.py      for each case, in parallel (8 at a time):
     a. classifier.py   sends email + prompt to LLM      (llm.py)
     b. scoring.py      checks category, asks judge to grade the summary
5. runner.py      adds everything up into a RunSummary
6. storage.py     saves the run in SQLite
7. compare.py     diffs it against baselines/baseline.<provider>.json
8. drift.py       checks the moving average of recent runs
9. report.py      writes reports/<run_id>.html and reports/latest.md
10. alerts.py     sends Slack message if status isn't "pass"
11. cli.py        exits with code 1 if CRITICAL → GitHub blocks the merge
```

Keep this list in your head. Almost every presentation question maps to one of these steps.

---

## 6. File-by-file walkthrough

### 6.1 `schemas.py`: the shapes of all data

Everything that moves between files is a **Pydantic model**. A Pydantic model is a Python class where you declare fields and their types; Pydantic then checks real data against it.

```python
class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"

class ClassificationOutput(BaseModel):
    category: Category
    summary: str = Field(..., min_length=1, max_length=400)
```

If the LLM returns `{"category": "shipping"}`, creating `ClassificationOutput(**data)` throws an error, because "shipping" isn't one of the four. That's the **interface contract**: input is email text (`EmailInput`), output is category + summary (`ClassificationOutput`).

Other important models:

- `PromptConfig`: one prompt file (version, timestamp, system prompt, category definitions, few-shot examples). It has a validator that refuses a prompt missing any category definition.
- `TestCase`: one golden test (id, input, expected answer, difficulty, tags, notes).
- `ExpectedOutput.accepts()`: returns True if the predicted category is the expected one **or** in `acceptable_categories` (for genuinely ambiguous emails).
- `CaseResult`: the score for one test case (all four dimensions).
- `RunSummary`: totals for a whole run (pass rate, per-category accuracy, latency, tokens).

**Why it matters:** bad data fails loudly at the edge instead of silently breaking a report later.

### 6.2 `config.py`: settings

Reads `config.yaml`, then lets **environment variables override** it. Order: env var > config.yaml > code default.

```python
_ENV_OVERRIDES = {
    "CRITICAL_DELTA_PCT": ("thresholds.critical_delta_pct", float),
    "OPENAI_API_KEY": ("openai_api_key", str),
    ...
}
```

The dotted name `thresholds.critical_delta_pct` is turned into nested dict keys by `_set_dotted`.

**Secrets (API key, Slack URL) are only read from env vars**, never written in config.yaml, so they never end up on GitHub.

`baseline_for("mock")` turns `baselines/baseline.{provider}.json` into `baselines/baseline.mock.json`.

### 6.3 `prompts/*.yaml` and `prompts.py`: versioned prompts

Each prompt is a YAML file:

```yaml
version: "1.1.0"
created_at: "2026-10-18T14:30:00+05:30"
model: gpt-4o-mini
temperature: 0.0
system_prompt: |
  ...
  Categories:
  {categories}
  Examples:
  {examples}
  Respond with JSON only: {{"category": "...", "summary": "..."}}
category_definitions:
  billing: "Charges, refunds, ..."
few_shot_examples:
  - email: "..."
    category: billing
    summary: "..."
```

- `{categories}` and `{examples}` are **placeholders**. `render_system_prompt()` fills them in with Python's `str.format`.
- `{{` and `}}` are how you write a literal `{` `}` inside a format string. The JSON example needs real braces.
- **Three versions ship:**
  - `v1.0.0`: first try, short definitions, no examples.
  - `v1.1.0`: **active**. Rich definitions, tie-break rules, 3 few-shot examples.
  - `v1.2.0`: the "bad" demo. Someone cut the prompt to save tokens. Tokens drop 80%, accuracy collapses. CI blocks it.

**Why versioned files?** You can see exactly what changed in a PR diff, and you can re-run any old version.

### 6.4 `llm.py`: talking to the model

Defines a tiny **interface** (`LLMProvider`) with one method:

```python
async def complete_json(self, *, system, user, model, temperature, task) -> LLMResponse
```

Any class with this method works. Two exist:

**`OpenAIProvider`**: the real one.
```python
resp = await self.client.chat.completions.create(
    model=model, temperature=temperature,
    response_format={"type": "json_object"},   # model MUST return valid JSON
    messages=[{"role": "system", "content": system},
              {"role": "user", "content": user}],
)
```
It returns the parsed JSON plus token counts from `resp.usage`.

**`MockProvider`**: the free offline fake. This is clever, so understand it:
- It **reads the category definitions out of the actual prompt text** (regex on lines like `- billing: Charges, refunds...`).
- It turns those into keyword sets, and counts how many keywords appear in the email.
- Highest count wins; no hits → `general`.
- So a prompt with vague definitions ("Money stuff.") gives the mock fewer keywords → **worse accuracy, just like a real model would do with a vague prompt.** That's why the v1.2.0 demo fails even in mock mode.
- Its "judge" scores a summary by how many important words of the ideal summary it contains.
- Latency is simulated from token counts so the numbers look realistic.

`_stem()` chops endings (`charged` → `charg`, `charges` → `charg`) so word forms match.

`make_provider("mock" | "openai")` picks one. If you choose openai with no key, it raises a clear error telling you to set the key or use mock.

**Why an interface?** To switch to Claude or Gemini later you add one class here. Nothing else changes.

### 6.5 `classifier.py`: the feature under test

```python
async def classify_email(email_text, prompt, provider) -> LLMCallResult:
    email = EmailInput(text=email_text)          # check input
    system = render_system_prompt(prompt)        # build the prompt
    start = time.perf_counter()
    try:
        resp = await provider.complete_json(...)
        output = ClassificationOutput(**resp.data)   # check output
    except ValidationError:  ... error = "breaks the contract"
    except Exception:        ... error = "timeout / network / bad JSON"
    latency = ...
    return LLMCallResult(output, error, latency_ms, tokens...)
```

Key points:
- **The prompt is a parameter.** Same function, any prompt version. That's what makes comparison possible.
- **It never crashes.** Errors are returned inside the result, so one bad response doesn't kill the whole run of 60.
- `time.perf_counter()` is a high-precision stopwatch for latency.
- `classify_email_sync` is a wrapper for non-async use (`asyncio.run`).

### 6.6 `data/golden_dataset.json`: the test set

60 emails: 16 billing, 16 technical, 16 account, 12 general. One case:

```json
{
  "id": "bil-009",
  "input": {"text": "Wow, amazing. You raised my price by 40% with zero warning..."},
  "expected": {"category": "billing", "acceptable_categories": [],
               "summary": "Customer is upset about an unannounced 40% price increase..."},
  "expected_difficulty": "hard",
  "tags": ["sarcasm"],
  "notes": "Sarcastic tone; positive words ('amazing') must not pull it to general."
}
```

**Edge cases on purpose** (about 40% of the set):

| Type | Example id | Why |
|---|---|---|
| Very short | `bil-007` "refund pls", `gen-007` "hi" | Model must not panic on tiny input |
| Typos | `bil-008`, `tec-006`, `acc-007` | Real customers don't spell-check |
| Mixed language | `bil-010` Hinglish, `bil-014` Spanish, `acc-014` French | Indian & global users code-switch |
| Sarcasm | `bil-009`, `tec-008`, `acc-009` | Positive words, negative meaning |
| Ambiguous | `bil-013` crash + charge, `tec-012` 2FA | Two labels are reasonable → `acceptable_categories` |
| Keyword traps | `tec-011` mentions "invoices" but search is broken | Tests intent vs keywords |

- **Stable IDs**: comparisons match cases by ID, so never renumber.
- **`dataset_version` + `changelog`**: when the test bar changes, it's recorded, and the report warns if baseline and run used different dataset versions.

> ⚠️ **Do this before presenting:** the spec says ground truth must be human-verified. Read through all 60 cases yourself, fix any label or summary you disagree with, and add a few emails from your own experience. Then you can honestly say *you* verified every case. Bump `dataset_version` if you change anything and re-promote the baseline.

### 6.7 `scoring.py`: grading each answer

Four dimensions per case:

1. **category_correct** (yes/no): `case.expected.accepts(predicted)`
2. **summary_score** (1–5): an **LLM-as-judge** call. The judge gets the email, the human's ideal summary, and the model's summary, plus a rubric (5 = every key fact, 1 = wrong). Returns `{"score": 4, "reason": "..."}`.
3. **latency_ms**
4. **tokens** (prompt + completion), a stand-in for cost

**Pass rule:** `passed = category_correct and summary_score >= 3`.

If the classifier errored, the case is an automatic fail with score 0. If the *judge* fails, score 0 and the error is recorded.

**Why not only check category?** A prompt could route perfectly but write useless summaries. Multi-dimensional scoring catches that.

### 6.8 `runner.py`: running everything fast

```python
sem = asyncio.Semaphore(concurrency)        # e.g. 8

async def _run_one(case, ...):
    async with sem:                         # wait if 8 are already running
        call = await classify_email(...)
        return await score_case(...)

tasks = [_run_one(c, ...) for c in dataset.cases]
for fut in asyncio.as_completed(tasks):      # handle each as it finishes
    results.append(await fut)
```

**Explain it like this:** "Instead of sending 60 requests one after another (slow), I start them all, but a semaphore lets only 8 run at the same time so I don't hit OpenAI's rate limit. It's like a restaurant with 8 tables: 60 customers, 8 eat at once."

`as_completed` returns results in finishing order, so they're sorted back into dataset order afterwards.

`summarise()` computes: pass rate, category accuracy, per-category accuracy, average summary score, average and **p95 latency** (95% of calls were faster than this), total tokens, error count, and the git commit SHA.

### 6.9 `storage.py`: remembering results

Two places, for two reasons:

| Where | What | Why |
|---|---|---|
| `results/evals.db` (SQLite) | Every run ever, plus every case result | Trend chart + drift detection |
| `baselines/baseline.<provider>.json` | One "approved" run | Committed to git so CI always has something to compare to |

SQLite tables:
- `runs`: one row per run (id, time, prompt version, pass rate, full summary JSON)
- `case_results`: one row per case per run

`with self.conn:` wraps inserts in a **transaction**: all rows save, or none do.

**Why commit the baseline?** GitHub Actions starts a fresh, empty machine every time. If the baseline only lived in a local database, CI would never find it. Also, changing the baseline becomes a visible, reviewed commit.

### 6.10 `compare.py`: the heart of the project

Takes the new run and the baseline and works out:

1. **Pass rate delta** in percentage points (78% → 33% = −45 pts).
2. **Per-category delta** (billing −81 pts, ...).
3. **Regressions**: cases that passed before and fail now.
4. **Improvements**: failed before, pass now.
5. **Status** using thresholds from config:
   - drop ≥ 8 pts → `fail` (critical, blocks merge)
   - drop ≥ 3 pts → `warn`
   - any category drops ≥ 10 pts → `warn`
   - more errors than baseline → `warn`
6. **Statistical significance** with **McNemar's test**.

**McNemar, explained simply:** look only at the cases that *changed*. Say 5 went pass→fail and 0 went fail→pass. If the prompt change made no real difference, flips would go either way like coin tosses. How likely is getting 5-vs-0 from fair coins? About 6%. That's the **p-value**. Small p (under 0.05) → "probably real". 5-vs-5 → p = 1.0 → "pure noise".

```python
def mcnemar_exact(b, c):
    n = b + c
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
```
`comb(n, i)` = number of ways to choose i items from n. This is the binomial distribution with p = 0.5.

**Design decision to mention:** thresholds decide pass/fail, and the p-value is shown as extra info. With only 60 cases, a real 8-point drop gives p ≈ 0.06, so if significance decided, real regressions would slip through.

### 6.11 `drift.py`: catching slow decline

Per-run checks miss "boiling frog" decline: losing 1.5 points per run never trips the 3% alarm, but after 6 runs you've lost 9 points.

So we compute a **7-run moving average** of pass rate and warn if:
- the average is below 85%, or
- the average fell ≥ 4 pts compared with the 7 runs before it.

```python
def moving_average(values, window):
    return [None if i + 1 < window else mean(values[i+1-window : i+1])
            for i in range(len(values))]
```

The **nightly scheduled run** in GitHub Actions is what feeds this. When OpenAI silently updates a model, nobody opens a PR, but the nightly run catches it.

### 6.12 `report.py` + `templates/report.html.j2`: the HTML report

Uses **Jinja2**, a templating engine: HTML with `{{ variable }}` and `{% for %}` loops that Python fills in.

The report shows:
- Run metadata (prompt version, model, dataset, commit)
- Big status banner with reasons
- Scorecard: pass rate, accuracy, summary score, latency, tokens, flips, each with delta vs baseline
- Per-category table
- **Trend chart** over the last N runs: drawn as **inline SVG** by `trend_svg()` (we compute x/y coordinates in Python), so the file has no outside dependencies and opens offline
- **Regressions side by side**: baseline output vs new output for every case that broke
- Improvements, and an expandable table of all 60 cases

`render_markdown()` builds the shorter version posted as the **PR comment**.

### 6.13 `alerts.py`: Slack

Builds a message in Slack's **Block Kit** format (header, fields, list of reasons, regressed case IDs, "Open diff report" button) and POSTs it to the webhook URL using Python's built-in `urllib`.

- `should_alert()`: only ping when status isn't pass, or drift is detected. No spam on good runs.
- No webhook set → **dry run**: the message is printed instead.
- If Slack is down, it prints the error and carries on. **An alert failing must never fail the eval.**

### 6.14 `cli.py`: the commands

| Command | What it does |
|---|---|
| `python -m regdetect run` | Full eval of the active prompt vs baseline |
| `... run --prompt prompts/classifier_v1.2.0.yaml` | Test a specific prompt |
| `... run --provider mock` | Free offline run |
| `... run --against previous` | Compare with the last run instead of the baseline |
| `... run --promote` | If not failing, save as new baseline |
| `python -m regdetect promote` | Make the latest run the baseline |
| `python -m regdetect history` | Table of past runs |
| `python -m regdetect classify "my card was charged twice"` | Try one email |
| `python -m regdetect validate` | Check dataset + prompt files are valid |

**Exit codes**: 0 = pass/warn, 1 = critical regression, 2 = config error. GitHub Actions treats non-zero as failure; that's how merge blocking works.

It also writes `status=` into `$GITHUB_OUTPUT` so later workflow steps can read the result.

### 6.15 `.github/workflows/eval.yml`: CI

Triggers:
- **pull_request** that changes `prompts/**`, `config.yaml`, or the dataset
- **schedule**: nightly (drift)
- **workflow_dispatch**: a manual "Run" button

Steps:
1. Check out the code, install Python 3.11 and requirements.
2. **Restore SQLite history** from the Actions cache (so drift works across runs).
3. Run the eval. `set +e` stops the script from aborting on exit code 1, so we can still post the comment. The exit code is saved.
4. Save history back to cache.
5. Upload `reports/` as a downloadable **artifact**.
6. **Comment on the PR** using `actions/github-script`. It looks for an earlier comment with a hidden marker and *updates* it instead of posting a new one every push.
7. If the exit code wasn't 0 → fail the job → merge blocked.

Provider choice: `${{ secrets.OPENAI_API_KEY != '' && 'openai' || 'mock' }}`. If the repo has an API key secret, use the real model, otherwise mock.

> Merge blocking needs one manual setting on GitHub: **Settings → Branches → branch protection rule on `main` → Require status checks → `eval`.**

`tests.yml` runs `pytest` and `docker build` on every push.

### 6.16 `Dockerfile`

```dockerfile
FROM python:3.11-slim          # small base image with Python
COPY requirements.txt .
RUN pip install -r requirements.txt   # cached layer: only reruns if deps change
COPY regdetect/ prompts/ data/ baselines/ config.yaml
USER runner                    # don't run as root (security)
ENTRYPOINT ["python", "-m", "regdetect"]
CMD ["run"]
```

Configure it with env vars: `docker run -e OPENAI_API_KEY -e SLACK_WEBHOOK_URL -e CRITICAL_DELTA_PCT=5 regdetect run`.

**Why copy requirements first?** Docker caches each step. Code changes often, dependencies rarely, so installing deps first means rebuilds are fast.

### 6.17 `tests/test_pipeline.py`

11 offline tests (all use the mock), covering:
- contract rejects an unknown category
- dataset is valid, balanced, every case has notes
- all prompts load and render
- classifier returns structured output
- McNemar gives sensible p-values
- compare flags warn at 5% drop and fail at 10%
- drift is caught even when every single step is under the 3% warning
- SQLite save/load round trip
- **end to end:** v1.2.0 vs v1.1.0 → status `fail`

Run: `python -m pytest -q`

---

## 7. The demo story (use this for Loom / presentation)

| | v1.1.0 (baseline) | v1.2.0 ("token saver") |
|---|---|---|
| Pass rate (mock) | 78.3% | 33.3% |
| Tokens per run | ~36,500 | ~6,100 (−83%) |
| Avg latency | ~970 ms | ~770 ms |
| Result | — | ❌ **CRITICAL, merge blocked** |

The story: *"A teammate cut the prompt to save 83% on tokens. Latency improved, cost improved, and the dashboard looked great. But 31 test cases broke, billing accuracy fell 81 points, and CI blocked the merge before a single customer saw it."*

**3-minute Loom script:**

1. **(0:00–0:20) Problem.** "Teams change prompts blind. Here's how I stop bad prompts from shipping."
2. **(0:20–0:50) Show the pieces.** `prompts/` (versioned YAML), `data/golden_dataset.json` (scroll to a sarcasm and a Hinglish case), `config.yaml` thresholds.
3. **(0:50–1:30) Make a bad change.** New branch, set `active_prompt` to `classifier_v1.2.0.yaml`, push, open PR. Show the Action running.
4. **(1:30–2:10) Results.** PR comment with ❌ FAIL and the table; the red check blocking merge; the Slack alert (after you set the webhook).
5. **(2:10–2:45) Diff report.** Download the artifact, open the HTML: scorecard, per-category drop, side-by-side regressions, trend chart.
6. **(2:45–3:00) Wrap.** "Every prompt change is now tested like code. Next: grow the dataset and add drift dashboards."

---

## 8. Questions you might get (with answers)

**Q: Why not just check accuracy?**
Accuracy hides two things: *which* cases broke (two 90% runs can fail different cases) and *summary quality*. I diff per case and score four dimensions.

**Q: Isn't LLM-as-judge unreliable?**
It can be biased, especially if it's the same model family. I use temperature 0, a clear 1–5 rubric, and a human-written ideal summary to compare against. To be safer you'd spot-check judge scores against human scores, or use a stronger judge model.

**Q: How do you know a drop isn't just noise?**
Two layers. Thresholds (3% warn, 8% critical) decide action. McNemar's test on the flipped cases tells you how likely the result is from chance. Temperature 0 also reduces randomness.

**Q: Why SQLite and JSON instead of a real database?**
Zero setup, works in CI, and the baseline is a file in git so changes are reviewed. If many teams used it, I'd move history to Postgres.

**Q: What does the mock provider prove?**
That the pipeline works end to end (CI, diffing, reports, alerts) without spending money. It reads the prompt's own category definitions, so a worse prompt really scores worse. Real quality decisions use the OpenAI provider.

**Q: How would you scale it?**
More test cases (100s), run only changed-category subsets on PRs and the full set nightly, cache LLM responses by (prompt hash, input) to cut cost, and store history centrally.

**Q: What happens when the dataset changes?**
`dataset_version` is bumped, the report warns that the bar moved, and the baseline is re-promoted in the same PR.

**Q: What's p95 latency?**
95% of requests were faster than this number. Averages hide slow outliers; p95 shows them.

**Q: Why async?**
LLM calls are mostly waiting on the network. Async sends many at once; a semaphore caps it at 8 to respect rate limits. 60 calls finish in roughly the time of 8.

**Q: What was the hardest part?**
(Pick your own true answer. Good candidates: designing the comparison so it's actionable per case, writing realistic edge cases, or making merge-blocking work while still posting the PR comment.)

---

## 9. Things to try yourself (to really learn it)

1. Run `python -m regdetect classify "my app keeps crashing"` with each prompt version.
2. Make a v1.3.0 prompt that fixes some of v1.1.0's failures (e.g. `gen-002` compliment, `tec-014` dark mode). Run it and watch improvements appear.
3. Change `critical_delta_pct` to 50 in `config.yaml`, re-run v1.2.0, and see it become a warning.
4. Add 5 new test cases of your own. Bump `dataset_version`.
5. Get an OpenAI key, set `EVAL_PROVIDER=openai`, run, then `promote` to create `baseline.openai.json`.
6. Create a Slack workspace, add a webhook, and send yourself a real alert.
7. Break the code on purpose (e.g. change `>=` to `>` in `compare.py`) and see which test fails.

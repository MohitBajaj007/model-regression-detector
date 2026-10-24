"""
Typed data contracts for the whole pipeline.

Every piece of data that moves between modules is defined here as a Pydantic
model. Pydantic checks types at runtime, so a malformed LLM response or a broken
dataset file fails loudly at the boundary instead of silently corrupting a run.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# The feature under test: customer-support email classification
# --------------------------------------------------------------------------- #
class Category(str, Enum):
    """The only four labels the classifier is allowed to return."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FewShotExample(BaseModel):
    """One example email + answer shown to the model inside the prompt."""

    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """
    A versioned prompt, loaded from a YAML file in /prompts.

    This is the "code" that CI tests. Changing any field here (or the model)
    is a change that must pass the golden dataset before it ships.
    """

    version: str = Field(..., description="Semantic version, e.g. 1.1.0")
    created_at: datetime
    description: str = ""
    model: str = Field("gpt-4o-mini", description="Model the prompt is tuned for")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    system_prompt: str
    category_definitions: dict[Category, str]
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @field_validator("category_definitions")
    @classmethod
    def all_categories_defined(cls, v: dict[Category, str]) -> dict[Category, str]:
        missing = set(Category) - set(v)
        if missing:
            raise ValueError(f"category_definitions is missing: {sorted(m.value for m in missing)}")
        return v


class EmailInput(BaseModel):
    """Input side of the interface contract."""

    text: str = Field(..., min_length=1)


class ClassificationOutput(BaseModel):
    """Output side of the interface contract: what the LLM must return."""

    category: Category
    summary: str = Field(..., min_length=1, max_length=400)


# --------------------------------------------------------------------------- #
# Golden dataset
# --------------------------------------------------------------------------- #
class ExpectedOutput(BaseModel):
    category: Category
    # Some edge cases are genuinely ambiguous: either label is acceptable.
    acceptable_categories: list[Category] = Field(default_factory=list)
    summary: str

    def accepts(self, category: Category) -> bool:
        return category == self.category or category in self.acceptable_categories


class TestCase(BaseModel):
    __test__ = False  # tell pytest this is not a test class

    id: str
    input: EmailInput
    expected: ExpectedOutput
    expected_difficulty: Difficulty
    tags: list[str] = Field(default_factory=list)
    notes: str


class GoldenDataset(BaseModel):
    dataset_version: str
    updated_at: datetime
    changelog: list[str] = Field(default_factory=list)
    cases: list[TestCase]

    @field_validator("cases")
    @classmethod
    def ids_unique(cls, v: list[TestCase]) -> list[TestCase]:
        ids = [c.id for c in v]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate test case ids: {sorted(dupes)}")
        return v


# --------------------------------------------------------------------------- #
# Eval results
# --------------------------------------------------------------------------- #
class LLMCallResult(BaseModel):
    """Raw result of one classifier call, before scoring."""

    output: Optional[ClassificationOutput] = None
    error: Optional[str] = None
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CaseResult(BaseModel):
    """One test case, fully scored. Stored per case in SQLite."""

    case_id: str
    expected_category: Category
    predicted_category: Optional[Category]
    predicted_summary: Optional[str]
    category_correct: bool
    summary_score: int = Field(..., ge=0, le=5)  # 0 = no output / judge failed
    passed: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    difficulty: Difficulty
    error: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    created_at: datetime
    prompt_version: str
    model: str
    dataset_version: str
    provider: str
    git_sha: Optional[str] = None
    total_cases: int
    pass_rate: float
    category_accuracy: float
    avg_summary_score: float
    per_category_accuracy: dict[str, float]
    avg_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    error_count: int


class EvalRun(BaseModel):
    summary: RunSummary
    cases: list[CaseResult]


Status = Literal["pass", "warn", "fail"]

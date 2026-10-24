"""Load versioned prompt files and turn them into the final system prompt text."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schemas import PromptConfig


def load_prompt(path: str | Path) -> PromptConfig:
    """Read one YAML prompt file and validate it into a PromptConfig."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PromptConfig(**data)


def list_prompts(prompts_dir: str | Path) -> list[PromptConfig]:
    """All prompt versions in the folder, oldest version first."""
    prompts = [load_prompt(p) for p in Path(prompts_dir).glob("*.yaml")]
    return sorted(prompts, key=lambda p: tuple(int(x) for x in p.version.split(".")))


def render_system_prompt(prompt: PromptConfig) -> str:
    """
    Fill the {categories} and {examples} placeholders in the template.

    Literal JSON braces in the YAML are written as {{ }} so that Python's
    str.format leaves them alone.
    """
    categories = "\n".join(
        f"- {cat.value}: {' '.join(defn.split())}"
        for cat, defn in prompt.category_definitions.items()
    )
    examples = "\n".join(
        f"Email: {ex.email}\nAnswer: "
        + json.dumps({"category": ex.category.value, "summary": ex.summary})
        for ex in prompt.few_shot_examples
    ) or "(none)"
    return prompt.system_prompt.format(categories=categories, examples=examples)

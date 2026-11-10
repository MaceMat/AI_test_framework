"""Golden datasets: versioned YAML files of test cases.

Example ``datasets/support_bot.yaml``::

    suite: support_bot
    description: Customer support answers
    cases:
      - id: refund-policy-basic
        input: "How long do I have to return an item?"
        expected: "30 days"
        must_include: ["30 days"]
        must_not_include: ["60 days"]
        tags: [smoke, policy]

Any extra keys on a case are kept in ``case.extra`` so each system type
(RAG, agent) can carry its own expectations without schema changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Case(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    input: Any
    expected: Any = None
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    rubric: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def extra(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)


class Dataset(BaseModel):
    suite: str
    description: str = ""
    cases: list[Case]

    @model_validator(mode="after")
    def _unique_ids(self) -> "Dataset":
        seen: set[str] = set()
        dupes = {c.id for c in self.cases if c.id in seen or seen.add(c.id)}
        if dupes:
            raise ValueError(f"Duplicate case ids in suite '{self.suite}': {sorted(dupes)}")
        return self

    def filter(self, tags: Iterable[str] | None = None) -> list[Case]:
        if not tags:
            return list(self.cases)
        wanted = set(tags)
        return [c for c in self.cases if wanted & set(c.tags)]

    def params(self, tags: Iterable[str] | None = None):
        """Return pytest params with readable ids: ``@pytest.mark.parametrize("case", ds.params())``."""
        import pytest

        return [pytest.param(c, id=c.id) for c in self.filter(tags)]


def load_dataset(path: str | Path) -> Dataset:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError(f"{p}: dataset must be a mapping with a 'cases' list")
    data.setdefault("suite", p.stem)
    return Dataset.model_validate(data)

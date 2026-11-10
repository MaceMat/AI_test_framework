"""Deterministic checks on output shape and content. Cheap — run these first."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from aitest.results import CheckResult


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating ```json fences and leading/trailing prose."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", candidate, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


def is_json(output: str) -> CheckResult:
    try:
        _extract_json(output)
        return CheckResult.from_bool("is_json", True)
    except (json.JSONDecodeError, TypeError) as e:
        return CheckResult.from_bool("is_json", False, f"not valid JSON: {e}")


def matches_schema(output: str | dict, model: type[BaseModel]) -> CheckResult:
    """Validate output against a Pydantic model (the contract your feature promises)."""
    try:
        data = output if isinstance(output, dict) else _extract_json(output)
        model.model_validate(data)
        return CheckResult.from_bool(f"schema:{model.__name__}", True)
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, er['loc'])) or '<root>'}: {er['msg']}" for er in e.errors())
        return CheckResult.from_bool(f"schema:{model.__name__}", False, errs)
    except (json.JSONDecodeError, TypeError) as e:
        return CheckResult.from_bool(f"schema:{model.__name__}", False, f"not valid JSON: {e}")


def contains_all(output: str, phrases: Iterable[str], case_sensitive: bool = False) -> CheckResult:
    phrases = list(phrases)
    if not phrases:
        return CheckResult.from_bool("contains_all", True, "nothing required")
    hay = output if case_sensitive else output.lower()
    missing = [p for p in phrases if (p if case_sensitive else p.lower()) not in hay]
    score = 1 - len(missing) / len(phrases)
    return CheckResult(
        "contains_all", score, not missing, f"missing: {missing}" if missing else "", {"missing": missing}
    )


def contains_none(output: str, phrases: Iterable[str], case_sensitive: bool = False) -> CheckResult:
    hay = output if case_sensitive else output.lower()
    found = [p for p in phrases if (p if case_sensitive else p.lower()) in hay]
    return CheckResult.from_bool("contains_none", not found, f"forbidden found: {found}" if found else "")


def matches_regex(output: str, pattern: str) -> CheckResult:
    ok = re.search(pattern, output, flags=re.DOTALL) is not None
    return CheckResult.from_bool("matches_regex", ok, "" if ok else f"no match for /{pattern}/")


def length_between(output: str, min_chars: int = 0, max_chars: int | None = None) -> CheckResult:
    n = len(output)
    ok = n >= min_chars and (max_chars is None or n <= max_chars)
    return CheckResult.from_bool("length_between", ok, f"length={n}", length=n)


def case_expectations(output: str, case: Any) -> list[CheckResult]:
    """Run the must_include / must_not_include lists defined on a dataset Case."""
    checks = []
    if getattr(case, "must_include", None):
        checks.append(contains_all(output, case.must_include))
    if getattr(case, "must_not_include", None):
        checks.append(contains_none(output, case.must_not_include))
    return checks

"""Result objects shared by every assertion and by the recorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Outcome of a single check.

    ``score`` is always normalised to 0..1 so different metrics can be
    aggregated and gated the same way. ``passed`` is score >= threshold.
    """

    name: str
    score: float
    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # allows `assert check(...)`
        return self.passed

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} score={self.score:.2f} {self.reason}".rstrip()

    @classmethod
    def from_bool(cls, name: str, ok: bool, reason: str = "", **details: Any) -> "CheckResult":
        return cls(name=name, score=1.0 if ok else 0.0, passed=ok, reason=reason, details=details)


@dataclass
class EvalRecord:
    """One recorded metric for one test case — what ends up in the JSON report."""

    suite: str
    case_id: str
    metric: str
    score: float
    passed: bool
    reason: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "case_id": self.case_id,
            "metric": self.metric,
            "score": round(self.score, 4),
            "passed": self.passed,
            "reason": self.reason,
            "tags": self.tags,
        }


def require(*checks: CheckResult) -> None:
    """Assert that all checks passed, with a readable failure message."""
    failed = [c for c in checks if not c.passed]
    if failed:
        raise AssertionError("\n".join(str(c) for c in failed))

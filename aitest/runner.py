"""Repeat a non-deterministic check and measure its pass rate."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from aitest.results import CheckResult


@dataclass
class RepeatOutcome:
    name: str
    runs: list[CheckResult] = field(default_factory=list)
    durations_s: list[float] = field(default_factory=list)
    min_pass_rate: float = 1.0

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.runs) / len(self.runs) if self.runs else 0.0

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.min_pass_rate

    @property
    def p95_latency_s(self) -> float:
        if not self.durations_s:
            return 0.0
        s = sorted(self.durations_s)
        return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]

    def as_check(self) -> CheckResult:
        fails = [str(r) for r in self.runs if not r.passed]
        return CheckResult(
            self.name,
            self.pass_rate,
            self.passed,
            f"{sum(r.passed for r in self.runs)}/{len(self.runs)} passed (min {self.min_pass_rate:.0%})"
            + (f"; failures: {fails[:3]}" if fails else ""),
            {"mean_score": self.mean_score, "p95_latency_s": self.p95_latency_s},
        )


def repeat(
    run_once: Callable[[], CheckResult],
    n: int = 3,
    min_pass_rate: float = 1.0,
    name: str | None = None,
) -> RepeatOutcome:
    """Call ``run_once`` n times. ``run_once`` should invoke the system AND check it.

    Example::

        outcome = repeat(lambda: contains_all(bot(q), ["30 days"]), n=5, min_pass_rate=0.8)
        assert outcome.passed, outcome.as_check()
    """
    out = RepeatOutcome(name=name or getattr(run_once, "__name__", "check"), min_pass_rate=min_pass_rate)
    for _ in range(n):
        t0 = time.perf_counter()
        result = run_once()
        out.durations_s.append(time.perf_counter() - t0)
        out.runs.append(result)
        if name is None:
            out.name = result.name
    return out

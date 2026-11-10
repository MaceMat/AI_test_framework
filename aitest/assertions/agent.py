"""Agent trajectory checks: did it use the right tools, in a sane order,
within budget, and stop properly? Final-answer quality is checked with
structural/semantic/judge checks on ``trace.final_answer``.

Dataset fields used by agent cases:
    expected_tools:   tools that must be called (any order)
    expected_order:   tools that must appear in this relative order
    forbidden_tools:  tools that must never be called
    max_steps:        upper bound on LLM turns
    expected_args:    {tool_name: {arg: value}} partial argument match
    allow_tool_errors: true if a failing tool call is part of the scenario
"""

from __future__ import annotations

from typing import Any

from aitest.results import CheckResult
from aitest.types import AgentTrace


def called_tools(trace: AgentTrace, expected: list[str]) -> CheckResult:
    missing = [t for t in expected if t not in trace.tool_names]
    score = 1 - len(missing) / len(expected) if expected else 1.0
    return CheckResult("called_tools", score, not missing, f"missing: {missing}; called: {trace.tool_names}" if missing else "")


def tool_order(trace: AgentTrace, expected_order: list[str]) -> CheckResult:
    """Expected tools appear as a subsequence (other calls may be in between)."""
    it = iter(trace.tool_names)
    ok = all(any(name == want for name in it) for want in expected_order)
    return CheckResult.from_bool("tool_order", ok, "" if ok else f"expected order {expected_order}, got {trace.tool_names}")


def no_forbidden_tools(trace: AgentTrace, forbidden: list[str]) -> CheckResult:
    used = sorted(set(trace.tool_names) & set(forbidden))
    return CheckResult.from_bool("no_forbidden_tools", not used, f"used forbidden: {used}" if used else "")


def within_steps(trace: AgentTrace, max_steps: int) -> CheckResult:
    n = trace.step_count
    return CheckResult.from_bool("within_steps", n <= max_steps, f"steps={n} max={max_steps}", steps=n)


def terminated(trace: AgentTrace) -> CheckResult:
    ok = trace.terminated and bool(trace.final_answer.strip())
    return CheckResult.from_bool("terminated", ok, "" if ok else "no final answer or hit step limit")


def no_tool_errors(trace: AgentTrace) -> CheckResult:
    errs = [c.name for c in trace.tool_calls if c.error]
    return CheckResult.from_bool("no_tool_errors", not errs, f"errors in: {errs}" if errs else "")


def no_repeated_calls(trace: AgentTrace, max_repeats: int = 2) -> CheckResult:
    """Detect loops: the same tool with the same args called too often."""
    from collections import Counter

    keys = Counter((c.name, repr(sorted(c.args.items()))) for c in trace.tool_calls)
    looped = [k[0] for k, n in keys.items() if n > max_repeats]
    return CheckResult.from_bool("no_repeated_calls", not looped, f"looping on: {looped}" if looped else "")


def tool_args_match(trace: AgentTrace, expected_args: dict[str, dict[str, Any]]) -> CheckResult:
    """At least one call to each tool has args that include the expected key/values."""
    problems = []
    for tool, want in expected_args.items():
        calls = [c for c in trace.tool_calls if c.name == tool]
        if not any(all(str(c.args.get(k)).lower() == str(v).lower() for k, v in want.items()) for c in calls):
            problems.append(f"{tool}{want} not matched (got {[c.args for c in calls]})")
    return CheckResult.from_bool("tool_args_match", not problems, "; ".join(problems))


def case_trajectory(trace: AgentTrace, case: Any) -> list[CheckResult]:
    """Run every trajectory check that the dataset case defines."""
    extra = case.extra if hasattr(case, "extra") else dict(case)
    checks = [terminated(trace), no_repeated_calls(trace)]
    if not extra.get("allow_tool_errors"):
        checks.append(no_tool_errors(trace))
    if extra.get("expected_tools"):
        checks.append(called_tools(trace, extra["expected_tools"]))
    if extra.get("expected_order"):
        checks.append(tool_order(trace, extra["expected_order"]))
    if extra.get("forbidden_tools"):
        checks.append(no_forbidden_tools(trace, extra["forbidden_tools"]))
    if extra.get("max_steps"):
        checks.append(within_steps(trace, int(extra["max_steps"])))
    if extra.get("expected_args"):
        checks.append(tool_args_match(trace, extra["expected_args"]))
    return checks

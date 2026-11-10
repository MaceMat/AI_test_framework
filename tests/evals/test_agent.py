"""Agent evals: trajectory (tools, order, args, steps, termination) + final answer."""

from __future__ import annotations

import pytest

from aitest import repeat
from aitest.assertions import agent as agent_checks
from aitest.assertions import structural
from aitest.results import CheckResult
from tests._data import dataset

pytestmark = pytest.mark.eval

DS = dataset("agent_tasks")


@pytest.mark.parametrize("case", DS.params())
def test_agent(case, agent, eval_recorder, repeats):
    traces = []

    def once() -> CheckResult:
        trace = agent(case.input)
        traces.append(trace)
        checks = agent_checks.case_trajectory(trace, case) + structural.case_expectations(trace.final_answer, case)
        return min(checks, key=lambda c: c.score)

    # Agents are the least deterministic systems: measure how consistently the path is right.
    stability = repeat(once, n=repeats, min_pass_rate=1.0, name="trajectory_stability").as_check()

    trace = traces[-1]
    checks = [
        stability,
        *agent_checks.case_trajectory(trace, case),
        *structural.case_expectations(trace.final_answer, case),
    ]
    eval_recorder.check("agent_tasks", case, *checks)

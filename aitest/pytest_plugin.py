"""pytest plugin (auto-loaded via the ``pytest11`` entry point).

Command-line options
--------------------
--aitest-report PATH   where to write the JSON eval report (default reports/eval_results.json)
--aitest-repeats N     how many times eval tests repeat non-deterministic cases (default 1, env AITEST_REPEATS)
--aitest-soft          record failing eval cases without failing pytest; let ``aitest-gate`` decide

Fixtures
--------
judge_llm      LLM client for judges / embeddings (provider from env, FakeLLM by default)
eval_recorder  session-wide recorder: ``eval_recorder.check(suite, case, *checks)``
repeats        the configured repeat count
"""

from __future__ import annotations

import os

import pytest

from aitest.llm import LLMClient, get_llm
from aitest.recorder import EvalRecorder

_RECORDER_KEY = pytest.StashKey[EvalRecorder]()


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("aitest")
    g.addoption("--aitest-report", default="reports/eval_results.json", help="JSON eval report path")
    g.addoption("--aitest-repeats", type=int, default=int(os.environ.get("AITEST_REPEATS", "1")),
                help="repeat count for non-deterministic eval cases")
    g.addoption("--aitest-soft", action="store_true", default=os.environ.get("AITEST_SOFT") == "1",
                help="record eval failures without failing tests; gate in CI instead")


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_RECORDER_KEY] = EvalRecorder(soft=config.getoption("--aitest-soft"))
    for line in (
        "unit: fast deterministic tests, no LLM calls",
        "eval: AI evaluation tests (results recorded)",
        "safety: red-team / safety probes",
    ):
        config.addinivalue_line("markers", line)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    rec = session.config.stash.get(_RECORDER_KEY, None)
    if rec and rec.records:
        path = rec.write(session.config.getoption("--aitest-report"))
        session.config._aitest_report_path = str(path)  # type: ignore[attr-defined]


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # type: ignore[no-untyped-def]
    rec = config.stash.get(_RECORDER_KEY, None)
    if not rec or not rec.records:
        return
    tr = terminalreporter
    tr.section("aitest eval summary")
    for suite, metrics in rec.summary().items():
        tr.write_line(f"{suite}")
        for metric, s in metrics.items():
            tr.write_line(f"  {metric:<24} n={s['n']:<4} pass_rate={s['pass_rate']:.2%}  mean_score={s['mean_score']:.3f}")
    tr.write_line(f"report: {getattr(config, '_aitest_report_path', config.getoption('--aitest-report'))}")


@pytest.fixture(scope="session")
def judge_llm() -> LLMClient:
    return get_llm()


@pytest.fixture(scope="session")
def eval_recorder(request: pytest.FixtureRequest) -> EvalRecorder:
    return request.config.stash[_RECORDER_KEY]


@pytest.fixture(scope="session")
def repeats(request: pytest.FixtureRequest) -> int:
    return max(1, request.config.getoption("--aitest-repeats"))

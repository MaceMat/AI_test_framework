"""Project-level fixtures: how to reach the systems under test.

This is the ONE place a project changes when adopting the framework:
swap the demo systems for adapters to your real app.
"""

from __future__ import annotations

import os
import pytest

from examples.systems import agent_app, rag_app, support_bot

USE_REAL = os.environ.get("AITEST_SUT_PROVIDER", "fake") != "fake"


@pytest.fixture(scope="session")
def bot() -> support_bot.SupportBot:
    if USE_REAL:
        from aitest.llm import get_llm
        return support_bot.SupportBot(get_llm(os.environ["AITEST_SUT_PROVIDER"]))
    return support_bot.SupportBot(support_bot.demo_llm())


@pytest.fixture(scope="session")
def rag() -> rag_app.RagApp:
    return rag_app.RagApp(rag_app.demo_llm(), k=3)


@pytest.fixture(scope="session")
def agent():
    return agent_app.run_agent

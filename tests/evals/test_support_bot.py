"""LLM feature evals: free-text answers and a structured (JSON) output contract."""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from aitest import repeat
from aitest.assertions import judge, semantic, structural
from tests._data import dataset

pytestmark = pytest.mark.eval

ANSWERS = dataset("support_bot")
CLASSIFY = dataset("support_bot_classification")


@pytest.mark.parametrize("case", ANSWERS.params())
def test_answers(case, bot, judge_llm, eval_recorder, repeats):
    # 1) Run the system (n times if configured) and apply cheap deterministic checks.
    outputs: list[str] = []

    def once():
        out = bot.answer(case.input)
        outputs.append(out)
        checks = structural.case_expectations(out, case) or [structural.length_between(out, 1)]
        worst = min(checks, key=lambda c: c.score)
        return worst

    stability = repeat(once, n=repeats, min_pass_rate=1.0, name="deterministic_checks").as_check()

    output = outputs[-1]
    checks = [stability, *structural.case_expectations(output, case)]

    # 2) Semantic similarity to the reference answer (embeddings if available, lexical fallback).
    if case.expected:
        checks.append(semantic.similarity(output, case.expected, llm=judge_llm, threshold=0.3))

    # 3) LLM-as-judge with a rubric (named rubric from RUBRICS or inline text).
    rubric = judge.RUBRICS.get(case.rubric or "correctness", case.rubric)
    checks.append(judge.llm_judge(judge_llm, output, rubric, input=case.input, reference=str(case.expected or "")))

    eval_recorder.check("support_bot", case, *checks)


class Classification(BaseModel):
    intent: Literal["refund", "shipping", "account_security", "other"]
    urgency: Literal["low", "medium", "high"]


@pytest.mark.parametrize("case", CLASSIFY.params())
def test_classification_contract(case, bot, eval_recorder):
    raw = bot.classify(case.input)
    checks = [structural.matches_schema(raw, Classification)]
    if checks[0].passed:
        data = Classification.model_validate(structural._extract_json(raw)).model_dump()
        mismatched = {k: (data.get(k), v) for k, v in case.expected.items() if data.get(k) != v}
        from aitest.results import CheckResult
        checks.append(CheckResult.from_bool("fields_match", not mismatched, f"got/expected: {mismatched}" if mismatched else ""))
    eval_recorder.check("support_bot_classification", case, *checks)

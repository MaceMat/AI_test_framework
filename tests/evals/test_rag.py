"""RAG evals: retrieval and generation are measured separately."""

from __future__ import annotations

import pytest

from aitest.assertions import rag as rag_checks
from aitest.assertions import structural
from tests._data import dataset

pytestmark = pytest.mark.eval

DS = dataset("rag_qa")


@pytest.mark.parametrize("case", DS.params())
def test_rag(case, rag, judge_llm, eval_recorder):
    result = rag.ask(case.input)
    relevant = case.get("relevant_ids", [])

    if case.get("answerable", True) is False:
        # Hallucination guard: unanswerable questions must be declined.
        eval_recorder.check("rag_qa", case, rag_checks.says_dont_know(result))
        return

    checks = [
        # Retrieval (deterministic) — did we fetch the right documents?
        rag_checks.recall_at_k(result, relevant, k=3),
        rag_checks.mrr(result, relevant),
        # Generation — is the answer grounded, relevant and correctly cited?
        rag_checks.faithfulness(judge_llm, result, case.input),
        rag_checks.answer_relevance(judge_llm, result, case.input),
        rag_checks.cites_sources(result),
        *structural.case_expectations(result.answer, case),
    ]
    eval_recorder.check("rag_qa", case, *checks)

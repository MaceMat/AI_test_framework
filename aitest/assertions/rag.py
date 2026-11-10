"""RAG checks. Test retrieval and generation separately — when an answer is
wrong you need to know which half broke.

Dataset cases for RAG carry ``relevant_ids`` (doc ids that should be retrieved).
"""

from __future__ import annotations

from aitest.assertions.judge import RUBRICS, llm_judge
from aitest.llm import LLMClient
from aitest.results import CheckResult
from aitest.types import RagResult

# ---------- retrieval (deterministic, no LLM) ----------


def recall_at_k(result: RagResult, relevant_ids: list[str], k: int = 5, threshold: float = 1.0) -> CheckResult:
    """Share of relevant docs that appear in the top-k retrieved."""
    if not relevant_ids:
        return CheckResult.from_bool(f"recall@{k}", True, "no relevant ids given")
    top = set(result.retrieved_ids[:k])
    hit = [r for r in relevant_ids if r in top]
    score = len(hit) / len(relevant_ids)
    missed = sorted(set(relevant_ids) - set(hit))
    return CheckResult(f"recall@{k}", score, score >= threshold, f"missed: {missed}" if missed else "")


def precision_at_k(result: RagResult, relevant_ids: list[str], k: int = 5, threshold: float = 0.2) -> CheckResult:
    """Share of the top-k retrieved docs that are relevant (noise in the context)."""
    top = result.retrieved_ids[:k]
    if not top:
        return CheckResult(f"precision@{k}", 0.0, False, "nothing retrieved")
    score = sum(1 for d in top if d in set(relevant_ids)) / len(top)
    return CheckResult(f"precision@{k}", score, score >= threshold)


def mrr(result: RagResult, relevant_ids: list[str], threshold: float = 0.5) -> CheckResult:
    """Reciprocal rank of the first relevant doc — is the best doc near the top?"""
    for rank, doc_id in enumerate(result.retrieved_ids, start=1):
        if doc_id in set(relevant_ids):
            score = 1 / rank
            return CheckResult("mrr", score, score >= threshold, f"first relevant at rank {rank}")
    return CheckResult("mrr", 0.0, False, "no relevant doc retrieved")


# ---------- generation (LLM judge) ----------


def faithfulness(llm: LLMClient, result: RagResult, question: str, threshold: float = 0.8) -> CheckResult:
    """Is every claim in the answer supported by the retrieved context? (hallucination check)"""
    return llm_judge(
        llm, result.answer, RUBRICS["faithfulness"], input=question, context=result.context,
        threshold=threshold, name="faithfulness",
    )


def answer_relevance(llm: LLMClient, result: RagResult, question: str, threshold: float = 0.7) -> CheckResult:
    return llm_judge(
        llm, result.answer, RUBRICS["answer_relevance"], input=question, threshold=threshold, name="answer_relevance"
    )


def cites_sources(result: RagResult, required: bool = True) -> CheckResult:
    """If your product shows citations like [doc-id], check they refer to retrieved docs."""
    import re

    cited = set(re.findall(r"\[([\w\-.:/]+)\]", result.answer))
    if not cited:
        return CheckResult.from_bool("cites_sources", not required, "no citations found")
    unknown = cited - set(result.retrieved_ids)
    return CheckResult.from_bool(
        "cites_sources", not unknown, f"cites docs that were not retrieved: {sorted(unknown)}" if unknown else ""
    )


def says_dont_know(result: RagResult, phrases: tuple[str, ...] = ("don't know", "do not know", "not able to find", "no information", "couldn't find", "could not find")) -> CheckResult:
    """For unanswerable questions: the system should admit it instead of inventing."""
    low = result.answer.lower()
    ok = any(p in low for p in phrases)
    return CheckResult.from_bool("says_dont_know", ok, "" if ok else "answered instead of declining")

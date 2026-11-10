"""Meaning-level similarity between an output and a reference answer.

Uses the LLM client's embeddings when available, otherwise a lexical
(bag-of-words cosine) fallback so the check still runs offline. The
fallback is coarse — calibrate thresholds separately for each mode.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from aitest.llm import LLMClient
from aitest.results import CheckResult

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "it", "for", "on",
    "with", "you", "your", "can", "be", "this", "that", "as", "at", "by", "we", "i",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def lexical_similarity(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    vocab = sorted(set(ca) | set(cb))
    return _cosine([ca[t] for t in vocab], [cb[t] for t in vocab])


def similarity(output: str, reference: str, llm: LLMClient | None = None, threshold: float = 0.8) -> CheckResult:
    vectors = llm.embed([output, reference]) if llm is not None else None
    if vectors:
        score, mode = _cosine(vectors[0], vectors[1]), "embedding"
    else:
        score, mode = lexical_similarity(output, reference), "lexical"
    return CheckResult(
        "similarity", score, score >= threshold, f"{mode} threshold={threshold}", {"mode": mode}
    )


def keyword_coverage(output: str, keywords: list[str], threshold: float = 1.0) -> CheckResult:
    """Fraction of key facts/terms present. Use for 'answer must mention X, Y, Z'."""
    if not keywords:
        return CheckResult.from_bool("keyword_coverage", True, "no keywords")
    low = output.lower()
    hit = [k for k in keywords if k.lower() in low]
    score = len(hit) / len(keywords)
    missing = [k for k in keywords if k not in hit]
    return CheckResult("keyword_coverage", score, score >= threshold, f"missing: {missing}" if missing else "")

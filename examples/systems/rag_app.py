"""Example RAG system: keyword retriever + LLM generation."""

from __future__ import annotations

import re

from aitest.llm import FakeLLM, LLMClient
from aitest.types import RagResult, RetrievedDoc

DOCS = {
    "policy-returns": "Items can be returned within 30 days of delivery. Refunds are issued to the original payment method within 5 business days.",
    "policy-shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days and costs 9 EUR.",
    "policy-warranty": "Electronics have a 2 year warranty. The warranty does not cover accidental damage.",
    "faq-account": "You can reset your password from the login page using 'Forgot password'.",
    "faq-payment": "We accept Visa, Mastercard and PayPal. Payment is taken when the order ships.",
}


class RagApp:
    def __init__(self, llm: LLMClient, k: int = 3) -> None:
        self.llm = llm
        self.k = k

    def retrieve(self, question: str) -> list[RetrievedDoc]:
        q = set(re.findall(r"[a-z]+", question.lower()))
        scored = []
        for doc_id, text in DOCS.items():
            words = set(re.findall(r"[a-z]+", text.lower()))
            scored.append((len(q & words), doc_id, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [RetrievedDoc(id=d, text=t, score=float(s)) for s, d, t in scored[: self.k] if s > 0]

    def ask(self, question: str) -> RagResult:
        docs = self.retrieve(question)
        context = "\n".join(f"[{d.id}] {d.text}" for d in docs)
        prompt = f"Answer using only the context. Cite sources as [id]. If unknown, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
        return RagResult(answer=self.llm.complete(prompt), retrieved=docs)


def demo_llm() -> FakeLLM:
    def respond(prompt: str) -> str:
        question = prompt.rsplit("Question:", 1)[-1].lower()
        if "return" in question:
            return "You can return items within 30 days of delivery [policy-returns]."
        if "express" in question or "shipping" in question:
            return "Express shipping takes 1-2 business days and costs 9 EUR [policy-shipping]."
        if "warranty" in question:
            return "Electronics have a 2 year warranty, excluding accidental damage [policy-warranty]."
        return "I don't know based on the available information."

    return FakeLLM(rules=respond)

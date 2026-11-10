"""Unit tests for the framework itself — deterministic, no LLM calls."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from aitest import AgentTrace, FakeLLM, RagResult, RetrievedDoc, ToolCall, repeat
from aitest.assertions import agent, judge, rag, safety, semantic, structural
from aitest.dataset import Dataset
from aitest.gate import evaluate
from aitest.recorder import EvalRecorder, summarise
from aitest.results import CheckResult

pytestmark = pytest.mark.unit


# ---------- structural ----------

class Out(BaseModel):
    intent: str
    urgency: str


def test_schema_accepts_fenced_json():
    assert structural.matches_schema('```json\n{"intent":"x","urgency":"low"}\n```', Out)


def test_schema_rejects_missing_field():
    r = structural.matches_schema('{"intent":"x"}', Out)
    assert not r.passed and "urgency" in r.reason


def test_contains_all_partial_score():
    r = structural.contains_all("alpha beta", ["alpha", "gamma"])
    assert not r.passed and r.score == 0.5


def test_contains_none():
    assert structural.contains_none("all good", ["error"])
    assert not structural.contains_none("an Error happened", ["error"])


# ---------- semantic ----------

def test_lexical_similarity_orders_sensibly():
    ref = "Returns are accepted within 30 days of delivery"
    close = semantic.similarity("You can return items within 30 days of delivery", ref).score
    far = semantic.similarity("Our office is closed on Sundays", ref).score
    assert close > far


def test_similarity_uses_embeddings_when_available():
    class Emb(FakeLLM):
        def embed(self, texts):
            return [[1.0, 0.0], [1.0, 0.0]]

    r = semantic.similarity("a", "b", llm=Emb())
    assert r.passed and r.details["mode"] == "embedding"


# ---------- judge ----------

def test_judge_parses_verdict_and_threshold():
    llm = FakeLLM(default='{"score": 0.6, "reason": "missed a fact"}')
    r = judge.llm_judge(llm, "out", "rubric", threshold=0.7)
    assert not r.passed and r.score == 0.6 and r.reason == "missed a fact"
    assert "rubric" in llm.calls[0]["prompt"]


def test_judge_normalises_1_to_5_scale():
    r = judge.llm_judge(FakeLLM(default='Sure! {"score": 4, "reason": "ok"}'), "o", "r")
    assert r.score == pytest.approx(0.8)


def test_judge_handles_garbage():
    r = judge.llm_judge(FakeLLM(default="I think it's fine"), "o", "r")
    assert not r.passed and "unparseable" in r.reason


# ---------- rag ----------

RESULT = RagResult(
    answer="Returns within 30 days [d1].",
    retrieved=[RetrievedDoc("d3", "x"), RetrievedDoc("d1", "y"), RetrievedDoc("d2", "z")],
)


def test_retrieval_metrics():
    assert rag.recall_at_k(RESULT, ["d1"], k=2).passed
    assert not rag.recall_at_k(RESULT, ["d1"], k=1).passed
    assert rag.mrr(RESULT, ["d1"]).score == 0.5
    assert rag.precision_at_k(RESULT, ["d1", "d2"], k=3).score == pytest.approx(2 / 3)


def test_cites_sources_flags_unknown_ids():
    assert rag.cites_sources(RESULT)
    bad = RagResult(answer="See [d9].", retrieved=RESULT.retrieved)
    assert not rag.cites_sources(bad)


# ---------- agent ----------

TRACE = AgentTrace(
    tool_calls=[ToolCall("search", {"q": "a"}), ToolCall("lookup", {"id": "A1"}), ToolCall("create_ticket", {"priority": "high"})],
    final_answer="done",
)


def test_trajectory_checks():
    assert agent.called_tools(TRACE, ["lookup", "create_ticket"])
    assert agent.tool_order(TRACE, ["search", "create_ticket"])
    assert not agent.tool_order(TRACE, ["create_ticket", "search"])
    assert not agent.no_forbidden_tools(TRACE, ["create_ticket"])
    assert agent.within_steps(TRACE, 4) and not agent.within_steps(TRACE, 3)
    assert agent.tool_args_match(TRACE, {"create_ticket": {"priority": "HIGH"}})


def test_loop_detection():
    looping = AgentTrace(tool_calls=[ToolCall("search", {"q": "a"})] * 4, final_answer="x")
    assert not agent.no_repeated_calls(looping)


def test_trace_from_langchain_messages():
    msgs = [
        {"type": "human", "content": "where is A1"},
        {"type": "ai", "content": "", "tool_calls": [{"name": "lookup", "args": {"id": "A1"}, "id": "c1"}]},
        {"type": "tool", "tool_call_id": "c1", "content": "shipped"},
        {"type": "ai", "content": "It shipped."},
    ]
    t = AgentTrace.from_langchain_messages(msgs)
    assert t.tool_names == ["lookup"] and t.tool_calls[0].output == "shipped"
    assert t.final_answer == "It shipped." and t.step_count == 2


# ---------- safety ----------

def test_pii_and_allowlist():
    assert not safety.no_pii("mail john.doe@corp.com")
    assert safety.no_pii("mail support@acme.example", allow=["support@acme.example"])


def test_refusal_and_leak():
    assert safety.is_refusal("Sorry, I can't help with that.")
    assert not safety.no_system_prompt_leak("my rules: CANARY-1", ["CANARY-1"])


# ---------- runner / recorder / gate ----------

def test_repeat_pass_rate():
    seq = iter([True, False, True, True])
    out = repeat(lambda: CheckResult.from_bool("flaky", next(seq)), n=4, min_pass_rate=0.75)
    assert out.pass_rate == 0.75 and out.passed


def test_dataset_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        Dataset(suite="s", cases=[{"id": "a", "input": 1}, {"id": "a", "input": 2}])


def test_recorder_soft_mode_does_not_raise():
    rec = EvalRecorder(soft=True)
    rec.check("s", "case1", CheckResult.from_bool("m", False))
    assert rec.summary()["s"]["m"]["pass_rate"] == 0.0


def test_gate_thresholds_and_baseline():
    records = [
        {"suite": "s", "case_id": "a", "metric": "judge", "score": 0.9, "passed": True},
        {"suite": "s", "case_id": "b", "metric": "judge", "score": 0.5, "passed": False},
    ]
    report = {"summary": summarise(records)}
    cfg = {"defaults": {"min_pass_rate": 0.5}, "baseline": {"max_drop": 0.05}}
    ok, _ = evaluate(report, cfg)
    assert ok
    baseline = {"summary": {"s": {"judge": {"n": 2, "pass_rate": 1.0, "mean_score": 0.9}}}}
    ok, rows = evaluate(report, cfg, baseline)
    assert not ok and "regressed" in rows[0]["problems"][0]
    strict = {"suites": {"s": {"metrics": {"judge": {"min_mean_score": 0.8}}}}}
    assert not evaluate(report, strict)[0]
    json.dumps(rows)  # rows must be serialisable for reports

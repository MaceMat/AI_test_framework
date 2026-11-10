# aitest — testing framework for AI systems

A small pytest-based framework the team uses to test **LLM features**, **RAG systems** and **tool-using agents**. It runs offline with a fake LLM for fast feedback, and against real models (OpenAI / Azure OpenAI) with an LLM judge and a CI quality gate.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai]"

pytest -m unit                 # framework tests, no LLM
pytest -m eval                 # example evals, offline (fake LLM)
aitest-gate reports/eval_results.json --thresholds thresholds.yaml --baseline baselines/main.json
```

Run against a real model:

```bash
export AITEST_LLM_PROVIDER=azure AITEST_SUT_PROVIDER=azure
export AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... AITEST_JUDGE_MODEL=gpt-4o
pytest -m eval --aitest-repeats 3 --aitest-soft
```

## Layout

```
aitest/                     the framework (copy or pip-install into projects)
  assertions/
    structural.py           JSON/schema, contains, regex, length          (deterministic)
    semantic.py             similarity to reference, keyword coverage      (embeddings or lexical)
    judge.py                LLM-as-judge + reusable RUBRICS
    rag.py                  recall@k, precision@k, MRR, faithfulness, citations, "don't know"
    agent.py                tools called, order, args, forbidden tools, steps, loops, termination
    safety.py               PII, refusal, prompt-leak canaries, injection probes, perturbations
  types.py                  RagResult, AgentTrace (+ LangGraph message adapter)
  dataset.py                YAML golden datasets
  runner.py                 repeat() -> pass rate, latency
  recorder.py               collects results -> reports/eval_results.json
  gate.py                   aitest-gate CLI: thresholds + baseline regression check
  pytest_plugin.py          fixtures: judge_llm, eval_recorder, repeats; CLI options
datasets/                   golden sets (versioned test assets)
examples/systems/           toy systems under test — replace with your app adapters
tests/unit/                 framework tests
tests/evals/                example eval suites (LLM, RAG, agent, safety)
thresholds.yaml             quality gate configuration
baselines/main.json         last accepted eval results
azure-pipelines.yml         CI: fast stage + model eval stage with gate
```

## Adopting it in a project

1. `pip install` this package (or vendor the `aitest/` folder) and add `pytest_plugins = ["aitest.pytest_plugin"]` to your root `conftest.py`.
2. In `tests/conftest.py`, write fixtures that call **your** system and return plain strings, `RagResult`, or `AgentTrace`.
   - LangGraph agent: `AgentTrace.from_langchain_messages(graph.invoke(...)["messages"])`
3. Create `datasets/<suite>.yaml` — start with 20–30 real cases, tag them (`smoke`, `safety`, …).
4. Copy the closest example from `tests/evals/` and adapt the checks.
5. Add the suite to `thresholds.yaml`, run once with the real model, review the results, then save the baseline:
   `aitest-gate reports/eval_results.json --update-baseline baselines/main.json`
6. Add the pipeline. Every failure found in production becomes a new dataset case.

## Writing checks

Every check returns a `CheckResult(name, score 0..1, passed, reason)`. Record them per case:

```python
@pytest.mark.parametrize("case", dataset("support_bot").params())
def test_answers(case, bot, judge_llm, eval_recorder):
    out = bot.answer(case.input)
    eval_recorder.check("support_bot", case,
        *structural.case_expectations(out, case),
        judge.llm_judge(judge_llm, out, judge.RUBRICS["correctness"], input=case.input, reference=case.expected),
    )
```

* Order checks cheap → expensive: structural, then semantic, then judge.
* Use `repeat(fn, n, min_pass_rate)` for non-deterministic behaviour.
* `--aitest-soft` records failures without failing pytest so the **gate** judges the aggregate (use in CI); without it each failing case fails its test (use locally while debugging).
* The default `FakeLLM` judge always passes — offline runs verify wiring and deterministic checks, not quality.

See the team guide for the full process.

## Team guide

Process, datasets, thresholds, judge calibration and definition of done: [AI Testing Framework — Team Guide](https://claude.ai/artifact/DPtzbovxPCwC2myEM8xZqz)

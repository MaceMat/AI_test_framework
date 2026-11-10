"""aitest: a small, opinionated framework for testing AI systems with pytest.

Core ideas
----------
* Golden datasets (YAML) are versioned test assets.
* Assertions return a CheckResult (score + passed + reason) instead of raising,
  so the same check can be used in a pass-rate loop, recorded, and gated in CI.
* Non-deterministic behaviour is measured as a pass *rate* over repeated runs.
* Every eval writes a JSON report; ``aitest-gate`` compares it to thresholds
  (and optionally a baseline) and fails the pipeline on regressions.
"""

from aitest.results import CheckResult, EvalRecord
from aitest.dataset import Case, Dataset, load_dataset
from aitest.llm import LLMClient, FakeLLM, get_llm
from aitest.runner import repeat, RepeatOutcome
from aitest.types import RagResult, RetrievedDoc, AgentTrace, ToolCall

__all__ = [
    "CheckResult",
    "EvalRecord",
    "Case",
    "Dataset",
    "load_dataset",
    "LLMClient",
    "FakeLLM",
    "get_llm",
    "repeat",
    "RepeatOutcome",
    "RagResult",
    "RetrievedDoc",
    "AgentTrace",
    "ToolCall",
]

__version__ = "0.1.0"

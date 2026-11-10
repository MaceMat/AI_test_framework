"""Collects EvalRecords during a pytest session and writes the JSON report."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aitest.results import CheckResult, EvalRecord


class EvalRecorder:
    def __init__(self, soft: bool = False) -> None:
        self.records: list[EvalRecord] = []
        self.soft = soft  # soft = record failures but let the gate decide

    def record(self, suite: str, case_id: str, *checks: CheckResult, tags: list[str] | None = None) -> None:
        for c in checks:
            self.records.append(
                EvalRecord(suite=suite, case_id=case_id, metric=c.name, score=c.score,
                           passed=c.passed, reason=c.reason, tags=list(tags or []))
            )

    def check(self, suite: str, case: Any, *checks: CheckResult) -> None:
        """Record every check for a case, then fail the test if any failed (unless soft mode)."""
        case_id = getattr(case, "id", str(case))
        tags = getattr(case, "tags", [])
        self.record(suite, case_id, *checks, tags=tags)
        failed = [c for c in checks if not c.passed]
        if failed and not self.soft:
            raise AssertionError(f"{suite}/{case_id}:\n  " + "\n  ".join(str(c) for c in failed))

    # ---------- reporting ----------

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        return summarise(self.records)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": _run_meta(),
            "summary": self.summary(),
            "records": [r.to_dict() for r in self.records],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p


def summarise(records: list[EvalRecord] | list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for r in records:
        d = r.to_dict() if isinstance(r, EvalRecord) else r
        groups[(d["suite"], d["metric"])].append(d)
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (suite, metric), rows in sorted(groups.items()):
        out[suite][metric] = {
            "n": len(rows),
            "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 4),
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4),
        }
    return dict(out)


def _run_meta() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.environ.get("BUILD_SOURCEVERSION") or git("rev-parse", "--short", "HEAD"),
        "build_id": os.environ.get("BUILD_BUILDID"),
        "llm_provider": os.environ.get("AITEST_LLM_PROVIDER", "fake"),
        "judge_model": os.environ.get("AITEST_JUDGE_MODEL"),
        "sut_model": os.environ.get("AITEST_SUT_MODEL"),
        "python": platform.python_version(),
    }

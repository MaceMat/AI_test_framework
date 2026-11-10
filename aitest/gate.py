"""Quality gate for CI: compare an eval report to thresholds and a baseline.

    aitest-gate reports/eval_results.json --thresholds thresholds.yaml \
        [--baseline baselines/main.json] [--markdown reports/summary.md]

Exit code 0 = pass, 1 = gate failed, 2 = bad input.

thresholds.yaml::

    defaults:
      min_pass_rate: 0.9          # applies to every suite/metric unless overridden
    baseline:
      max_drop: 0.05              # max allowed drop in mean_score vs baseline
    suites:
      support_bot:
        min_pass_rate: 0.95
        metrics:
          llm_judge: {min_mean_score: 0.8}
          contains_none: {min_pass_rate: 1.0}   # forbidden content: zero tolerance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _limits(cfg: dict[str, Any], suite: str, metric: str) -> dict[str, float]:
    limits = dict(cfg.get("defaults", {}))
    s = cfg.get("suites", {}).get(suite, {}) or {}
    limits.update({k: v for k, v in s.items() if k != "metrics"})
    limits.update((s.get("metrics", {}) or {}).get(metric, {}) or {})
    return limits


def evaluate(report: dict, cfg: dict, baseline: dict | None = None) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    max_drop = (cfg.get("baseline") or {}).get("max_drop")
    base_summary = (baseline or {}).get("summary", {})

    for suite, metrics in report.get("summary", {}).items():
        for metric, s in metrics.items():
            lim = _limits(cfg, suite, metric)
            problems = []
            if "min_pass_rate" in lim and s["pass_rate"] < lim["min_pass_rate"]:
                problems.append(f"pass_rate {s['pass_rate']:.2%} < {lim['min_pass_rate']:.2%}")
            if "min_mean_score" in lim and s["mean_score"] < lim["min_mean_score"]:
                problems.append(f"mean_score {s['mean_score']:.3f} < {lim['min_mean_score']:.3f}")
            base = base_summary.get(suite, {}).get(metric)
            delta = None
            if base is not None:
                delta = round(s["mean_score"] - base["mean_score"], 4)
                if max_drop is not None and delta < -max_drop:
                    problems.append(f"regressed {delta:+.3f} vs baseline (max drop {max_drop})")
            rows.append({
                "suite": suite, "metric": metric, "n": s["n"], "pass_rate": s["pass_rate"],
                "mean_score": s["mean_score"], "delta": delta, "ok": not problems, "problems": problems,
            })

    # Suites/metrics that existed in the baseline but vanished are suspicious.
    for suite, metrics in base_summary.items():
        for metric in metrics:
            if metric not in report.get("summary", {}).get(suite, {}):
                rows.append({"suite": suite, "metric": metric, "n": 0, "pass_rate": None, "mean_score": None,
                             "delta": None, "ok": False, "problems": ["missing from this run (was in baseline)"]})

    return all(r["ok"] for r in rows), rows


def to_markdown(rows: list[dict[str, Any]], ok: bool, meta: dict | None = None) -> str:
    lines = [f"## AI eval gate: {'✅ PASSED' if ok else '❌ FAILED'}", ""]
    if meta:
        lines.append(f"commit `{meta.get('git_commit')}` · provider `{meta.get('llm_provider')}` · judge `{meta.get('judge_model')}`")
        lines.append("")
    lines += ["| Suite | Metric | n | Pass rate | Mean score | Δ baseline | Status |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        pr = "-" if r["pass_rate"] is None else f"{r['pass_rate']:.0%}"
        ms = "-" if r["mean_score"] is None else f"{r['mean_score']:.3f}"
        d = "" if r["delta"] is None else f"{r['delta']:+.3f}"
        st = "✅" if r["ok"] else "❌ " + "; ".join(r["problems"])
        lines.append(f"| {r['suite']} | {r['metric']} | {r['n']} | {pr} | {ms} | {d} | {st} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aitest-gate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    ap.add_argument("--baseline")
    ap.add_argument("--markdown", help="write a markdown summary (e.g. for a PR comment / pipeline summary)")
    ap.add_argument("--update-baseline", help="on pass, copy the report to this path")
    args = ap.parse_args(argv)

    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        cfg = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8")) or {}
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline and Path(args.baseline).exists() else None
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as e:
        print(f"aitest-gate: cannot read inputs: {e}", file=sys.stderr)
        return 2

    ok, rows = evaluate(report, cfg, baseline)
    md = to_markdown(rows, ok, report.get("meta"))
    print(md)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(md, encoding="utf-8")
    if ok and args.update_baseline:
        Path(args.update_baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(args.update_baseline).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

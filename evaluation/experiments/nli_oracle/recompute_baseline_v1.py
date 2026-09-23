"""
Freezes the original Oracle NLI run as "raw NLI baseline v1".

Reuses the raw per-example predictions already collected by
run_oracle_eval.py (evaluation/outputs/nli_oracle_report.json) -
no new model calls. Only the metrics aggregation is recomputed, using
the fixed metrics.py (macro F1 no longer silently drops RELEVANT_NEUTRAL,
and support_to_contradict_rate / neutral_to_contradict_rate are added).

Run with the experiment's isolated environment is NOT required here -
this script only does arithmetic over stored JSON, no torch/transformers
import - the project's main .venv is fine.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.core.io import save_json
from evaluation.experiments.nli_oracle.metrics import compute_metrics_block, compute_transition_rate

SOURCE_REPORT = PROJECT_ROOT / "evaluation/outputs/nli_oracle_report.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_baseline_v1_report.json"


def run() -> dict:
    import json

    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    examples = source["examples"]
    deduped = _deduplicate(examples)

    chunk_level_metrics = compute_metrics_block(examples)
    dedup_metrics = compute_metrics_block(deduped)

    reliability_critical = {
        "chunk_level": {
            "neutral_to_support_rate": compute_transition_rate(examples, "RELEVANT_NEUTRAL", "SUPPORT"),
            "contradict_to_support_rate": compute_transition_rate(examples, "CONTRADICT", "SUPPORT"),
            "support_to_neutral_rate": compute_transition_rate(examples, "SUPPORT", "RELEVANT_NEUTRAL"),
            "support_to_contradict_rate": compute_transition_rate(examples, "SUPPORT", "CONTRADICT"),
            "neutral_to_contradict_rate": compute_transition_rate(examples, "RELEVANT_NEUTRAL", "CONTRADICT"),
        },
        "deduplicated": {
            "neutral_to_support_rate": compute_transition_rate(deduped, "RELEVANT_NEUTRAL", "SUPPORT"),
            "contradict_to_support_rate": compute_transition_rate(deduped, "CONTRADICT", "SUPPORT"),
            "support_to_neutral_rate": compute_transition_rate(deduped, "SUPPORT", "RELEVANT_NEUTRAL"),
            "support_to_contradict_rate": compute_transition_rate(deduped, "SUPPORT", "CONTRADICT"),
            "neutral_to_contradict_rate": compute_transition_rate(deduped, "RELEVANT_NEUTRAL", "CONTRADICT"),
        },
    }

    by_hypothesis = {}
    for ex in examples:
        by_hypothesis.setdefault(ex["hypothesis"], []).append(ex)
    per_hypothesis = {
        hyp: {"n": len(rows), "accuracy": compute_metrics_block(rows)["accuracy"]}
        for hyp, rows in by_hypothesis.items()
    }

    report = {
        "run_metadata": {**source["run_metadata"], "frozen_as": "raw_nli_baseline_v1", "input_version": "v1"},
        "chunk_level_metrics": chunk_level_metrics,
        "deduplicated_metrics": dedup_metrics,
        "reliability_critical_metrics": reliability_critical,
        "per_hypothesis": per_hypothesis,
        "stratified_by_shape": source["stratified_by_shape"],
        "errors": source["errors"],
        "n_errors": source["n_errors"],
        "low_confidence_correct_predictions": source["low_confidence_correct_predictions"],
        "examples": examples,
    }

    save_json(OUTPUT_PATH, report)
    print(f"Baseline v1 frozen (recomputed with fixed metrics) -> {OUTPUT_PATH}")
    print(f"chunk-level accuracy: {chunk_level_metrics['accuracy']}")
    print(f"macro_f1_observed_classes (chunk-level): {chunk_level_metrics['macro_f1_observed_classes']}")
    print(f"macro_f1_observed_classes (dedup): {dedup_metrics['macro_f1_observed_classes']}")

    return report


def _deduplicate(examples: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for ex in examples:
        key = (ex["premise"], ex["hypothesis"], ex["gold_relation"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


if __name__ == "__main__":
    run()

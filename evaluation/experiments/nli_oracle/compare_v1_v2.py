"""
Compares raw NLI baseline v1 vs input-semantics v2 on the dimensions
requested: overall/dedup accuracy, SUPPORT recall, NEUTRAL recall,
support->contradict, neutral->contradict, per-hypothesis results,
structured vs free-text. Same model in both runs - only input
construction differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.core.io import save_json

V1_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_baseline_v1_report.json"
V2_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_v2_report.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_v1_vs_v2_comparison.json"


def run() -> dict:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))

    def pc(report, level, label):
        return report[f"{level}_metrics"]["per_class"][label]

    comparison = {
        "accuracy": {
            "chunk_level": {"v1": v1["chunk_level_metrics"]["accuracy"], "v2": v2["chunk_level_metrics"]["accuracy"]},
            "deduplicated": {"v1": v1["deduplicated_metrics"]["accuracy"], "v2": v2["deduplicated_metrics"]["accuracy"]},
        },
        "macro_f1_observed_classes": {
            "chunk_level": {
                "v1": v1["chunk_level_metrics"]["macro_f1_observed_classes"],
                "v2": v2["chunk_level_metrics"]["macro_f1_observed_classes"],
            },
            "deduplicated": {
                "v1": v1["deduplicated_metrics"]["macro_f1_observed_classes"],
                "v2": v2["deduplicated_metrics"]["macro_f1_observed_classes"],
            },
        },
        "support_recall": {
            "chunk_level": {"v1": pc(v1, "chunk_level", "SUPPORT")["recall"], "v2": pc(v2, "chunk_level", "SUPPORT")["recall"]},
            "deduplicated": {"v1": pc(v1, "deduplicated", "SUPPORT")["recall"], "v2": pc(v2, "deduplicated", "SUPPORT")["recall"]},
        },
        "neutral_recall": {
            "chunk_level": {"v1": pc(v1, "chunk_level", "RELEVANT_NEUTRAL")["recall"], "v2": pc(v2, "chunk_level", "RELEVANT_NEUTRAL")["recall"]},
            "deduplicated": {"v1": pc(v1, "deduplicated", "RELEVANT_NEUTRAL")["recall"], "v2": pc(v2, "deduplicated", "RELEVANT_NEUTRAL")["recall"]},
        },
        "support_to_contradict_rate": {
            "v1": v1["reliability_critical_metrics"]["chunk_level"]["support_to_contradict_rate"]["rate"],
            "v2": v2["reliability_critical_metrics"]["chunk_level"]["support_to_contradict_rate"]["rate"],
        },
        "neutral_to_contradict_rate": {
            "v1": v1["reliability_critical_metrics"]["chunk_level"]["neutral_to_contradict_rate"]["rate"],
            "v2": v2["reliability_critical_metrics"]["chunk_level"]["neutral_to_contradict_rate"]["rate"],
        },
        "per_hypothesis": {
            "v1": v1["per_hypothesis"],
            "v2": v2["per_hypothesis"],
        },
        "structured_vs_free_text": {
            "v2_only_note": "v1 had no structured/free-text split - this dimension is new in v2.",
            "structured": v2["structured_vs_free_text"]["structured"],
            "free_text": v2["structured_vs_free_text"]["free_text"],
        },
        "n_examples": {"v1": v1["chunk_level_metrics"]["n"], "v2": v2["chunk_level_metrics"]["n"]},
        "n_unique_examples": {"v1": v1["deduplicated_metrics"]["n"], "v2": v2["deduplicated_metrics"]["n"]},
    }

    save_json(OUTPUT_PATH, comparison)
    print(f"Saved comparison to {OUTPUT_PATH}")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

    return comparison


if __name__ == "__main__":
    run()

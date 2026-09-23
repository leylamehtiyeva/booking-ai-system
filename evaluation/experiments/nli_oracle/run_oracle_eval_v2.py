"""
Oracle NLI evaluation on the v2 input set (nli_input_v2.jsonl) -
same model as v1 (GuardrailsAI/finetuned_nli_provenance), redesigned
input construction only. See build_input_v2.py for how v2 differs
from v1.

Run with the experiment's isolated environment:
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.run_oracle_eval_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.core.io import load_jsonl, save_json
from evaluation.experiments.nli_oracle.metrics import compute_metrics_block, compute_transition_rate
from evaluation.experiments.nli_oracle.model import MAX_LENGTH, MODEL_NAME, NliOracleModel

INPUT_PATH = Path(__file__).resolve().parent / "nli_input_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_v2_report.json"

LOW_CONFIDENCE_THRESHOLD = 0.6


def deduplicate(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ex in examples:
        key = (ex["premise"], ex["hypothesis"], ex["gold_relation"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def reliability_block(rows: list[dict]) -> dict:
    return {
        "neutral_to_support_rate": compute_transition_rate(rows, "RELEVANT_NEUTRAL", "SUPPORT"),
        "contradict_to_support_rate": compute_transition_rate(rows, "CONTRADICT", "SUPPORT"),
        "support_to_neutral_rate": compute_transition_rate(rows, "SUPPORT", "RELEVANT_NEUTRAL"),
        "support_to_contradict_rate": compute_transition_rate(rows, "SUPPORT", "CONTRADICT"),
        "neutral_to_contradict_rate": compute_transition_rate(rows, "RELEVANT_NEUTRAL", "CONTRADICT"),
    }


def run() -> dict[str, Any]:
    examples = load_jsonl(INPUT_PATH)
    print(f"Loaded {len(examples)} v2 examples from {INPUT_PATH}")

    print(f"Loading model {MODEL_NAME} (max_length={MAX_LENGTH})...")
    model = NliOracleModel()
    print("Model loaded. Running inference...")

    for i, ex in enumerate(examples, start=1):
        result = model.predict(ex["premise"], ex["hypothesis"])
        ex["raw_label"] = result.raw_label
        ex["predicted_relation"] = result.predicted_relation
        ex["probs"] = result.probs
        ex["token_count"] = result.token_count
        ex["truncated"] = result.truncated
        ex["correct"] = ex["predicted_relation"] == ex["gold_relation"]
        ex["top_prob"] = max(result.probs.values())
        if i % 20 == 0 or i == len(examples):
            print(f"  [{i}/{len(examples)}]")

    deduped = deduplicate(examples)

    chunk_level_metrics = compute_metrics_block(examples)
    dedup_metrics = compute_metrics_block(deduped)

    reliability_critical = {
        "chunk_level": reliability_block(examples),
        "deduplicated": reliability_block(deduped),
    }

    by_hypothesis: dict[str, list[dict]] = {}
    for ex in examples:
        by_hypothesis.setdefault(ex["hypothesis"], []).append(ex)
    per_hypothesis = {
        hyp: {
            "subclaim": rows[0]["subclaim"],
            "n": len(rows),
            "accuracy": compute_metrics_block(rows)["accuracy"],
        }
        for hyp, rows in by_hypothesis.items()
    }

    structured_rows = [ex for ex in examples if ex["is_structured"]]
    free_text_rows = [ex for ex in examples if not ex["is_structured"]]
    structured_vs_free_text = {
        "structured": compute_metrics_block(structured_rows),
        "free_text": compute_metrics_block(free_text_rows),
        "structured_dedup": compute_metrics_block(deduplicate(structured_rows)),
        "free_text_dedup": compute_metrics_block(deduplicate(free_text_rows)),
    }

    errors = [
        {
            "pair_id": ex["pair_id"],
            "subclaim": ex["subclaim"],
            "chunk_id": ex["chunk_id"],
            "source_chunk_id": ex["source_chunk_id"],
            "is_structured": ex["is_structured"],
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "gold_relation": ex["gold_relation"],
            "predicted_relation": ex["predicted_relation"],
            "probs": ex["probs"],
        }
        for ex in examples
        if not ex["correct"]
    ]

    low_confidence_correct = [
        {
            "pair_id": ex["pair_id"],
            "subclaim": ex["subclaim"],
            "chunk_id": ex["chunk_id"],
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "gold_relation": ex["gold_relation"],
            "predicted_relation": ex["predicted_relation"],
            "probs": ex["probs"],
            "top_prob": ex["top_prob"],
        }
        for ex in examples
        if ex["correct"] and ex["top_prob"] < LOW_CONFIDENCE_THRESHOLD
    ]

    report = {
        "run_metadata": {
            "model": MODEL_NAME,
            "input_version": "v2",
            "max_length": MAX_LENGTH,
            "threshold_applied": False,
            "n_examples": len(examples),
            "n_unique_examples": len(deduped),
            "n_structured": len(structured_rows),
            "n_free_text": len(free_text_rows),
            "structured_rule": "word_count(premise) <= 3",
            "relation_distribution": {
                label: sum(1 for ex in examples if ex["gold_relation"] == label)
                for label in ["SUPPORT", "CONTRADICT", "RELEVANT_NEUTRAL"]
            },
        },
        "chunk_level_metrics": chunk_level_metrics,
        "deduplicated_metrics": dedup_metrics,
        "reliability_critical_metrics": reliability_critical,
        "per_hypothesis": per_hypothesis,
        "structured_vs_free_text": structured_vs_free_text,
        "errors": errors,
        "n_errors": len(errors),
        "low_confidence_correct_predictions": low_confidence_correct,
        "examples": examples,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== ORACLE NLI v2: SUMMARY ===")
    print(f"chunk-level accuracy: {chunk_level_metrics['accuracy']}  macro_f1_observed: {chunk_level_metrics['macro_f1_observed_classes']}")
    print(f"dedup accuracy:       {dedup_metrics['accuracy']}  macro_f1_observed: {dedup_metrics['macro_f1_observed_classes']}")
    print(f"n_errors: {len(errors)}")
    print(f"Saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()

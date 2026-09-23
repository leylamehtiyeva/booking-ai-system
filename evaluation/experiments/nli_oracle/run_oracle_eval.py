"""
Oracle NLI experiment: evaluates GuardrailsAI/finetuned_nli_provenance
against the hand-labeled (premise, hypothesis, relation) triples in
evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl.

"Oracle" = the model is run on the manually-annotated gold evidence
chunks directly, not on retrieval top-k output - this isolates NLI
quality from retrieval quality, as requested.

Does NOT touch production code, the LLM fallback, the retrieval
implementation, or do any YES/NO/UNCERTAIN aggregation. Raw argmax
only - no confidence threshold is applied to the predicted label.

Run with the experiment's own isolated environment (not the project's
main .venv):
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.run_oracle_eval
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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl"
)
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_report.json"

LOW_CONFIDENCE_THRESHOLD = 0.6  # diagnostic only - not used to gate any decision


def shape_bucket(text: str) -> str:
    n = len(text)
    words = text.split()
    if len(words) <= 3 and n <= 25:
        return "short_tag"
    if n > 200:
        return "long_policy"
    return "sentence"


def build_examples() -> list[dict[str, Any]]:
    pairs = load_jsonl(CHECKPOINT_PATH)
    examples: list[dict[str, Any]] = []

    for pair in pairs:
        for subclaim_key, subclaim in pair["subclaims"].items():
            hypothesis = subclaim["hypothesis"]
            for evidence in subclaim["evidence"]:
                examples.append(
                    {
                        "pair_id": pair["pair_id"],
                        "property_slug": pair["property_slug"],
                        "constraint_text": pair["constraint_text"],
                        "subclaim": subclaim_key,
                        "chunk_id": evidence["chunk_id"],
                        "premise": evidence["text"],
                        "hypothesis": hypothesis,
                        "gold_relation": evidence["relation"],
                        "shape_bucket": shape_bucket(evidence["text"]),
                    }
                )

    return examples


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


def run() -> dict[str, Any]:
    examples = build_examples()
    print(f"Loaded {len(examples)} chunk-level (premise, hypothesis) examples from {CHECKPOINT_PATH}")

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
        if i % 10 == 0 or i == len(examples):
            print(f"  [{i}/{len(examples)}]")

    deduped = deduplicate(examples)

    n_examples = len(examples)
    n_unique_examples = len(deduped)

    n_truncated = sum(1 for ex in examples if ex["truncated"])
    max_token_count = max(ex["token_count"] for ex in examples)

    chunk_level_metrics = compute_metrics_block(examples)
    dedup_metrics = compute_metrics_block(deduped)

    reliability_critical = {
        "chunk_level": {
            "neutral_to_support_rate": compute_transition_rate(examples, "RELEVANT_NEUTRAL", "SUPPORT"),
            "contradict_to_support_rate": compute_transition_rate(examples, "CONTRADICT", "SUPPORT"),
            "support_to_neutral_rate": compute_transition_rate(examples, "SUPPORT", "RELEVANT_NEUTRAL"),
        },
        "deduplicated": {
            "neutral_to_support_rate": compute_transition_rate(deduped, "RELEVANT_NEUTRAL", "SUPPORT"),
            "contradict_to_support_rate": compute_transition_rate(deduped, "CONTRADICT", "SUPPORT"),
            "support_to_neutral_rate": compute_transition_rate(deduped, "SUPPORT", "RELEVANT_NEUTRAL"),
        },
    }

    stratified_by_shape = {}
    for bucket in ("short_tag", "sentence", "long_policy"):
        bucket_rows_chunk = [ex for ex in examples if ex["shape_bucket"] == bucket]
        bucket_rows_dedup = [ex for ex in deduped if ex["shape_bucket"] == bucket]
        stratified_by_shape[bucket] = {
            "chunk_level": compute_metrics_block(bucket_rows_chunk),
            "deduplicated": compute_metrics_block(bucket_rows_dedup),
        }

    errors = [
        {
            "pair_id": ex["pair_id"],
            "subclaim": ex["subclaim"],
            "chunk_id": ex["chunk_id"],
            "shape_bucket": ex["shape_bucket"],
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "gold_relation": ex["gold_relation"],
            "predicted_relation": ex["predicted_relation"],
            "probs": ex["probs"],
            "truncated": ex["truncated"],
        }
        for ex in examples
        if not ex["correct"]
    ]

    low_confidence_correct = [
        {
            "pair_id": ex["pair_id"],
            "subclaim": ex["subclaim"],
            "chunk_id": ex["chunk_id"],
            "shape_bucket": ex["shape_bucket"],
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
            "base_model": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            "label_mapping": {
                "entailment": "SUPPORT",
                "neutral": "RELEVANT_NEUTRAL",
                "contradiction": "CONTRADICT",
            },
            "premise_field": "evidence[].text",
            "hypothesis_field": "subclaims[].hypothesis",
            "max_length": MAX_LENGTH,
            "threshold_applied": False,
            "low_confidence_diagnostic_threshold": LOW_CONFIDENCE_THRESHOLD,
            "dataset_source": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "n_examples": n_examples,
            "n_unique_examples": n_unique_examples,
            "n_truncated": n_truncated,
            "max_token_count_observed": max_token_count,
            "relation_distribution_chunk_level": {
                label: sum(1 for ex in examples if ex["gold_relation"] == label)
                for label in ["SUPPORT", "CONTRADICT", "RELEVANT_NEUTRAL"]
            },
            "relation_distribution_deduplicated": {
                label: sum(1 for ex in deduped if ex["gold_relation"] == label)
                for label in ["SUPPORT", "CONTRADICT", "RELEVANT_NEUTRAL"]
            },
        },
        "chunk_level_metrics": chunk_level_metrics,
        "deduplicated_metrics": dedup_metrics,
        "reliability_critical_metrics": reliability_critical,
        "stratified_by_shape": stratified_by_shape,
        "errors": errors,
        "n_errors": len(errors),
        "low_confidence_correct_predictions": low_confidence_correct,
        "n_low_confidence_correct": len(low_confidence_correct),
        "examples": examples,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== ORACLE NLI EXPERIMENT: SUMMARY ===")
    print(f"n_examples (chunk-level): {n_examples}, n_unique_examples (deduplicated): {n_unique_examples}")
    print(f"truncated: {n_truncated}/{n_examples}, max_token_count_observed: {max_token_count}")
    print()
    print(f"chunk-level accuracy: {chunk_level_metrics['accuracy']}")
    print(f"deduplicated accuracy: {dedup_metrics['accuracy']}")
    print(f"n_errors: {len(errors)}")
    print(f"n_low_confidence_correct: {len(low_confidence_correct)}")
    print()
    print(f"Full report saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()

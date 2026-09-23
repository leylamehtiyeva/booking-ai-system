"""
Runs the A/B/C semantic verifier comparison on the FROZEN held-out
semantic challenge set (benchmark_v1.jsonl, N=65).

A: GuardrailsAI/finetuned_nli_provenance (current baseline)
   - A1: honest 3-class (entailment/neutral/contradiction -> SUPPORT/NOT_ENOUGH_EVIDENCE/CONTRADICT)
   - A2: SUPPORT-only mode (entailment -> SUPPORT, neutral/contradiction -> NOT_ENOUGH_EVIDENCE)
B: MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli
C: gemini-2.5-flash, structured JSON output, temperature=0, no few-shot

Primary metrics: high-confidence examples only.
Secondary: medium-confidence examples, reported separately.
Template-deduplicated metrics computed additionally on the primary set.

All raw scores/probabilities/logits/LLM responses are preserved in the
output report for a future, separate ranking experiment.

Run with the isolated environment (torch/transformers + google-genai,
all now in the same nli_oracle venv):
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.verifier_comparison.run_comparison
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from evaluation.core.io import load_jsonl, save_json
from evaluation.experiments.nli_oracle.model import NliOracleModel, MODEL_NAME as MODEL_A_NAME, MAX_LENGTH as MAX_LENGTH_A
from evaluation.experiments.verifier_comparison.metrics import (
    compute_metrics_block,
    deduplicate_by_template,
    template_consistency,
)
from evaluation.experiments.verifier_comparison.model_b import DebertaNliModel, MODEL_NAME as MODEL_B_NAME
from evaluation.experiments.verifier_comparison.verifier_c_llm import GeminiVerifier, MODEL_NAME as MODEL_C_NAME

BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark_v1.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/verifier_comparison_report.json"

A_RAW_TO_RELATION = {"entailment": "SUPPORT", "neutral": "NOT_ENOUGH_EVIDENCE", "contradiction": "CONTRADICT"}


def run() -> dict[str, Any]:
    examples = load_jsonl(BENCHMARK_PATH)
    print(f"Loaded {len(examples)} FROZEN benchmark examples from {BENCHMARK_PATH}")

    print(f"Loading model A ({MODEL_A_NAME})...")
    model_a = NliOracleModel()
    print(f"Loading model B ({MODEL_B_NAME})...")
    model_b = DebertaNliModel()
    print(f"Setting up verifier C ({MODEL_C_NAME})...")
    verifier_c = GeminiVerifier()
    print("All three verifiers ready. Running inference...")

    for i, ex in enumerate(examples, start=1):
        evidence = ex["evidence_text"]
        hypothesis = ex["hypothesis"]

        # ---- A: baseline NLI, both A1 (honest 3-class) and A2 (SUPPORT-only) ----
        t0 = time.perf_counter()
        res_a = model_a.predict(evidence, hypothesis)
        latency_a_ms = (time.perf_counter() - t0) * 1000
        a1_relation = A_RAW_TO_RELATION[res_a.raw_label]
        a2_relation = "SUPPORT" if res_a.raw_label == "entailment" else "NOT_ENOUGH_EVIDENCE"
        ex["A"] = {
            "model": MODEL_A_NAME,
            "raw_label": res_a.raw_label,
            "probs": res_a.probs,
            "token_count": res_a.token_count,
            "truncated": res_a.truncated,
            "latency_ms": round(latency_a_ms, 2),
            "A1_predicted_relation": a1_relation,
            "A2_predicted_relation": a2_relation,
        }

        # ---- B: alternative NLI ----
        t0 = time.perf_counter()
        res_b = model_b.predict(evidence, hypothesis)
        latency_b_ms = (time.perf_counter() - t0) * 1000
        ex["B"] = {
            "model": MODEL_B_NAME,
            "raw_label": res_b.raw_label,
            "probs": res_b.probs,
            "token_count": res_b.token_count,
            "truncated": res_b.truncated,
            "latency_ms": round(latency_b_ms, 2),
            "predicted_relation": res_b.predicted_relation,
        }

        # ---- C: LLM verifier ----
        res_c = verifier_c.predict(evidence, hypothesis)
        ex["C"] = {
            "model": MODEL_C_NAME,
            "predicted_relation": res_c.predicted_relation,
            "reason": res_c.reason,
            "raw_response_text": res_c.raw_response_text,
            "latency_ms": res_c.latency_ms,
            "prompt_tokens": res_c.prompt_tokens,
            "completion_tokens": res_c.completion_tokens,
            "total_tokens": res_c.total_tokens,
            "estimated_cost_usd": res_c.estimated_cost_usd,
            "parse_failure": res_c.parse_failure,
            "error": res_c.error,
        }

        if i % 10 == 0 or i == len(examples):
            print(f"  [{i}/{len(examples)}]")

    # ---------------- split by confidence ----------------
    primary = [ex for ex in examples if ex["annotation_confidence"] == "high"]
    secondary = [ex for ex in examples if ex["annotation_confidence"] == "medium"]
    print(f"PRIMARY (high-confidence): {len(primary)}, SECONDARY (medium-confidence): {len(secondary)}")

    def rows_for(exs: list[dict], pred_field: str) -> list[dict]:
        return [{"gold_relation": ex["gold_relation"], "predicted_relation": ex[pred_field]} for ex in exs]

    def a1_rows(exs):
        return [{"gold_relation": ex["gold_relation"], "predicted_relation": ex["A"]["A1_predicted_relation"]} for ex in exs]

    def a2_rows(exs):
        return [{"gold_relation": ex["gold_relation"], "predicted_relation": ex["A"]["A2_predicted_relation"]} for ex in exs]

    def b_rows(exs):
        return [{"gold_relation": ex["gold_relation"], "predicted_relation": ex["B"]["predicted_relation"]} for ex in exs]

    def c_rows(exs):
        return [{"gold_relation": ex["gold_relation"], "predicted_relation": ex["C"]["predicted_relation"]} for ex in exs]

    def full_rows_with_template(exs: list[dict], pred_fn) -> list[dict]:
        base = pred_fn(exs)
        for r, ex in zip(base, exs):
            r["template_group_id"] = ex.get("template_group_id")
        return base

    methods = {
        "A1_baseline_3class": a1_rows,
        "A2_baseline_support_only": a2_rows,
        "B_deberta_large": b_rows,
        "C_gemini_2_5_flash": c_rows,
    }

    primary_metrics = {}
    secondary_metrics = {}
    primary_template_dedup_metrics = {}
    primary_template_consistency = {}

    for name, fn in methods.items():
        p_rows = full_rows_with_template(primary, fn)
        s_rows = fn(secondary)

        primary_metrics[name] = compute_metrics_block(p_rows)
        secondary_metrics[name] = compute_metrics_block(s_rows) if secondary else None

        deduped = deduplicate_by_template(p_rows)
        primary_template_dedup_metrics[name] = compute_metrics_block(deduped)
        primary_template_consistency[name] = template_consistency(p_rows)

    n_parse_failures_c = sum(1 for ex in examples if ex["C"]["parse_failure"])

    report = {
        "run_metadata": {
            "benchmark": "held-out semantic challenge set v1 (FROZEN)",
            "benchmark_path": "evaluation/experiments/verifier_comparison/benchmark_v1.jsonl",
            "n_total": len(examples),
            "n_primary_high_confidence": len(primary),
            "n_secondary_medium_confidence": len(secondary),
            "methods": {
                "A1": {"model": MODEL_A_NAME, "mode": "honest 3-class"},
                "A2": {"model": MODEL_A_NAME, "mode": "SUPPORT-only (neutral+contradiction -> NOT_ENOUGH_EVIDENCE)"},
                "B": {"model": MODEL_B_NAME, "mode": "honest 3-class"},
                "C": {"model": MODEL_C_NAME, "mode": "structured JSON, temperature=0, no few-shot"},
            },
            "n_parse_failures_C": n_parse_failures_c,
        },
        "primary_metrics_high_confidence": primary_metrics,
        "secondary_metrics_medium_confidence": secondary_metrics,
        "primary_template_deduplicated_metrics": primary_template_dedup_metrics,
        "primary_template_consistency": primary_template_consistency,
        "examples_full_raw": examples,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== VERIFIER COMPARISON: PRIMARY (high-confidence) SUMMARY ===")
    for name, m in primary_metrics.items():
        rc = m["reliability_critical"]
        print(
            f"{name}: acc={m['accuracy']} macroF1={m['macro_f1']} "
            f"CONTRADICT->SUPPORT={rc['CONTRADICT_to_SUPPORT_rate']['rate']} "
            f"NEE->SUPPORT={rc['NEE_to_SUPPORT_rate']['rate']} "
            f"false_SUPPORT={rc['false_SUPPORT_rate']['rate']} "
            f"parse_failures={m['n_parse_failures']}"
        )
    print()
    print(f"Saved full report to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()

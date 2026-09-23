"""
Compares baseline end-to-end v1 (original 32-subclaim checkpoint, no
cleanup) vs architecture-cleanup end-to-end v2 (atomic quiet/FC1,
heading filter, dedup, deterministic routing), split into:

A. retrieval-only metrics (pre-verification)
B. verification metrics conditioned on retrieval already having
   succeeded (isolates verifier quality from retrieval quality)
C. final subclaim-level end-to-end metrics (both together)

Reads only already-saved reports - no new inference, no changes to
either baseline file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.core.io import save_json

V1_CANDIDATES_PATH = PROJECT_ROOT / "evaluation/experiments/nli_oracle/retrieval_candidates.jsonl"
V1_E2E_REPORT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_end_to_end_verification_report.json"
V2_RETRIEVAL_ONLY_PATH = PROJECT_ROOT / "evaluation/experiments/pipeline_cleanup_v2/retrieval_only_metrics_v2.json"
V2_E2E_REPORT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_end_to_end_verification_v2_report.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_end_to_end_v1_vs_v2_comparison.json"


def recall_at(rows: list[dict], k: int) -> dict:
    supported = [r for r in rows if r["gold_status"] == "SUPPORTED"]
    hits = 0
    for r in supported:
        gold = set(r["gold_support_chunk_ids"])
        top_ids = {c["chunk_id"] for c in r["top_k"][:k]}
        if gold & top_ids:
            hits += 1
    return {"k": k, "n_supported": len(supported), "hits": hits, "recall": round(hits / len(supported), 4) if supported else None}


def _is_verified_support(candidate: dict) -> bool:
    """
    v1's stored candidates use the 3-class relation label
    ("SUPPORT"/"CONTRADICT"/"RELEVANT_NEUTRAL", from NliResult.predicted_relation);
    v2's stored candidates use the binary verifier label directly
    ("VERIFIED_SUPPORT"/"NOT_VERIFIED"). Both mean "entailment fired"
    when the value is "SUPPORT" or "VERIFIED_SUPPORT" - normalize here
    rather than editing either stored report.
    """
    return candidate["predicted_relation"] in ("SUPPORT", "VERIFIED_SUPPORT")


def verification_given_retrieval(subclaims_with_candidates: list[dict], k: int) -> dict:
    """
    B-block, computed generically from raw per-example (top_k with
    raw_label/predicted_relation already attached) data - works for
    both v1's and v2's stored candidate format since both store
    "top_k" or "cleaned_candidates" with chunk_id + predicted_relation.
    """
    supported = [sc for sc in subclaims_with_candidates if sc["gold_status"] == "SUPPORTED"]
    not_supported = [sc for sc in subclaims_with_candidates if sc["gold_status"] == "NOT_SUPPORTED"]

    def candidates_for(sc, k):
        pool = sc.get("cleaned_candidates", sc.get("top_k"))
        return [c for c in pool if c["rank"] <= k]

    retrieval_succeeded = [
        sc for sc in supported
        if set(sc["gold_support_chunk_ids"]) & {c["chunk_id"] for c in candidates_for(sc, k)}
    ]
    verified_given_retrieval = sum(
        1 for sc in retrieval_succeeded
        if any(
            _is_verified_support(c) and c["chunk_id"] in sc["gold_support_chunk_ids"]
            for c in candidates_for(sc, k)
        )
    )
    fp_count = sum(
        1 for sc in not_supported
        if any(_is_verified_support(c) for c in candidates_for(sc, k))
    )
    return {
        "n_supported_with_gold_retrieved": len(retrieval_succeeded),
        "n_verified_given_retrieval_succeeded": verified_given_retrieval,
        "verification_recall_given_retrieval_success": (
            round(verified_given_retrieval / len(retrieval_succeeded), 4) if retrieval_succeeded else None
        ),
        "n_not_supported": len(not_supported),
        "n_false_support_among_not_supported": fp_count,
        "false_support_rate": round(fp_count / len(not_supported), 4) if not_supported else None,
    }


def run() -> dict:
    from evaluation.core.io import load_jsonl

    v1_candidates = load_jsonl(V1_CANDIDATES_PATH)
    v1_e2e = json.loads(V1_E2E_REPORT_PATH.read_text(encoding="utf-8"))
    v2_retrieval_only = json.loads(V2_RETRIEVAL_ONLY_PATH.read_text(encoding="utf-8"))
    v2_e2e = json.loads(V2_E2E_REPORT_PATH.read_text(encoding="utf-8"))

    # normalize v1's raw_label/predicted_relation onto its top_k in-place
    # (v1_e2e["subclaims_with_candidates"] already has predicted_relation
    # attached per candidate from the original run - reuse directly)
    v1_subclaims = v1_e2e["subclaims_with_candidates"]

    # ---- A. retrieval-only ----
    a_v1_r1 = recall_at(v1_candidates, 1)
    a_v1_r3 = recall_at(v1_candidates, 3)
    block_a = {
        "v1": {"n_subclaims_total": 32, "n_subclaims_supported": a_v1_r1["n_supported"], "recall_at_1": a_v1_r1, "recall_at_3": a_v1_r3},
        "v2": v2_retrieval_only,
    }

    # ---- B. verification given retrieval success ----
    block_b = {
        "v1": {"1": verification_given_retrieval(v1_subclaims, 1), "3": verification_given_retrieval(v1_subclaims, 3)},
        "v2": v2_e2e["verification_only_metrics_B"],
    }

    # ---- C. final end-to-end ----
    block_c = {
        "v1": {"1": v1_e2e["per_k"]["1"]["metrics"], "3": v1_e2e["per_k"]["3"]["metrics"]},
        "v2": {"1": v2_e2e["per_k"]["1"]["metrics"], "3": v2_e2e["per_k"]["3"]["metrics"]},
    }

    # ---- summary deltas ----
    def delta(v1_val, v2_val):
        if v1_val is None or v2_val is None:
            return None
        return round(v2_val - v1_val, 4)

    summary = {}
    for k in ("1", "3"):
        m1, m2 = block_c["v1"][k], block_c["v2"][k]
        summary[k] = {
            "precision": {"v1": m1["precision_verified_support"], "v2": m2["precision_verified_support"], "delta": delta(m1["precision_verified_support"], m2["precision_verified_support"])},
            "recall": {"v1": m1["recall_support_coverage"], "v2": m2["recall_support_coverage"], "delta": delta(m1["recall_support_coverage"], m2["recall_support_coverage"])},
            "f1": {"v1": m1["f1"], "v2": m2["f1"], "delta": delta(m1["f1"], m2["f1"])},
            "false_support_rate": {"v1": m1["false_support_rate"], "v2": m2["false_support_rate"], "delta": delta(m1["false_support_rate"], m2["false_support_rate"])},
            "fp_count": {"v1": m1["fp"], "v2": m2["fp"]},
            "fn_count": {"v1": m1["fn"], "v2": m2["fn"]},
            "n_subclaims": {"v1": m1["n"], "v2": m2["n"]},
        }

    error_types_v1 = {}
    for fn in v1_e2e["false_negatives_by_k"].get("3", []):
        error_types_v1[fn.get("failure_mode", "unknown")] = error_types_v1.get(fn.get("failure_mode", "unknown"), 0) + 1
    error_types_v2_fn = {}
    for fn in v2_e2e["false_negatives_by_k"].get("3", []):
        error_types_v2_fn[fn["error_type"]] = error_types_v2_fn.get(fn["error_type"], 0) + 1
    error_types_v2_fp = {}
    for fp in v2_e2e["false_positives_by_k"].get("3", []):
        error_types_v2_fp[fp["error_type"]] = error_types_v2_fp.get(fp["error_type"], 0) + 1

    nli_calls = {
        "v1": {"total_nli_calls": sum(len(sc["top_k"]) for sc in v1_subclaims)},
        "v2": v2_e2e["run_metadata"]["candidate_counts"],
    }

    report = {
        "A_retrieval_only": block_a,
        "B_verification_given_retrieval_success": block_b,
        "C_final_end_to_end": block_c,
        "summary_by_k": summary,
        "error_types_v1_fn_at_k3": error_types_v1,
        "error_types_v2_fn_at_k3": error_types_v2_fn,
        "error_types_v2_fp_at_k3": error_types_v2_fp,
        "nli_call_counts": nli_calls,
        "regression_checks_v2": [
            {"name": rc["name"], "passed": rc["passed"]} for rc in v2_e2e["regression_checks"]
        ],
    }

    save_json(OUTPUT_PATH, report)
    print(f"Saved comparison to {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))
    print()
    print("A. retrieval-only recall@1/3:")
    print("  v1:", block_a["v1"]["recall_at_1"], block_a["v1"]["recall_at_3"])
    print("  v2:", block_a["v2"]["recall_at_1"], block_a["v2"]["recall_at_3"])

    return report


if __name__ == "__main__":
    run()

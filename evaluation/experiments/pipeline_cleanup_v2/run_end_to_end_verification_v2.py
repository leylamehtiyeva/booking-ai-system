"""
Architecture-cleanup end-to-end experiment, phase 2/2: heading
filtering -> deduplication -> source-type routing (deterministic vs
NLI) -> subclaim-level OR aggregation, for K in {1, 3}.

Run with the isolated nli_oracle environment:
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.pipeline_cleanup_v2.run_end_to_end_verification_v2
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.core.io import load_jsonl, save_json
from evaluation.experiments.nli_oracle.model import MAX_LENGTH, MODEL_NAME, NliOracleModel
from evaluation.experiments.pipeline_cleanup_v2.heading_rules import is_heading
from evaluation.experiments.pipeline_cleanup_v2.deduplication import deduplicate_candidates
from evaluation.experiments.pipeline_cleanup_v2.deterministic_mapping import deterministic_support

CANDIDATES_PATH = Path(__file__).resolve().parent / "retrieval_candidates_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_end_to_end_verification_v2_report.json"

K_VALUES = (1, 3)


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else (0.0 if precision is not None and recall is not None else None)
    )
    false_support_rate = fp / (fp + tn) if (fp + tn) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_verified_support": round(precision, 4) if precision is not None else None,
        "recall_support_coverage": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "false_support_rate": round(false_support_rate, 4) if false_support_rate is not None else None,
        "false_negative_rate": round(false_negative_rate, 4) if false_negative_rate is not None else None,
        "accuracy": round((tp + tn) / n, 4) if n else None,
    }


def classify_fp_reason(cand: dict[str, Any]) -> str:
    if cand["verification_method"] == "deterministic":
        return "deterministic_mapping_failure"
    return "NLI_verification_failure"


def run() -> dict[str, Any]:
    subclaims = load_jsonl(CANDIDATES_PATH)
    print(f"Loaded {len(subclaims)} subclaims from {CANDIDATES_PATH}")

    print(f"Loading model {MODEL_NAME} (max_length={MAX_LENGTH})...")
    model = NliOracleModel()
    print("Model loaded.")

    n_before_cleanup = 0
    n_after_heading_filter = 0
    n_after_dedup = 0
    n_deterministic = 0
    n_nli = 0

    for sc in subclaims:
        n_before_cleanup += len(sc["top_k"])

        non_heading = [c for c in sc["top_k"] if not is_heading(c["path"])]
        n_after_heading_filter += len(non_heading)

        deduped = deduplicate_candidates(non_heading)
        n_after_dedup += len(deduped)

        for cand in deduped:
            if deterministic_support(sc["subclaim"], cand["source_type"], cand["text"]):
                cand["verification_method"] = "deterministic"
                cand["predicted_relation"] = "VERIFIED_SUPPORT"
                cand["raw_label"] = None
                cand["entailment_prob"] = None
                n_deterministic += 1
            else:
                result = model.predict(cand["text"], sc["hypothesis"])
                cand["verification_method"] = "nli"
                cand["raw_label"] = result.raw_label
                cand["predicted_relation"] = (
                    "VERIFIED_SUPPORT" if result.raw_label == "entailment" else "NOT_VERIFIED"
                )
                cand["entailment_prob"] = result.probs["entailment"]
                n_nli += 1

        sc["cleaned_candidates"] = deduped

    n_naive_nli_calls = n_before_cleanup  # if every original top-5 candidate had gone through NLI, uncleaned
    n_saved_by_heading_filter = n_before_cleanup - n_after_heading_filter
    n_saved_by_dedup = n_after_heading_filter - n_after_dedup
    n_saved_by_deterministic_routing = n_deterministic
    n_actual_nli_calls = n_nli

    print(
        f"candidates: before_cleanup={n_before_cleanup} after_heading_filter={n_after_heading_filter} "
        f"after_dedup={n_after_dedup} deterministic={n_deterministic} nli_calls={n_nli}"
    )

    # ---------------- per-K subclaim-level scoring ----------------
    per_k: dict[int, dict[str, Any]] = {}
    for k in K_VALUES:
        tp = fp = fn = tn = 0
        false_positives = []
        false_negatives = []
        rows = []

        for sc in subclaims:
            window = [c for c in sc["cleaned_candidates"] if c["rank"] <= k]
            predicted_status = (
                "VERIFIED_SUPPORT"
                if any(c["predicted_relation"] == "VERIFIED_SUPPORT" for c in window)
                else "NOT_VERIFIED"
            )
            gold_status = sc["gold_status"]

            if gold_status == "SUPPORTED" and predicted_status == "VERIFIED_SUPPORT":
                tp += 1
                outcome = "TP"
            elif gold_status == "NOT_SUPPORTED" and predicted_status == "VERIFIED_SUPPORT":
                fp += 1
                outcome = "FP"
            elif gold_status == "SUPPORTED" and predicted_status == "NOT_VERIFIED":
                fn += 1
                outcome = "FN"
            else:
                tn += 1
                outcome = "TN"

            rows.append(
                {"pair_id": sc["pair_id"], "subclaim": sc["subclaim"], "gold_status": gold_status,
                 "predicted_status": predicted_status, "outcome": outcome}
            )

            if outcome == "FP":
                for cand in window:
                    if cand["predicted_relation"] == "VERIFIED_SUPPORT":
                        false_positives.append(
                            {
                                "property_slug": sc["property_slug"],
                                "constraint_text": sc["constraint_text"],
                                "subclaim": sc["subclaim"],
                                "hypothesis": sc["hypothesis"],
                                "retrieval_rank": cand["rank"],
                                "evidence_text": cand["text"],
                                "embedding_similarity_score": cand["similarity_score"],
                                "nli_entailment_probability": cand["entailment_prob"],
                                "source_type": cand["source_type"],
                                "chunk_id": cand["chunk_id"],
                                "duplicate_chunk_ids": cand["duplicate_chunk_ids"],
                                "verification_method": cand["verification_method"],
                                "error_type": classify_fp_reason(cand),
                            }
                        )

            if outcome == "FN":
                gold_ids = set(sc["gold_support_chunk_ids"])
                window_ids = {c["chunk_id"] for c in window}
                full_top5_ids = {c["chunk_id"] for c in sc["top_k"]}
                cleaned_ids = {c["chunk_id"] for c in sc["cleaned_candidates"]}

                gold_in_window = gold_ids & window_ids
                gold_in_cleaned_not_window = (gold_ids & cleaned_ids) - window_ids
                gold_in_full_not_cleaned = (gold_ids & full_top5_ids) - cleaned_ids
                gold_anywhere = bool(gold_ids & full_top5_ids)

                if gold_in_window:
                    error_type = "NLI_verification_failure"
                    explanation = (
                        f"Gold SUPPORT chunk(s) {sorted(gold_in_window)} were retrieved, survived "
                        f"cleanup, and were within top-{k}, but verification (method="
                        f"{next(c['verification_method'] for c in window if c['chunk_id'] in gold_in_window)}) "
                        "did not confirm them."
                    )
                elif gold_in_cleaned_not_window:
                    error_type = "retrieval_rank_failure"
                    explanation = (
                        f"Gold SUPPORT chunk(s) {sorted(gold_in_cleaned_not_window)} survived cleanup "
                        f"but ranked below top-{k} - would be found at a larger K."
                    )
                elif gold_in_full_not_cleaned:
                    error_type = "heading/preprocessing_failure"
                    explanation = (
                        f"Gold SUPPORT chunk(s) {sorted(gold_in_full_not_cleaned)} were retrieved in the "
                        "full top-5, but were removed by heading-filtering or absorbed as a non-canonical "
                        "duplicate during deduplication."
                    )
                elif gold_anywhere:
                    error_type = "other"
                    explanation = "Gold chunk present somewhere but not accounted for by the above categories - needs manual review."
                else:
                    error_type = "retrieval_semantic_failure"
                    explanation = (
                        f"None of the gold SUPPORT chunk(s) {sorted(gold_ids)} appear anywhere in the "
                        "full top-5 retrieval at all."
                    )

                false_negatives.append(
                    {
                        "property_slug": sc["property_slug"],
                        "constraint_text": sc["constraint_text"],
                        "subclaim": sc["subclaim"],
                        "hypothesis": sc["hypothesis"],
                        "gold_support_chunk_ids": sorted(gold_ids),
                        "error_type": error_type,
                        "explanation": explanation,
                        "top_k_window": [
                            {
                                "rank": c["rank"], "chunk_id": c["chunk_id"], "text": c["text"],
                                "similarity_score": c["similarity_score"],
                                "verification_method": c["verification_method"],
                                "predicted_relation": c["predicted_relation"],
                                "entailment_prob": c["entailment_prob"],
                                "is_gold_support": c["chunk_id"] in gold_ids,
                            }
                            for c in window
                        ],
                        "full_top5_raw": [
                            {"rank": c["rank"], "chunk_id": c["chunk_id"], "text": c["text"],
                             "is_gold_support": c["chunk_id"] in gold_ids}
                            for c in sc["top_k"]
                        ],
                    }
                )

        metrics = binary_metrics(tp, fp, fn, tn)
        per_k[k] = {"metrics": metrics, "false_positives": false_positives, "false_negatives": false_negatives, "rows": rows}

    # ---------------- B: verification-only metrics (conditioned on retrieval success) ----------------
    verification_only = {}
    for k in K_VALUES:
        supported = [sc for sc in subclaims if sc["gold_status"] == "SUPPORTED"]
        retrieval_succeeded = [
            sc for sc in supported
            if set(sc["gold_support_chunk_ids"]) & {c["chunk_id"] for c in sc["cleaned_candidates"] if c["rank"] <= k}
        ]
        verified_given_retrieval = sum(
            1 for sc in retrieval_succeeded
            if any(
                c["predicted_relation"] == "VERIFIED_SUPPORT" and c["chunk_id"] in sc["gold_support_chunk_ids"]
                for c in sc["cleaned_candidates"] if c["rank"] <= k
            )
        )
        not_supported = [sc for sc in subclaims if sc["gold_status"] == "NOT_SUPPORTED"]
        fp_count = sum(
            1 for sc in not_supported
            if any(c["predicted_relation"] == "VERIFIED_SUPPORT" for c in sc["cleaned_candidates"] if c["rank"] <= k)
        )
        verification_only[k] = {
            "n_supported_with_gold_retrieved": len(retrieval_succeeded),
            "n_verified_given_retrieval_succeeded": verified_given_retrieval,
            "verification_recall_given_retrieval_success": (
                round(verified_given_retrieval / len(retrieval_succeeded), 4) if retrieval_succeeded else None
            ),
            "n_not_supported": len(not_supported),
            "n_false_support_among_not_supported": fp_count,
            "false_support_rate": round(fp_count / len(not_supported), 4) if not_supported else None,
        }

    # ---------------- deterministic-routing ablation ----------------
    det_tp_contribution = {}
    for k in K_VALUES:
        det_only_tp = sum(
            1 for sc in subclaims if sc["gold_status"] == "SUPPORTED" and any(
                c["verification_method"] == "deterministic" and c["chunk_id"] in sc["gold_support_chunk_ids"]
                for c in sc["cleaned_candidates"] if c["rank"] <= k
            )
        )
        det_tp_contribution[k] = det_only_tp

    # ---------------- regression checks ----------------
    regression_checks = _run_regression_checks(subclaims)

    report = {
        "run_metadata": {
            "model": MODEL_NAME,
            "checkpoint_schema": "v2/atomic (Q1/Q2/Q3 quiet, FC1 sentence-split)",
            "n_subclaims_total": len(subclaims),
            "candidate_counts": {
                "n_before_cleanup": n_before_cleanup,
                "n_after_heading_filter": n_after_heading_filter,
                "n_after_dedup": n_after_dedup,
                "n_deterministic_decisions": n_deterministic,
                "n_nli_calls": n_nli,
                "n_nli_calls_saved_by_heading_filter": n_saved_by_heading_filter,
                "n_nli_calls_saved_by_dedup": n_saved_by_dedup,
                "n_nli_calls_saved_by_deterministic_routing": n_saved_by_deterministic_routing,
                "n_naive_nli_calls_if_uncleaned": n_naive_nli_calls,
                "total_nli_calls_saved": n_naive_nli_calls - n_actual_nli_calls,
            },
        },
        "per_k": {str(k): {"metrics": per_k[k]["metrics"], "n_fp": len(per_k[k]["false_positives"]), "n_fn": len(per_k[k]["false_negatives"])} for k in K_VALUES},
        "verification_only_metrics_B": {str(k): verification_only[k] for k in K_VALUES},
        "deterministic_routing_tp_contribution": {str(k): det_tp_contribution[k] for k in K_VALUES},
        "false_positives_by_k": {str(k): per_k[k]["false_positives"] for k in K_VALUES},
        "false_negatives_by_k": {str(k): per_k[k]["false_negatives"] for k in K_VALUES},
        "subclaim_outcomes_by_k": {str(k): per_k[k]["rows"] for k in K_VALUES},
        "regression_checks": regression_checks,
        "subclaims_full": subclaims,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== ARCHITECTURE-CLEANUP END-TO-END v2: SUMMARY ===")
    for k in K_VALUES:
        m = per_k[k]["metrics"]
        print(
            f"K={k}: n={m['n']} TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
            f"precision={m['precision_verified_support']} recall={m['recall_support_coverage']} "
            f"F1={m['f1']} false_support_rate={m['false_support_rate']} accuracy={m['accuracy']}"
        )
    print()
    print("Regression checks:")
    for rc in regression_checks:
        status = "PASS" if rc["passed"] else ("N/A (not retrieved anywhere)" if rc["passed"] is None else "FAIL")
        print(f"  [{status}] {rc['name']}")
        for o in rc["occurrences"]:
            print(f"      {o['pair_id']}/{o['subclaim']} rank={o['rank']} status={o['status']}")
    print()
    print(f"Saved to: {OUTPUT_PATH}")

    return report


def _find_occurrences(subclaims: list[dict[str, Any]], text_contains: str) -> list[dict[str, Any]]:
    """
    All occurrences of a text fragment anywhere in the RAW top-5
    (before cleanup), across every subclaim, with what happened to
    each one (verified / not-verified / removed by heading-filter /
    absorbed by dedup / never retrieved elsewhere).
    """
    occurrences = []
    for sc in subclaims:
        cleaned_by_id = {c["chunk_id"]: c for c in sc.get("cleaned_candidates", [])}
        for raw in sc["top_k"]:
            if text_contains not in raw["text"]:
                continue
            cleaned = cleaned_by_id.get(raw["chunk_id"])
            if cleaned is not None:
                status = (
                    "VERIFIED_SUPPORT_DETERMINISTIC"
                    if cleaned["verification_method"] == "deterministic"
                    else cleaned["predicted_relation"]
                )
            else:
                status = "REMOVED_BY_HEADING_FILTER" if is_heading(raw["path"]) else "ABSORBED_BY_DEDUP"
            occurrences.append(
                {
                    "pair_id": sc["pair_id"],
                    "subclaim": sc["subclaim"],
                    "text": raw["text"],
                    "rank": raw["rank"],
                    "status": status,
                    "entailment_prob": cleaned.get("entailment_prob") if cleaned else None,
                }
            )
    return occurrences


def _run_regression_checks(subclaims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Each check searches ALL occurrences of the fragment across the
    full v2 candidate set (not one hardcoded location) - quiet is now
    3 subclaims, so a fragment can behave differently under Q1 vs Q2
    vs Q3, and that difference is itself the interesting result, not
    collapsed away.
    """
    SUPPORT_LIKE = {"VERIFIED_SUPPORT", "VERIFIED_SUPPORT_DETERMINISTIC"}
    NOT_SUPPORT_LIKE = {"NOT_VERIFIED", "REMOVED_BY_HEADING_FILTER", "ABSORBED_BY_DEDUP"}

    checks = [
        ("Non-smoking throughout -> quiet (expect NOT SUPPORT everywhere it appears)",
         "Non-smoking throughout", "not_support_everywhere", None),
        ("Continental breakfast included -> good breakfast quality (expect NOT SUPPORT everywhere)",
         "Continental breakfast included", "not_support_everywhere", None),
        ("Children and beds -> family layout (expect NOT SUPPORT everywhere - filtered as heading)",
         "Children and beds", "not_support_everywhere", None),
        ("Soundproofing -> Q1 soundproofing (expect SUPPORT, deterministic or NLI)",
         "Soundproofing", "support_in", "Q1"),
        ("Quiet street view -> Q2 explicitly quiet surroundings (expect SUPPORT)",
         "Quiet street view", "support_in", "Q2"),
        ("Children of any age are welcome -> FC1 (expect SUPPORT)",
         "Children of any age are welcome", "support_in", "FC1"),
        ("Free WiFi -> RW2a (expect deterministic SUPPORT)",
         "Free WiFi", "deterministic_support_in", "RW2a"),
        ("Desk -> RW1 (expect deterministic SUPPORT)",
         "Desk", "deterministic_support_in", "RW1"),
    ]

    results = []
    for name, fragment, rule, target_subclaim in checks:
        occurrences = _find_occurrences(subclaims, fragment)

        # ABSORBED_BY_DEDUP entries are duplicates that were correctly
        # folded into a canonical candidate elsewhere in the same
        # subclaim - they never get their own verification decision,
        # so they must not be judged as pass/fail in their own right.
        canonical_occurrences = [o for o in occurrences if o["status"] != "ABSORBED_BY_DEDUP"]

        if rule == "not_support_everywhere":
            passed = all(o["status"] in NOT_SUPPORT_LIKE for o in canonical_occurrences) if occurrences else None
        elif rule == "support_in":
            relevant = [o for o in canonical_occurrences if o["subclaim"] == target_subclaim]
            passed = bool(relevant) and all(o["status"] in SUPPORT_LIKE for o in relevant)
        elif rule == "deterministic_support_in":
            relevant = [o for o in canonical_occurrences if o["subclaim"] == target_subclaim]
            passed = bool(relevant) and all(o["status"] == "VERIFIED_SUPPORT_DETERMINISTIC" for o in relevant)
        else:
            passed = None

        results.append(
            {
                "name": name,
                "rule": rule,
                "passed": passed,
                "occurrences": occurrences,
            }
        )
    return results


if __name__ == "__main__":
    run()

"""
Strict structured-tag routing experiment (final routing tightening
before architecture freeze).

Reuses retrieval_candidates_v2.jsonl AS IS - no new retrieval, no
chunking/decomposition change, no embedding calls. Only the routing
decision changes from v2:

v2 rule:   structured source_type + exact alias match -> deterministic
           SUPPORT; otherwise -> NLI (even for unaliased structured tags)

v3 rule:   path matches Facility.name pattern (listing.facilities[N].name
           or rooms[N].facilities[M].name) -> deterministic ALWAYS:
               known alias  -> VERIFIED_SUPPORT
               no alias     -> NOT_VERIFIED, NLI never called
           anything else (including .overview, RoomOption.choices,
           and all non-facilities source types) -> NLI, unchanged

Heading filtering and deduplication are unchanged from v2.

Run with the isolated nli_oracle environment:
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.pipeline_cleanup_v2.run_strict_structured_routing
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
from evaluation.experiments.pipeline_cleanup_v2.deduplication import deduplicate_candidates
from evaluation.experiments.pipeline_cleanup_v2.deterministic_mapping import DETERMINISTIC_ALIASES
from evaluation.experiments.pipeline_cleanup_v2.heading_rules import is_heading
from evaluation.experiments.pipeline_cleanup_v2.structured_tag_rule import is_controlled_structured_tag

CANDIDATES_PATH = Path(__file__).resolve().parent / "retrieval_candidates_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_end_to_end_strict_routing_report.json"

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


def route(subclaim: str, cand: dict[str, Any]) -> str:
    if is_controlled_structured_tag(cand["path"]):
        return "deterministic"
    return "nli"


def run() -> dict[str, Any]:
    subclaims = load_jsonl(CANDIDATES_PATH)
    print(f"Loaded {len(subclaims)} subclaims from {CANDIDATES_PATH} (reused, not re-retrieved)")

    print(f"Loading model {MODEL_NAME} (max_length={MAX_LENGTH})...")
    model = NliOracleModel()
    print("Model loaded.")

    n_before_cleanup = 0
    n_after_heading_filter = 0
    n_after_dedup = 0
    n_deterministic = 0
    n_deterministic_support = 0
    n_deterministic_not_verified = 0
    n_nli = 0

    for sc in subclaims:
        n_before_cleanup += len(sc["top_k"])
        non_heading = [c for c in sc["top_k"] if not is_heading(c["path"])]
        n_after_heading_filter += len(non_heading)
        deduped = deduplicate_candidates(non_heading)
        n_after_dedup += len(deduped)

        for cand in deduped:
            method = route(sc["subclaim"], cand)
            if method == "deterministic":
                aliases = DETERMINISTIC_ALIASES.get(sc["subclaim"], set())
                is_alias = cand["text"].strip().casefold() in aliases
                cand["verification_method"] = "deterministic"
                cand["predicted_relation"] = "VERIFIED_SUPPORT" if is_alias else "NOT_VERIFIED"
                cand["raw_label"] = None
                cand["entailment_prob"] = None
                n_deterministic += 1
                if is_alias:
                    n_deterministic_support += 1
                else:
                    n_deterministic_not_verified += 1
            else:
                result = model.predict(cand["text"], sc["hypothesis"])
                cand["verification_method"] = "nli"
                cand["raw_label"] = result.raw_label
                cand["predicted_relation"] = "VERIFIED_SUPPORT" if result.raw_label == "entailment" else "NOT_VERIFIED"
                cand["entailment_prob"] = result.probs["entailment"]
                n_nli += 1

        sc["cleaned_candidates"] = deduped

    n_naive_nli_calls = n_before_cleanup
    print(
        f"candidates: before_cleanup={n_before_cleanup} after_heading_filter={n_after_heading_filter} "
        f"after_dedup={n_after_dedup} deterministic={n_deterministic} "
        f"(support={n_deterministic_support}, not_verified={n_deterministic_not_verified}) nli_calls={n_nli}"
    )

    per_k: dict[int, dict[str, Any]] = {}
    for k in K_VALUES:
        tp = fp = fn = tn = 0
        false_positives, false_negatives, rows = [], [], []

        for sc in subclaims:
            window = [c for c in sc["cleaned_candidates"] if c["rank"] <= k]
            predicted_status = "VERIFIED_SUPPORT" if any(c["predicted_relation"] == "VERIFIED_SUPPORT" for c in window) else "NOT_VERIFIED"
            gold_status = sc["gold_status"]

            if gold_status == "SUPPORTED" and predicted_status == "VERIFIED_SUPPORT":
                tp += 1; outcome = "TP"
            elif gold_status == "NOT_SUPPORTED" and predicted_status == "VERIFIED_SUPPORT":
                fp += 1; outcome = "FP"
            elif gold_status == "SUPPORTED" and predicted_status == "NOT_VERIFIED":
                fn += 1; outcome = "FN"
            else:
                tn += 1; outcome = "TN"

            rows.append({"pair_id": sc["pair_id"], "subclaim": sc["subclaim"], "gold_status": gold_status, "predicted_status": predicted_status, "outcome": outcome})

            if outcome == "FP":
                for cand in window:
                    if cand["predicted_relation"] == "VERIFIED_SUPPORT":
                        false_positives.append({
                            "property_slug": sc["property_slug"], "constraint_text": sc["constraint_text"],
                            "subclaim": sc["subclaim"], "hypothesis": sc["hypothesis"],
                            "retrieval_rank": cand["rank"], "evidence_text": cand["text"],
                            "embedding_similarity_score": cand["similarity_score"],
                            "nli_entailment_probability": cand["entailment_prob"],
                            "source_type": cand["source_type"], "chunk_id": cand["chunk_id"],
                            "verification_method": cand["verification_method"],
                        })

            if outcome == "FN":
                gold_ids = set(sc["gold_support_chunk_ids"])
                window_ids = {c["chunk_id"] for c in window}
                cleaned_ids = {c["chunk_id"] for c in sc["cleaned_candidates"]}
                full_ids = {c["chunk_id"] for c in sc["top_k"]}
                if gold_ids & window_ids:
                    error_type = "verification_failure"
                elif gold_ids & cleaned_ids:
                    error_type = "retrieval_rank_failure"
                elif gold_ids & full_ids:
                    error_type = "heading_or_dedup_failure"
                else:
                    error_type = "retrieval_semantic_failure"
                false_negatives.append({
                    "property_slug": sc["property_slug"], "constraint_text": sc["constraint_text"],
                    "subclaim": sc["subclaim"], "hypothesis": sc["hypothesis"],
                    "gold_support_chunk_ids": sorted(gold_ids), "error_type": error_type,
                    "window": [{"rank": c["rank"], "chunk_id": c["chunk_id"], "text": c["text"],
                                "verification_method": c["verification_method"], "predicted_relation": c["predicted_relation"]}
                               for c in window],
                })

        per_k[k] = {"metrics": binary_metrics(tp, fp, fn, tn), "false_positives": false_positives, "false_negatives": false_negatives, "rows": rows}

    # ---- A/B/C split: deterministic branch, NLI branch, combined ----
    branch_metrics = {}
    for k in K_VALUES:
        det_tp = det_fp = det_fn = det_tn = 0
        nli_tp = nli_fp = nli_fn = nli_tn = 0

        for sc in subclaims:
            window = [c for c in sc["cleaned_candidates"] if c["rank"] <= k]
            det_window = [c for c in window if c["verification_method"] == "deterministic"]
            nli_window = [c for c in window if c["verification_method"] == "nli"]

            gold_status = sc["gold_status"]

            det_status = "VERIFIED_SUPPORT" if any(c["predicted_relation"] == "VERIFIED_SUPPORT" for c in det_window) else "NOT_VERIFIED"
            nli_status = "VERIFIED_SUPPORT" if any(c["predicted_relation"] == "VERIFIED_SUPPORT" for c in nli_window) else "NOT_VERIFIED"

            if det_window:
                if gold_status == "SUPPORTED" and det_status == "VERIFIED_SUPPORT": det_tp += 1
                elif gold_status == "NOT_SUPPORTED" and det_status == "VERIFIED_SUPPORT": det_fp += 1
                elif gold_status == "SUPPORTED" and det_status == "NOT_VERIFIED": det_fn += 1
                else: det_tn += 1

            if nli_window:
                if gold_status == "SUPPORTED" and nli_status == "VERIFIED_SUPPORT": nli_tp += 1
                elif gold_status == "NOT_SUPPORTED" and nli_status == "VERIFIED_SUPPORT": nli_fp += 1
                elif gold_status == "SUPPORTED" and nli_status == "NOT_VERIFIED": nli_fn += 1
                else: nli_tn += 1

        branch_metrics[k] = {
            "A_deterministic_branch": binary_metrics(det_tp, det_fp, det_fn, det_tn),
            "B_nli_branch": binary_metrics(nli_tp, nli_fp, nli_fn, nli_tn),
            "C_combined": per_k[k]["metrics"],
        }

    regression_checks = _run_regression_checks(subclaims)

    report = {
        "run_metadata": {
            "model": MODEL_NAME,
            "routing_rule": "path == listing.facilities[N].name or rooms[N].facilities[M].name -> deterministic (alias or NOT_VERIFIED, NLI skipped); else -> NLI",
            "candidates_source": "reused retrieval_candidates_v2.jsonl - no new retrieval",
            "n_subclaims_total": len(subclaims),
            "candidate_counts": {
                "n_before_cleanup": n_before_cleanup,
                "n_after_heading_filter": n_after_heading_filter,
                "n_after_dedup": n_after_dedup,
                "n_deterministic_decisions": n_deterministic,
                "n_deterministic_verified_support": n_deterministic_support,
                "n_deterministic_not_verified_no_nli_call": n_deterministic_not_verified,
                "n_nli_calls": n_nli,
                "n_naive_nli_calls_if_uncleaned": n_naive_nli_calls,
                "total_nli_calls_saved": n_naive_nli_calls - n_nli,
            },
        },
        "per_k": {str(k): {"metrics": per_k[k]["metrics"], "n_fp": len(per_k[k]["false_positives"]), "n_fn": len(per_k[k]["false_negatives"])} for k in K_VALUES},
        "branch_metrics_ABC": {str(k): branch_metrics[k] for k in K_VALUES},
        "false_positives_by_k": {str(k): per_k[k]["false_positives"] for k in K_VALUES},
        "false_negatives_by_k": {str(k): per_k[k]["false_negatives"] for k in K_VALUES},
        "regression_checks": regression_checks,
        "subclaims_full": subclaims,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== STRICT STRUCTURED ROUTING: SUMMARY ===")
    for k in K_VALUES:
        m = per_k[k]["metrics"]
        print(f"K={k}: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} precision={m['precision_verified_support']} recall={m['recall_support_coverage']} F1={m['f1']} false_support_rate={m['false_support_rate']}")
    print()
    for rc in regression_checks:
        status = "PASS" if rc["passed"] else ("N/A" if rc["passed"] is None else "FAIL")
        print(f"[{status}] {rc['name']}")
        for o in rc["occurrences"]:
            print(f"    {o['pair_id']}/{o['subclaim']} rank={o['rank']} status={o['status']}")
    print()
    print(f"Saved to: {OUTPUT_PATH}")
    return report


def _find_occurrences(subclaims: list[dict[str, Any]], text_contains: str) -> list[dict[str, Any]]:
    occurrences = []
    for sc in subclaims:
        cleaned_by_id = {c["chunk_id"]: c for c in sc.get("cleaned_candidates", [])}
        for raw in sc["top_k"]:
            if text_contains not in raw["text"]:
                continue
            cleaned = cleaned_by_id.get(raw["chunk_id"])
            if cleaned is not None:
                status = cleaned["predicted_relation"]
                if cleaned["verification_method"] == "deterministic" and status == "VERIFIED_SUPPORT":
                    status = "VERIFIED_SUPPORT_DETERMINISTIC"
                elif cleaned["verification_method"] == "deterministic" and status == "NOT_VERIFIED":
                    status = "NOT_VERIFIED_DETERMINISTIC_NO_NLI_CALL"
            else:
                status = "REMOVED_BY_HEADING_FILTER" if is_heading(raw["path"]) else "ABSORBED_BY_DEDUP"
            occurrences.append({"pair_id": sc["pair_id"], "subclaim": sc["subclaim"], "text": raw["text"], "rank": raw["rank"], "status": status})
    return occurrences


def _run_regression_checks(subclaims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    SUPPORT_LIKE = {"VERIFIED_SUPPORT", "VERIFIED_SUPPORT_DETERMINISTIC"}
    NOT_SUPPORT_LIKE = {"NOT_VERIFIED", "NOT_VERIFIED_DETERMINISTIC_NO_NLI_CALL", "REMOVED_BY_HEADING_FILTER", "ABSORBED_BY_DEDUP"}

    checks = [
        ("Non-smoking throughout -> quiet (expect NOT SUPPORT everywhere)", "Non-smoking throughout", "not_support_everywhere", None),
        ("Continental breakfast included -> good breakfast quality (expect NOT SUPPORT everywhere)", "Continental breakfast included", "not_support_everywhere", None),
        ("Children and beds -> family layout (expect NOT SUPPORT everywhere)", "Children and beds", "not_support_everywhere", None),
        ("Soundproofing -> Q1 (expect SUPPORT)", "Soundproofing", "support_in", "Q1"),
        ("Quiet street view -> Q2 (expect SUPPORT)", "Quiet street view", "support_in", "Q2"),
        ("Children of any age are welcome -> FC1 (expect SUPPORT)", "Children of any age are welcome", "support_in", "FC1"),
        ("Free WiFi -> RW2a (expect deterministic SUPPORT)", "Free WiFi", "deterministic_support_in", "RW2a"),
        ("Desk -> RW1 (expect deterministic SUPPORT)", "Desk", "deterministic_support_in", "RW1"),
    ]

    results = []
    for name, fragment, rule, target in checks:
        occurrences = _find_occurrences(subclaims, fragment)
        canonical = [o for o in occurrences if o["status"] != "ABSORBED_BY_DEDUP"]

        if rule == "not_support_everywhere":
            passed = all(o["status"] in NOT_SUPPORT_LIKE for o in canonical) if occurrences else None
        elif rule == "support_in":
            relevant = [o for o in canonical if o["subclaim"] == target]
            passed = bool(relevant) and all(o["status"] in SUPPORT_LIKE for o in relevant)
        elif rule == "deterministic_support_in":
            relevant = [o for o in canonical if o["subclaim"] == target]
            passed = bool(relevant) and all(o["status"] == "VERIFIED_SUPPORT_DETERMINISTIC" for o in relevant)
        else:
            passed = None

        results.append({"name": name, "passed": passed, "occurrences": occurrences})
    return results


if __name__ == "__main__":
    run()

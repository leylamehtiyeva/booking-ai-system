"""
End-to-end SUPPORT verification, phase 2/2: NLI on retrieval top-k.

Loads retrieval_candidates.jsonl (built by build_retrieval_candidates.py,
using the unmodified retrieval implementation over the full evidence
pool - NOT oracle gold-only evidence). Runs the SAME NLI model used
throughout (GuardrailsAI/finetuned_nli_provenance, unchanged) on every
retrieved candidate fresh - oracle predictions are never reused here,
per instructions; the oracle dataset only supplies gold_status /
gold_support_chunk_ids for scoring.

    raw entailment    -> VERIFIED_SUPPORT
    raw neutral       -> NOT_VERIFIED
    raw contradiction -> NOT_VERIFIED   (still not treated as negative evidence)

Computes subclaim-level binary metrics for K in {1, 3, 5}, all 32
atomic subclaims included (no zero-evidence exclusion this time - see
build_retrieval_candidates.py / the run_metadata note below for why
that's the correct denominator here).

For every false positive: full detail (rank, score, NLI prob,
source_type). For every false negative: whether the gold SUPPORT
chunk appeared anywhere in top-5 at all (retrieval failure if not;
verification failure if it appeared but NLI never called entailment
on it, at any rank within top-5).

Run with the experiment's isolated environment:
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.run_end_to_end_verification
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

CANDIDATES_PATH = Path(__file__).resolve().parent / "retrieval_candidates.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_end_to_end_verification_report.json"

K_VALUES = (1, 3, 5)


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
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_verified_support": round(precision, 4) if precision is not None else None,
        "recall_support_coverage": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "false_support_rate": round(false_support_rate, 4) if false_support_rate is not None else None,
        "false_negative_rate": round(false_negative_rate, 4) if false_negative_rate is not None else None,
        "accuracy": round((tp + tn) / n, 4) if n else None,
    }


def run() -> dict[str, Any]:
    subclaims = load_jsonl(CANDIDATES_PATH)
    print(f"Loaded {len(subclaims)} subclaims with retrieval candidates from {CANDIDATES_PATH}")

    print(f"Loading model {MODEL_NAME} (max_length={MAX_LENGTH})...")
    model = NliOracleModel()
    print("Model loaded. Running fresh NLI inference on retrieval candidates (oracle predictions NOT reused)...")

    n_calls = 0
    for sc in subclaims:
        for cand in sc["top_k"]:
            result = model.predict(cand["text"], sc["hypothesis"])
            cand["raw_label"] = result.raw_label
            cand["predicted_relation"] = result.predicted_relation
            cand["probs"] = result.probs
            cand["entailment_prob"] = result.probs["entailment"]
            n_calls += 1
        print(f"  {sc['pair_id']} / {sc['subclaim']}: {len(sc['top_k'])} candidates scored")

    print(f"Total fresh NLI calls: {n_calls}")

    # ---------------- per-K subclaim-level scoring ----------------
    per_k: dict[int, dict[str, Any]] = {}
    for k in K_VALUES:
        tp = fp = fn = tn = 0
        rows_for_k = []
        false_positives = []
        false_negatives = []

        for sc in subclaims:
            top_k_candidates = sc["top_k"][:k]
            predicted_status = (
                "VERIFIED_SUPPORT"
                if any(c["raw_label"] == "entailment" for c in top_k_candidates)
                else "NOT_VERIFIED"
            )
            gold_status = sc["gold_status"]

            outcome = None
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

            rows_for_k.append(
                {
                    "pair_id": sc["pair_id"],
                    "subclaim": sc["subclaim"],
                    "gold_status": gold_status,
                    "predicted_status": predicted_status,
                    "outcome": outcome,
                }
            )

            if outcome == "FP":
                for cand in top_k_candidates:
                    if cand["raw_label"] == "entailment":
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
                            }
                        )

            if outcome == "FN":
                gold_ids = set(sc["gold_support_chunk_ids"])
                retrieved_ids_top_k = {c["chunk_id"] for c in top_k_candidates}
                retrieved_ids_full_top5 = {c["chunk_id"] for c in sc["top_k"]}
                gold_in_top_k = gold_ids & retrieved_ids_top_k
                gold_in_full_top5_not_in_top_k = (gold_ids & retrieved_ids_full_top5) - retrieved_ids_top_k

                if gold_in_top_k:
                    failure_mode = "verification_failure"
                    explanation = (
                        f"Gold SUPPORT chunk(s) {sorted(gold_in_top_k)} WERE retrieved within top-{k}, "
                        "but NLI did not call entailment on any of them (raw_label was neutral or "
                        "contradiction) - a verification failure, not a retrieval failure."
                    )
                elif gold_in_full_top5_not_in_top_k:
                    failure_mode = "retrieval_rank_failure"
                    explanation = (
                        f"Gold SUPPORT chunk(s) {sorted(gold_in_full_top5_not_in_top_k)} appear in the "
                        f"full top-5 retrieval, but ranked below top-{k} - a retrieval-ranking failure "
                        f"specific to this K, not a verification failure."
                    )
                else:
                    failure_mode = "retrieval_failure"
                    explanation = (
                        f"None of the gold SUPPORT chunk(s) {sorted(gold_ids)} appear anywhere in the "
                        "full top-5 retrieval at all - retrieval never surfaced the right evidence, "
                        "independent of K or NLI."
                    )

                false_negatives.append(
                    {
                        "property_slug": sc["property_slug"],
                        "constraint_text": sc["constraint_text"],
                        "subclaim": sc["subclaim"],
                        "hypothesis": sc["hypothesis"],
                        "gold_support_chunk_ids": sorted(gold_ids),
                        "failure_mode": failure_mode,
                        "explanation": explanation,
                        "top_k_retrieved": [
                            {
                                "rank": c["rank"],
                                "chunk_id": c["chunk_id"],
                                "text": c["text"],
                                "similarity_score": c["similarity_score"],
                                "source_type": c["source_type"],
                                "raw_label": c["raw_label"],
                                "entailment_prob": c["entailment_prob"],
                                "is_gold_support": c["chunk_id"] in gold_ids,
                            }
                            for c in top_k_candidates
                        ],
                        "full_top5_retrieved": [
                            {
                                "rank": c["rank"],
                                "chunk_id": c["chunk_id"],
                                "text": c["text"],
                                "similarity_score": c["similarity_score"],
                                "is_gold_support": c["chunk_id"] in gold_ids,
                            }
                            for c in sc["top_k"]
                        ],
                    }
                )

        metrics = binary_metrics(tp, fp, fn, tn)
        per_k[k] = {
            "metrics": metrics,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "rows": rows_for_k,
        }

    # ---------------- known no-evidence case tracking ----------------
    no_evidence_cases = {
        "good_breakfast": [sc for sc in subclaims if sc["constraint_text"] == "good breakfast"],
        "remote_work_reliability_RW2b": [sc for sc in subclaims if sc["subclaim"] == "RW2b"],
        "quiet_no_evidence_properties": [
            sc for sc in subclaims if sc["constraint_text"] == "quiet" and sc["gold_status"] == "NOT_SUPPORTED"
        ],
    }
    no_evidence_summary = {}
    for name, rows in no_evidence_cases.items():
        top1_fp = sum(
            1 for sc in rows if any(c["raw_label"] == "entailment" for c in sc["top_k"][:1])
        )
        top5_fp = sum(
            1 for sc in rows if any(c["raw_label"] == "entailment" for c in sc["top_k"][:5])
        )
        no_evidence_summary[name] = {
            "n_subclaims": len(rows),
            "false_positive_at_top1": top1_fp,
            "false_positive_at_top5": top5_fp,
            "subclaims": [
                {
                    "pair_id": sc["pair_id"],
                    "top1_text": sc["top_k"][0]["text"] if sc["top_k"] else None,
                    "top1_score": sc["top_k"][0]["similarity_score"] if sc["top_k"] else None,
                    "top1_raw_label": sc["top_k"][0]["raw_label"] if sc["top_k"] else None,
                    "top1_entailment_prob": sc["top_k"][0]["entailment_prob"] if sc["top_k"] else None,
                }
                for sc in rows
            ],
        }

    report = {
        "run_metadata": {
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "threshold_applied": False,
            "pipeline": "atomic subclaim -> semantic retrieval (unchanged embedding model) -> top-k candidates -> NLI binary verifier (fresh inference, no oracle reuse)",
            "checkpoint_source": "evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl (original 32-subclaim schema, NOT the Q1/Q2/Q3 or FC1-atomic NLI-only redesign)",
            "n_subclaims_total": len(subclaims),
            "n_subclaims_supported": sum(1 for sc in subclaims if sc["gold_status"] == "SUPPORTED"),
            "n_subclaims_not_supported": sum(1 for sc in subclaims if sc["gold_status"] == "NOT_SUPPORTED"),
            "denominator_note": (
                "32 total atomic subclaims (4 constraints x up to 3 subclaims x 4 properties, minus "
                "structural gaps) is the correct denominator for this experiment - see the standalone "
                "explanation delivered before this run. Earlier oracle reports used smaller denominators "
                "(45, 89, or 22+7=29 examples/groups) because they excluded zero-evidence subclaims by "
                "design (recall/verification-only framing) or replaced/dropped subclaims during the "
                "NLI-input-v2 redesign (quiet -> Q1/Q2/Q3, good_breakfast and RW2b dropped entirely, "
                "some FC2/FC3/RW1 zero-evidence slots dropped). This end-to-end experiment must use the "
                "full, unmodified 32-subclaim checkpoint because false positives on zero-evidence "
                "subclaims are exactly what is being measured."
            ),
            "n_fresh_nli_calls": n_calls,
        },
        "per_k": {
            str(k): {
                "metrics": per_k[k]["metrics"],
                "n_false_positives": len(per_k[k]["false_positives"]),
                "n_false_negatives": len(per_k[k]["false_negatives"]),
            }
            for k in K_VALUES
        },
        "false_positives_by_k": {str(k): per_k[k]["false_positives"] for k in K_VALUES},
        "false_negatives_by_k": {str(k): per_k[k]["false_negatives"] for k in K_VALUES},
        "subclaim_outcomes_by_k": {str(k): per_k[k]["rows"] for k in K_VALUES},
        "no_evidence_case_analysis": no_evidence_summary,
        "subclaims_with_candidates": subclaims,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== END-TO-END SUPPORT VERIFICATION: SUMMARY ===")
    for k in K_VALUES:
        m = per_k[k]["metrics"]
        print(
            f"K={k}: n={m['n']} TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
            f"precision={m['precision_verified_support']} recall={m['recall_support_coverage']} "
            f"F1={m['f1']} false_support_rate={m['false_support_rate']} accuracy={m['accuracy']}"
        )
    print()
    print(f"Saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()

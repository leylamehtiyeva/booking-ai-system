"""
Re-analyzes the ALREADY-COLLECTED v2 Oracle NLI predictions
(evaluation/outputs/nli_oracle_v2_report.json) as a binary SUPPORT
verifier, instead of a 3-class classifier.

No new model inference. No changes to production code, retrieval, or
the v1/v2 3-class baseline reports - this is a separate report.

Mapping (this stage only - NOT a claim that contradiction == neutral
in general, just that we are not yet using the contradiction signal):
    raw entailment    -> VERIFIED_SUPPORT
    raw neutral       -> NOT_VERIFIED
    raw contradiction -> NOT_VERIFIED

Subclaim-level is the primary unit of analysis (grouped by
(pair_id, subclaim) - e.g. ("fermaart_hotel__quiet", "Q1")). Chunk-
level binary metrics are computed for reference only.

Scope note: only (property, subclaim) groups that have >=1 evidence
chunk in the v2 oracle input are included. Subclaims with zero
evidence at all (a retrieval/data-availability gap, e.g.
apt_city_center's FC2) are excluded from the confusion matrix - the
verifier is never given a chance to be right or wrong there, so
counting it as a trivial true negative would conflate retrieval
coverage with verifier quality. Their count is reported separately as
a limitation.

source_type is derived by looking up each example's source_chunk_id
against evaluation.experiments.evidence_retrieval.chunking
(read-only import of already-existing code, not modified) applied to
the same property JSON snapshots already used to build v2 - not
invented, not guessed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.listing import ListingRaw
from evaluation.core.io import save_json
from evaluation.experiments.evidence_retrieval.chunking import build_evidence_chunks

V2_REPORT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_v2_report.json"
PROPERTIES_DIR = (
    PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/properties"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "evaluation/outputs/nli_oracle_support_verifier_v2_analysis.json"
)

STRUCTURED_SOURCE_TYPES = {"facilities", "room_facilities"}
FREE_TEXT_SOURCE_TYPES = {"description", "policies", "highlights", "title", "property_type", "other"}


def load_source_type_lookup() -> dict[str, str]:
    """
    property-agnostic chunk_id -> source_type, derived by re-running
    the existing, unmodified build_evidence_chunks() over the same
    property snapshots already used for v2. chunk_id already embeds
    the property id, so a flat dict is safe (no collisions).
    """
    lookup: dict[str, str] = {}
    for path in sorted(PROPERTIES_DIR.glob("*.json")):
        listing = ListingRaw.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for chunk in build_evidence_chunks(listing):
            lookup[chunk.chunk_id] = chunk.source_type
    return lookup


def source_bucket(source_type: str | None) -> str:
    if source_type in STRUCTURED_SOURCE_TYPES:
        return "structured_like"
    if source_type in FREE_TEXT_SOURCE_TYPES:
        return "free_text"
    return "unknown"


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else (0.0 if precision is not None and recall is not None else None)
    )
    false_support_rate = fp / (fp + tn) if (fp + tn) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_verified_support": round(precision, 4) if precision is not None else None,
        "recall_supported": round(recall, 4) if recall is not None else None,
        "support_coverage": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "false_support_rate": round(false_support_rate, 4) if false_support_rate is not None else None,
        "n": tp + fp + fn + tn,
    }


def run() -> dict[str, Any]:
    v2 = json.loads(V2_REPORT_PATH.read_text(encoding="utf-8"))
    examples = v2["examples"]

    source_type_lookup = load_source_type_lookup()
    unresolved_source_chunk_ids = set()

    for ex in examples:
        ex["binary_gold"] = "SUPPORT" if ex["gold_relation"] == "SUPPORT" else "NOT_SUPPORT"
        ex["binary_predicted"] = "VERIFIED_SUPPORT" if ex["raw_label"] == "entailment" else "NOT_VERIFIED"
        st = source_type_lookup.get(ex["source_chunk_id"])
        if st is None:
            unresolved_source_chunk_ids.add(ex["source_chunk_id"])
        ex["source_type"] = st
        ex["source_bucket"] = source_bucket(st)

    # ---------------- chunk-level binary reference metrics ----------------
    c_tp = sum(1 for e in examples if e["binary_gold"] == "SUPPORT" and e["binary_predicted"] == "VERIFIED_SUPPORT")
    c_fp = sum(1 for e in examples if e["binary_gold"] == "NOT_SUPPORT" and e["binary_predicted"] == "VERIFIED_SUPPORT")
    c_fn = sum(1 for e in examples if e["binary_gold"] == "SUPPORT" and e["binary_predicted"] == "NOT_VERIFIED")
    c_tn = sum(1 for e in examples if e["binary_gold"] == "NOT_SUPPORT" and e["binary_predicted"] == "NOT_VERIFIED")
    chunk_level_binary = binary_metrics(c_tp, c_fp, c_fn, c_tn)
    chunk_level_binary["accuracy"] = round((c_tp + c_tn) / len(examples), 4) if examples else None

    # ---------------- subclaim-level aggregation ----------------
    groups: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        key = (ex["pair_id"], ex["subclaim"])
        groups.setdefault(key, []).append(ex)

    subclaim_rows = []
    for (pair_id, subclaim), rows in sorted(groups.items()):
        gold_status = "SUPPORTED" if any(r["gold_relation"] == "SUPPORT" for r in rows) else "NOT_SUPPORTED"
        predicted_status = "VERIFIED_SUPPORT" if any(r["raw_label"] == "entailment" for r in rows) else "NOT_VERIFIED"
        subclaim_rows.append(
            {
                "pair_id": pair_id,
                "subclaim": subclaim,
                "property_slug": rows[0]["property_slug"],
                "constraint_text": rows[0]["constraint_text"],
                "hypothesis": rows[0]["hypothesis"],
                "n_chunks": len(rows),
                "gold_status": gold_status,
                "predicted_status": predicted_status,
                "correct": (gold_status == "SUPPORTED") == (predicted_status == "VERIFIED_SUPPORT"),
                "chunks": [
                    {
                        "chunk_id": r["chunk_id"],
                        "source_chunk_id": r["source_chunk_id"],
                        "premise": r["premise"],
                        "gold_relation": r["gold_relation"],
                        "raw_label": r["raw_label"],
                        "source_type": r["source_type"],
                        "source_bucket": r["source_bucket"],
                    }
                    for r in rows
                ],
            }
        )

    s_tp = sum(1 for r in subclaim_rows if r["gold_status"] == "SUPPORTED" and r["predicted_status"] == "VERIFIED_SUPPORT")
    s_fp = sum(1 for r in subclaim_rows if r["gold_status"] == "NOT_SUPPORTED" and r["predicted_status"] == "VERIFIED_SUPPORT")
    s_fn = sum(1 for r in subclaim_rows if r["gold_status"] == "SUPPORTED" and r["predicted_status"] == "NOT_VERIFIED")
    s_tn = sum(1 for r in subclaim_rows if r["gold_status"] == "NOT_SUPPORTED" and r["predicted_status"] == "NOT_VERIFIED")
    subclaim_level_binary = binary_metrics(s_tp, s_fp, s_fn, s_tn)
    subclaim_level_binary["accuracy"] = (
        round((s_tp + s_tn) / len(subclaim_rows), 4) if subclaim_rows else None
    )

    false_negatives = [
        r for r in subclaim_rows
        if r["gold_status"] == "SUPPORTED" and r["predicted_status"] == "NOT_VERIFIED"
    ]
    false_positives = [
        r for r in subclaim_rows
        if r["gold_status"] == "NOT_SUPPORTED" and r["predicted_status"] == "VERIFIED_SUPPORT"
    ]

    # ---------------- source/evidence-type breakdown (chunk-level) ----------------
    by_bucket: dict[str, list[dict]] = {}
    for ex in examples:
        by_bucket.setdefault(ex["source_bucket"], []).append(ex)

    source_breakdown = {}
    for bucket, rows in by_bucket.items():
        tp = sum(1 for r in rows if r["binary_gold"] == "SUPPORT" and r["binary_predicted"] == "VERIFIED_SUPPORT")
        fp = sum(1 for r in rows if r["binary_gold"] == "NOT_SUPPORT" and r["binary_predicted"] == "VERIFIED_SUPPORT")
        fn = sum(1 for r in rows if r["binary_gold"] == "SUPPORT" and r["binary_predicted"] == "NOT_VERIFIED")
        tn = sum(1 for r in rows if r["binary_gold"] == "NOT_SUPPORT" and r["binary_predicted"] == "NOT_VERIFIED")
        m = binary_metrics(tp, fp, fn, tn)
        m["source_types_included"] = sorted({r["source_type"] for r in rows if r["source_type"]})
        source_breakdown[bucket] = m

    # which source_type specifically drives NOT_VERIFIED on gold-SUPPORT chunks
    missed_support_by_source_type: dict[str, int] = {}
    for ex in examples:
        if ex["binary_gold"] == "SUPPORT" and ex["binary_predicted"] == "NOT_VERIFIED":
            st = ex["source_type"] or "unknown"
            missed_support_by_source_type[st] = missed_support_by_source_type.get(st, 0) + 1

    n_zero_evidence_subclaims = _count_zero_evidence_subclaims()

    report = {
        "run_metadata": {
            "purpose": "Re-evaluate the already-collected v2 Oracle NLI predictions as a binary "
            "SUPPORT verifier (entailment -> VERIFIED_SUPPORT, neutral/contradiction -> NOT_VERIFIED). "
            "No new model inference. Separate from the 3-class v1/v2 baseline reports.",
            "source_report": "evaluation/outputs/nli_oracle_v2_report.json",
            "n_chunk_level_examples": len(examples),
            "n_subclaim_groups": len(subclaim_rows),
            "n_zero_evidence_subclaims_excluded": n_zero_evidence_subclaims,
            "source_type_resolution": {
                "method": "looked up via evaluation.experiments.evidence_retrieval.chunking.build_evidence_chunks "
                "on the same property snapshots used to build v2 input - not invented",
                "unresolved_source_chunk_ids": sorted(unresolved_source_chunk_ids),
            },
        },
        "subclaim_level_binary_metrics": subclaim_level_binary,
        "chunk_level_binary_metrics_reference": chunk_level_binary,
        "false_negatives_subclaim_level": false_negatives,
        "false_positives_subclaim_level": false_positives,
        "source_breakdown_chunk_level": source_breakdown,
        "missed_support_by_source_type": missed_support_by_source_type,
        "subclaim_rows": subclaim_rows,
    }

    save_json(OUTPUT_PATH, report)

    print("=== SUPPORT VERIFIER ANALYSIS (v2 predictions, no new inference) ===")
    print(f"subclaim groups: {len(subclaim_rows)} (zero-evidence excluded: {n_zero_evidence_subclaims})")
    print("subclaim-level:", json.dumps(subclaim_level_binary, indent=2))
    print()
    print("chunk-level (reference):", json.dumps(chunk_level_binary, indent=2))
    print()
    print(f"false negatives (subclaim): {len(false_negatives)}")
    print(f"false positives (subclaim): {len(false_positives)}")
    print()
    print("source_breakdown_chunk_level:")
    for bucket, m in source_breakdown.items():
        print(f"  {bucket}: n={m['n']} recall_supported={m['recall_supported']} false_support_rate={m['false_support_rate']}")
    print()
    print(f"Saved to: {OUTPUT_PATH}")

    return report


def _count_zero_evidence_subclaims() -> int:
    """
    Counts zero-evidence (property, subclaim) slots only among the
    subclaim keys v2 actually carries over unchanged (FC2/FC3/RW1/
    RW2a) - these are exactly the gaps missing from the 89 v2
    examples. "quiet" (old "main") and "good_breakfast" are excluded
    from this count: v2 redesigned/replaced those subclaims entirely
    (Q1/Q2/Q3, FC1-atomic), so their old zero-evidence slots aren't
    part of v2's scope to begin with, not a comparable gap.
    """
    from evaluation.core.io import load_jsonl

    checkpoint = load_jsonl(
        PROJECT_ROOT / "evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl"
    )
    carried_over = {"FC2", "FC3", "RW1", "RW2a"}
    count = 0
    for pair in checkpoint:
        for subclaim_key, subclaim in pair["subclaims"].items():
            if subclaim_key in carried_over and not subclaim["evidence"]:
                count += 1
    return count


if __name__ == "__main__":
    run()

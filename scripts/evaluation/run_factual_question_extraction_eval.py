"""
Step 1 eval gate for the factual-question extractor (LISTING_QUESTION ->
ResultQuery). Mirrors evaluation/tasks/constraint_extraction/runner.py's
run_eval() exactly, swapping only the adapter (extract_factual_constraints_async
instead of route_intent_adk_async) - comparator.compare_case, metrics.compute_metrics
and the ConstraintEvalCase dataset schema are reused completely unmodified.

Makes real Gemini API calls - run only when you intend to spend the
corresponding quota/cost.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from app.logic.factual_question_extraction import (
    extract_factual_constraints_async,
)
from evaluation.core.io import load_jsonl, save_json
from evaluation.tasks.constraint_extraction.comparator import compare_case
from evaluation.tasks.constraint_extraction.dataset import ConstraintEvalCase
from evaluation.tasks.constraint_extraction.metrics import compute_metrics

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "constraint_extraction"
    / "factual_question_experimental_set.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "outputs"
    / "factual_question_extraction_report.json"
)


async def run_case(user_message: str) -> dict:
    result_query = await extract_factual_constraints_async(user_message)
    return result_query.model_dump(mode="json", exclude_none=True)


async def run_eval(dataset_path, output_path) -> dict:
    raw_cases = load_jsonl(dataset_path)
    cases = [ConstraintEvalCase.model_validate(x) for x in raw_cases]

    case_results = []

    for case in cases:
        try:
            result = await run_case(case.user_message)
            compared = compare_case(case, result)
            compared["raw_predicted_constraints"] = result.get("constraints")
            case_results.append(compared)
        except Exception as e:
            case_results.append(
                {
                    "case_id": case.case_id,
                    "group": case.group,
                    "user_message": case.user_message,
                    "error": str(e),
                    "constraint_extraction": {
                        "gold_count": len(case.expected_constraints),
                        "pred_count": 0,
                        "correct_count": 0,
                        "missed_constraints": [
                            c.normalized_text for c in case.expected_constraints
                        ],
                        "extra_constraints": [],
                        "exact_constraint_set_match": False,
                        "exact_full_case_match": False,
                    },
                    "matched_rows": [],
                }
            )

    metrics = compute_metrics(case_results)

    report = {
        "dataset_path": str(dataset_path),
        "n_cases": len(cases),
        "metrics": metrics,
        "cases": case_results,
    }

    save_json(output_path, report)
    return report


def print_report(report: dict) -> None:
    print("=" * 78)
    for case in report["cases"]:
        ce = case["constraint_extraction"]
        print(
            f"[{case['case_id']}] {case['group']} | {case['user_message']!r}"
        )
        if "error" in case:
            print(f"  ERROR: {case['error']}")
            continue
        print(f"  predicted (raw): {case.get('raw_predicted_constraints')}")
        print(
            f"  gold={ce['gold_count']} pred={ce['pred_count']} "
            f"correct={ce['correct_count']} missed={ce['missed_constraints']} "
            f"extra={ce['extra_constraints']} "
            f"exact_full_case_match={ce['exact_full_case_match']}"
        )
        for row in case["matched_rows"]:
            if not row["fully_correct_on_matched_constraint"]:
                print(f"    MISMATCH on '{row['normalized_text']}': {row}")

    print("=" * 78)
    print("METRICS")
    print(report["metrics"])


if __name__ == "__main__":
    report = asyncio.run(run_eval(DATASET_PATH, OUTPUT_PATH))
    print_report(report)

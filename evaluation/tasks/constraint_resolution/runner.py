from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tasks.constraint_resolution.adapter import run_case
from evaluation.tasks.constraint_resolution.comparator import compare_case
from evaluation.tasks.constraint_resolution.dataset import load_constraint_resolution_dataset
from evaluation.tasks.constraint_resolution.metrics import (
    compute_group_metrics,
    compute_metrics,
    get_errors,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation/datasets/constraint_resolution/constraint_resolution_golden_set_120_cleaned.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation/outputs/constraint_resolution_eval_report_new_ver3.json"
)

ERRORS_PATH = (
    PROJECT_ROOT
    / "evaluation/outputs/constraint_resolution_eval_runtime_errors.json"
)

REQUEST_DELAY_SECONDS = 10.0
MAX_RETRIES_PER_CASE = 1

MAX_CASES: int | None = None


def build_runtime_error_result(case: Any, error: Exception) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "difficulty": case.difficulty,
        "expected_decision": case.expected_decision,
        "predicted_decision": "ERROR",
        "decision_correct": False,
        "expected_explicit_negative": case.expected_explicit_negative,
        "predicted_explicit_negative": False,
        "explicit_negative_correct": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "constraint": case.constraint.model_dump(mode="json"),
        "listing_evidence": [
            evidence.model_dump(mode="json")
            for evidence in case.listing_evidence
        ],
        "raw_prediction": None,
    }
    
async def run_case_with_retry(case: Any) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES_PER_CASE + 1):
        try:
            return await run_case(case)

        except Exception as e:
            last_error = e
            message = str(e)

            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                sleep_seconds = 15 * attempt
                print(
                    f"429 for {case.case_id}. "
                    f"Retry {attempt}/{MAX_RETRIES_PER_CASE}. "
                    f"Sleeping {sleep_seconds}s"
                )
                await asyncio.sleep(sleep_seconds)
                continue

            raise

    raise RuntimeError(f"Failed after retries: {case.case_id}: {last_error}")


async def run_async() -> dict[str, Any]:
    cases = load_constraint_resolution_dataset(DATASET_PATH)
    
    # CASE_IDS = {"cr_012", "cr_017", "cr_032", "cr_037", "cr_045"}

    # cases = [
    #     case for case in cases
    #     if case.case_id in CASE_IDS
    # ]

    if MAX_CASES is not None:
        cases = cases[:MAX_CASES]

    results: list[dict[str, Any]] = []
    runtime_errors: list[dict[str, Any]] = []

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] Running {case.case_id} ({case.case_type})")

        try:
            prediction = await run_case_with_retry(case)

            result = compare_case(
                case=case,
                prediction=prediction,
            )

            results.append(result)

        except Exception as e:
            print(f"ERROR in {case.case_id}: {type(e).__name__}: {e}")

            error_result = build_runtime_error_result(case, e)
            results.append(error_result)
            runtime_errors.append(error_result)

        # Rate limiting for Gemini API
        await asyncio.sleep(REQUEST_DELAY_SECONDS)

        # Save partial progress after every case
        partial_report = {
            "completed_cases": len(results),
            "total_cases": len(cases),
            "runtime_error_count": len(runtime_errors),
            "cases": results,
            "runtime_errors": runtime_errors,
        }

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(partial_report, f, ensure_ascii=False, indent=2)

    valid_results = [
        result
        for result in results
        if result.get("predicted_decision") in {"YES", "NO", "UNCERTAIN"}
    ]

    overall_metrics = compute_metrics(valid_results) if valid_results else {}
    group_metrics = compute_group_metrics(valid_results) if valid_results else {}
    errors = get_errors(valid_results) if valid_results else []

    report = {
        "overall_metrics": overall_metrics,
        "group_metrics": group_metrics,
        "errors": errors,
        "runtime_errors": runtime_errors,
        "runtime_error_count": len(runtime_errors),
        "completed_cases": len(results),
        "valid_evaluated_cases": len(valid_results),
        "total_cases": len(cases),
        "cases": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(runtime_errors, f, ensure_ascii=False, indent=2)

    print("\n=== CONSTRAINT RESOLUTION EVAL ===")
    print(f"completed_cases: {len(results)}")
    print(f"valid_evaluated_cases: {len(valid_results)}")
    print(f"runtime_errors: {len(runtime_errors)}")

    if overall_metrics:
        print("\nOverall metrics:")
        print(f"total_cases: {overall_metrics['total_cases']}")
        print(f"accuracy: {overall_metrics['accuracy']}")
        print(f"decision_correct_count: {overall_metrics['decision_correct_count']}")

        print("\nCritical errors:")
        print(f"NO -> YES: {overall_metrics['critical_no_to_yes_count']}")
        print(f"YES -> NO: {overall_metrics['critical_yes_to_no_count']}")
        print(
            "YES -> UNCERTAIN:",
            overall_metrics["over_conservative_yes_to_uncertain_count"],
        )
        print(
            "UNCERTAIN -> YES:",
            overall_metrics["under_conservative_uncertain_to_yes_count"],
        )

        print("\nConfusion matrix:")
        print(json.dumps(overall_metrics["confusion_matrix"], indent=2))

        print("\nBy case_type:")
        for case_type, metrics in group_metrics.items():
            print(
                f"{case_type}: "
                f"accuracy={metrics['accuracy']} "
                f"total={metrics['total_cases']}"
            )

        print(f"\nDecision errors: {len(errors)}")

    print(f"\nSaved report to: {OUTPUT_PATH}")
    print(f"Saved runtime errors to: {ERRORS_PATH}")

    return report


def run() -> dict[str, Any]:
    return asyncio.run(run_async())


if __name__ == "__main__":
    run()
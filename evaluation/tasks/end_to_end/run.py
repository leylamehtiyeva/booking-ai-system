# evaluation/tasks/end_to_end/run.py

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evaluation.tasks.end_to_end.evaluator import evaluate_predictions
from evaluation.tasks.end_to_end.report import save_evaluation_report
from evaluation.tasks.end_to_end.runner import run_predictions


DATASET_PATH = Path(
    "evaluation/datasets/end_to_end/end2end_golden_set.jsonl"
)

PREDICTIONS_PATH = Path(
    "evaluation/outputs/end_to_end/predictions.jsonl"
)

REPORT_PATH = Path(
    "evaluation/outputs/end_to_end/report.json"
)

FAILURES_PATH = Path(
    "evaluation/outputs/end_to_end/failures.jsonl"
)


async def predict_command(limit: int | None) -> None:
    await run_predictions(
        dataset_path=DATASET_PATH,
        output_path=PREDICTIONS_PATH,
        limit=limit,
    )


def evaluate_command(limit: int | None = None) -> None:
    evaluation_result = evaluate_predictions(
        dataset_path=DATASET_PATH,
        predictions_path=PREDICTIONS_PATH,
        limit=limit,
    )

    save_evaluation_report(
        evaluation_result=evaluation_result,
        report_path=REPORT_PATH,
        failures_path=FAILURES_PATH,
    )

    metrics = evaluation_result["metrics"]

    print("End-to-end evaluation completed")
    print(f"Total: {metrics['total']}")
    print(f"OK: {metrics['ok_total']}")
    print(f"Errors: {metrics['error_total']}")
    print(f"Decision accuracy: {metrics['decision_accuracy']:.3f}")
    print(f"Top-1 selection accuracy: {metrics['top1_selection_accuracy']:.3f}")
    print(f"Top-K contains acceptable: {metrics['topk_contains_acceptable_rate']:.3f}")
    print(f"MRR: {metrics['mrr']:.3f}")
    print(f"Estimated total cost: ${metrics['cost']['total_estimated_usd']:.4f}")
    print(f"Mean runtime: {metrics['runtime_ms']['mean']:.0f} ms")
    print(f"Total LLM calls: {metrics['llm_calls']['total']}")
    print(f"Critical false YES: {metrics['critical_false_yes_count']}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Failures saved to: {FAILURES_PATH}")


async def all_command(limit: int | None) -> None:
    await predict_command(limit=limit)
    evaluate_command(limit=limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation for Booking AI."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run Booking AI on golden set and save predictions.",
    )
    predict_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N cases.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate saved predictions without rerunning Booking AI.",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Run predictions and then evaluation.",
    )
    all_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N cases.",
    )

    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        await predict_command(limit=args.limit)

    elif args.command == "evaluate":
        evaluate_command()

    elif args.command == "all":
        await all_command(limit=args.limit)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())
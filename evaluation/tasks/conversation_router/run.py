from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "conversation_router"
    / "router_eval_dev.jsonl"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "outputs"
    / "conversation_router"
)


def parse_profile_interval(
    value: str,
) -> tuple[str, float]:
    profile, separator, seconds_text = (
        value.partition("=")
    )

    if not separator or not profile:
        raise argparse.ArgumentTypeError(
            "Expected PROFILE=SECONDS"
        )

    try:
        seconds = float(seconds_text)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Interval must be a number"
        ) from exc

    if seconds < 0:
        raise argparse.ArgumentTypeError(
            "Interval cannot be negative"
        )

    return profile, seconds


async def predict_command(
    *,
    profiles: list[str],
    repeats: int,
    limit: int | None,
    delay_seconds: float,
    profile_min_interval_seconds: dict[str, float],
    run_name: str,
    overwrite: bool,
) -> None:
    from evaluation.tasks.conversation_router.runner import (
        run_predictions,
    )

    output_path = (
        OUTPUT_ROOT
        / run_name
        / "predictions.jsonl"
    )

    completed = await run_predictions(
        dataset_path=DATASET_PATH,
        output_path=output_path,
        profiles=profiles,
        repeats=repeats,
        limit=limit,
        delay_seconds=delay_seconds,
        profile_min_interval_seconds=(
            profile_min_interval_seconds
        ),
        overwrite=overwrite,
    )

    print(
        "Conversation router prediction run completed"
    )
    print(
        f"Predictions: {completed}"
    )
    print(
        f"Saved to: {output_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run conversation router evaluation "
            "predictions."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help=(
            "Run router predictions and save "
            "raw results."
        ),
    )

    predict_parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help=(
            "LLM profile names to compare."
        ),
    )

    predict_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Number of runs per case and profile."
        ),
    )

    predict_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Run only the first N dataset cases."
        ),
    )

    predict_parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Global delay between evaluation calls."
        ),
    )

    predict_parser.add_argument(
        "--profile-min-interval",
        action="append",
        type=parse_profile_interval,
        default=[],
        metavar="PROFILE=SECONDS",
        help=(
            "Minimum interval between calls for "
            "one profile. Can be specified "
            "multiple times."
        ),
    )

    predict_parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "Name of the output experiment "
            "directory."
        ),
    )

    predict_parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacing an existing "
            "predictions file."
        ),
    )
    
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help=(
            "Calculate metrics from saved router "
            "predictions."
        ),
    )

    evaluate_parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "Name of the experiment directory "
            "containing predictions.jsonl."
        ),
    )

    return parser


async def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        profile_min_interval_seconds = dict(
            args.profile_min_interval
        )

        await predict_command(
            profiles=args.profiles,
            repeats=args.repeats,
            limit=args.limit,
            delay_seconds=args.delay_seconds,
            profile_min_interval_seconds=(
                profile_min_interval_seconds
            ),
            run_name=args.run_name,
            overwrite=args.overwrite,
        )

        return
    
    if args.command == "evaluate":
        evaluate_command(
            run_name=args.run_name,
        )
        return

    raise ValueError(
        f"Unknown command: {args.command}"
    )
    
    
def evaluate_command(
    *,
    run_name: str,
) -> None:
    from evaluation.tasks.conversation_router.evaluator import (
        evaluate_predictions,
        load_router_predictions,
    )
    from evaluation.tasks.conversation_router.report import (
        save_evaluation_report,
    )

    run_directory = (
        OUTPUT_ROOT
        / run_name
    )

    predictions_path = (
        run_directory
        / "predictions.jsonl"
    )

    report_path = (
        run_directory
        / "report.json"
    )

    failures_path = (
        run_directory
        / "failures.jsonl"
    )

    predictions = load_router_predictions(
        predictions_path
    )

    evaluation_result = evaluate_predictions(
        predictions
    )

    save_evaluation_report(
        evaluation_result=evaluation_result,
        report_path=report_path,
        failures_path=failures_path,
    )

    print("Conversation router evaluation completed")
    print(
        f"Predictions: "
        f"{evaluation_result['total_prediction_rows']}"
    )
    print(
        f"Failures: "
        f"{evaluation_result['failures_count']}"
    )
    print(f"Report: {report_path}")
    print(f"Failures: {failures_path}")


if __name__ == "__main__":
    asyncio.run(main())
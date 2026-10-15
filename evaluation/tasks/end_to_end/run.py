# evaluation/tasks/end_to_end/run.py

from __future__ import annotations

# Must stay the first project import: loads .env before app settings are read.
from evaluation.tasks.end_to_end import env  # noqa: F401

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.tasks.end_to_end.contract import CONTRACT_VERSION
from evaluation.tasks.end_to_end.evaluator import evaluate_predictions
from evaluation.tasks.end_to_end.replay import replay_predictions
from evaluation.tasks.end_to_end.report import save_evaluation_report, save_json, save_jsonl
from evaluation.tasks.end_to_end.run_config import (
    LEGACY_RUN_METADATA,
    EvalRunConfig,
    current_git_commit,
)
from evaluation.tasks.end_to_end.runner import RUN_METADATA_FILENAME, run_predictions


DATASET_PATH = Path("evaluation/datasets/end_to_end/end2end_golden_set.jsonl")
DEFAULT_OUTPUT_DIR = Path("evaluation/outputs/end_to_end")


class OutputPaths:
    def __init__(self, output_dir: Path) -> None:
        self.dir = output_dir
        self.predictions = output_dir / "predictions.jsonl"
        self.report = output_dir / "report.json"
        self.failures = output_dir / "failures.jsonl"
        self.replay_dir = output_dir / "replay"
        self.run_metadata = output_dir / RUN_METADATA_FILENAME


def _evaluation_metadata() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluator_git_commit": current_git_commit(),
    }


def _load_run_metadata(paths: OutputPaths) -> dict[str, Any]:
    if not paths.run_metadata.exists():
        return LEGACY_RUN_METADATA
    return json.loads(paths.run_metadata.read_text(encoding="utf-8"))


def _print_summary(title: str, metrics: dict[str, Any]) -> None:
    decision = metrics["decision"]
    selection = metrics["selection"]
    legacy = metrics["legacy"]
    print(title)
    print(f"Cases: {metrics['total']} (ok {metrics['ok_total']}, errors {metrics['error_total']})")
    print(f"[{decision['contract_version']}] decision accuracy: {decision['accuracy']:.3f}")
    print(
        f"  critical false YES: {decision['critical_false_yes_count']}/{decision['rejection_total']}"
        f" | false NO: {decision['false_no_count']}/{decision['yes_total']}"
        f" | conservative: {decision['conservative']}"
    )
    print(f"  confusion: {decision['confusion_matrix']}")
    print(
        f"Top-1 selection: {selection['top1_accuracy']:.3f}"
        f" | top-k contains acceptable: {selection['topk_contains_acceptable_rate']:.3f}"
        f" | MRR: {selection['mrr']:.3f}"
    )
    print(
        f"[legacy {legacy['contract_version']}] decision accuracy: {legacy['decision_accuracy']:.3f}"
        f" | critical false YES: {legacy['critical_false_yes_count']}"
    )


async def predict_command(
    paths: OutputPaths, limit: int | None, config: EvalRunConfig, max_llm_calls: int | None
) -> None:
    await run_predictions(
        dataset_path=DATASET_PATH,
        output_path=paths.predictions,
        limit=limit,
        config=config,
        max_llm_calls=max_llm_calls,
    )


def evaluate_command(paths: OutputPaths, limit: int | None = None) -> None:
    evaluation_result = evaluate_predictions(
        dataset_path=DATASET_PATH,
        predictions_path=paths.predictions,
        limit=limit,
    )

    save_evaluation_report(
        evaluation_result=evaluation_result,
        report_path=paths.report,
        failures_path=paths.failures,
        run_metadata=_load_run_metadata(paths),
        evaluation_metadata={**_evaluation_metadata(), "case_limit": limit},
    )

    _print_summary("End-to-end evaluation", evaluation_result["metrics"])
    print(f"Report saved to: {paths.report}")
    print(f"Failures saved to: {paths.failures}")


async def replay_command(paths: OutputPaths, config: EvalRunConfig) -> None:
    result = await replay_predictions(
        dataset_path=DATASET_PATH,
        predictions_path=paths.predictions,
        config=config,
    )

    save_json(
        data={
            "intents_run_metadata": _load_run_metadata(paths),
            "evaluation_metadata": _evaluation_metadata(),
            "replay": result["replay"],
            "skipped_case_ids": result["skipped_case_ids"],
            "metrics": result["metrics"],
        },
        path=paths.replay_dir / "report.json",
    )
    save_jsonl(rows=result["rows"], path=paths.replay_dir / "rows.jsonl")

    _print_summary(
        "Replay of saved intents through the current backend (no LLM, textual fallback off)",
        result["metrics"],
    )
    print(f"Skipped (no saved intent): {result['skipped_case_ids']}")
    print(f"Replay saved to: {paths.replay_dir}")


async def all_command(
    paths: OutputPaths, limit: int | None, config: EvalRunConfig, max_llm_calls: int | None
) -> None:
    await predict_command(paths, limit=limit, config=config, max_llm_calls=max_llm_calls)
    evaluate_command(paths, limit=limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation for Booking AI."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where predictions, metadata and reports live (default: {DEFAULT_OUTPUT_DIR}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_run_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--top-n",
            type=int,
            default=None,
            help="Results shown per search (default: the production value).",
        )

    def add_budget_option(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--max-llm-calls",
            type=int,
            default=None,
            help="Spend guard: stop starting new cases once this many LLM calls were made.",
        )

    predict_parser = subparsers.add_parser(
        "predict",
        help="PAID: run Booking AI (LLM calls) on the golden set and save predictions.",
    )
    predict_parser.add_argument("--limit", type=int, default=None, help="Run only first N cases.")
    add_run_options(predict_parser)
    add_budget_option(predict_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Free: score saved predictions with the current decision contract.",
    )
    evaluate_parser.add_argument("--limit", type=int, default=None, help="Score only the first N cases.")

    replay_parser = subparsers.add_parser(
        "replay",
        help="Free: re-run saved intents through the current backend without any LLM.",
    )
    add_run_options(replay_parser)

    all_parser = subparsers.add_parser("all", help="PAID: predict, then evaluate.")
    all_parser.add_argument("--limit", type=int, default=None, help="Run only first N cases.")
    add_run_options(all_parser)
    add_budget_option(all_parser)

    return parser


def _config_from(args: argparse.Namespace) -> EvalRunConfig:
    config = EvalRunConfig()
    if getattr(args, "top_n", None) is not None:
        config = config.model_copy(update={"top_n": args.top_n})
    return config


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paths = OutputPaths(args.output_dir)

    if args.command == "predict":
        await predict_command(paths, limit=args.limit, config=_config_from(args), max_llm_calls=args.max_llm_calls)

    elif args.command == "evaluate":
        evaluate_command(paths, limit=args.limit)

    elif args.command == "replay":
        await replay_command(paths, _config_from(args))

    elif args.command == "all":
        await all_command(paths, limit=args.limit, config=_config_from(args), max_llm_calls=args.max_llm_calls)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())

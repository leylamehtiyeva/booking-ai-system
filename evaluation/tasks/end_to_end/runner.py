# evaluation/tasks/end_to_end/runner.py

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.tasks.end_to_end.adapter import EvalTurnFailure, run_case
from evaluation.tasks.end_to_end.dataset import load_end_to_end_dataset
from evaluation.tasks.end_to_end.report import save_json
from evaluation.tasks.end_to_end.run_config import EvalRunConfig, build_run_metadata

RUN_METADATA_FILENAME = "run_metadata.json"

# Typed turn failures caused by the provider, not by model quality.
PROVIDER_FAILURE_CODES = frozenset({"rate_limited", "timeout", "provider_unavailable"})


class _ExecutionStats:
    """Aggregates per-case telemetry into the run_metadata execution summary."""

    def __init__(self) -> None:
        self.cases = 0
        self.ok_cases = 0
        self.error_cases = 0
        self.provider_failures: Counter[str] = Counter()
        self.llm_calls = 0
        self.llm_calls_by_step: Counter[str] = Counter()
        self.failed_llm_attempts: Counter[str] = Counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = 0.0
        self.runtimes_ms: list[int] = []

    def add(self, *, runtime_ms: int, telemetry: dict[str, Any] | None, ok: bool) -> None:
        self.cases += 1
        self.ok_cases += ok
        self.error_cases += not ok
        self.runtimes_ms.append(runtime_ms)

        telemetry = telemetry or {}
        self.estimated_cost_usd += float((telemetry.get("cost") or {}).get("estimated_total_usd") or 0.0)
        for call in (telemetry.get("llm") or {}).get("calls") or []:
            self.llm_calls += 1
            self.llm_calls_by_step[call.get("step")] += 1
            self.prompt_tokens += call.get("prompt_tokens") or 0
            self.completion_tokens += call.get("completion_tokens") or 0
            if not call.get("success", True):
                self.failed_llm_attempts[f"{call.get('step')}:{call.get('error')}"] += 1

    def summary(self) -> dict[str, Any]:
        runtimes = sorted(self.runtimes_ms)

        def percentile(q: float) -> int | None:
            if not runtimes:
                return None
            return runtimes[min(len(runtimes) - 1, int(round(q * (len(runtimes) - 1))))]

        return {
            "cases_ok": self.ok_cases,
            "cases_error": self.error_cases,
            "provider_failure_cases": dict(self.provider_failures),
            "llm_calls_used": self.llm_calls,
            "llm_calls_by_step": dict(self.llm_calls_by_step),
            "failed_llm_attempts": dict(self.failed_llm_attempts),
            "tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "note": "From turn telemetry; several stages estimate tokens from text and "
                "omit the system instruction, so prompt tokens are undercounted.",
            },
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "case_latency_ms": {
                "mean": round(statistics.mean(runtimes)) if runtimes else None,
                "p50": percentile(0.5),
                "p90": percentile(0.9),
                "max": runtimes[-1] if runtimes else None,
            },
        }


async def run_predictions(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    config: EvalRunConfig | None = None,
    max_llm_calls: int | None = None,
    max_consecutive_failures: int | None = None,
) -> None:
    """
    Two guards, both checked between cases (partial results and metadata
    are always kept):

    - max_llm_calls - spend guard: stop once the LLM calls recorded so far
      (each turn's telemetry, failed attempts included) reach it. The last
      case may overshoot it by its own calls.
    - max_consecutive_failures - provider-outage guard: stop after this many
      cases in a row ended in a provider failure (rate limit, timeout,
      provider unavailable). Quality errors never count.

    A stopped run, or one with any provider failure, is marked
    valid_as_baseline = false.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    config = config or EvalRunConfig()
    metadata_path = output_path.parent / RUN_METADATA_FILENAME

    output_path.parent.mkdir(parents=True, exist_ok=True)

    limits = {"max_llm_calls": max_llm_calls, "max_consecutive_failures": max_consecutive_failures}
    metadata = build_run_metadata(config=config, dataset_path=dataset_path, case_limit=limit)
    metadata["execution"] = {**limits, "status": "running"}
    # Written before any case runs, so even a partial run says what it was.
    save_json(data=metadata, path=metadata_path)

    cases = load_end_to_end_dataset(dataset_path)
    if limit is not None:
        cases = cases[:limit]

    stats = _ExecutionStats()
    consecutive_provider_failures = 0
    stopped_reason = None
    started = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as f:
        for case in tqdm(cases, desc="Running end-to-end predictions"):
            if max_llm_calls is not None and stats.llm_calls >= max_llm_calls:
                stopped_reason = f"max_llm_calls reached ({stats.llm_calls} >= {max_llm_calls})"
                break

            if (
                max_consecutive_failures is not None
                and consecutive_provider_failures >= max_consecutive_failures
            ):
                stopped_reason = (
                    f"{consecutive_provider_failures} consecutive provider failures "
                    f"(limit {max_consecutive_failures})"
                )
                break

            start_time = time.perf_counter()

            try:
                prediction = await run_case(case, config=config)
                runtime_ms = int((time.perf_counter() - start_time) * 1000)
                stats.add(runtime_ms=runtime_ms, telemetry=prediction.raw_response.get("telemetry"), ok=True)
                consecutive_provider_failures = 0

                record: dict[str, Any] = {
                    "case_id": case.case_id,
                    "status": "ok",
                    "runtime_ms": runtime_ms,
                    "prediction": prediction.model_dump(mode="json"),
                }

            except EvalTurnFailure as e:
                runtime_ms = int((time.perf_counter() - start_time) * 1000)
                stats.add(runtime_ms=runtime_ms, telemetry=e.telemetry, ok=False)

                code = e.failure.get("code")
                if code in PROVIDER_FAILURE_CODES:
                    stats.provider_failures[f"{e.failure.get('stage')}:{code}"] += 1
                    consecutive_provider_failures += 1
                else:
                    consecutive_provider_failures = 0

                record = {
                    "case_id": case.case_id,
                    "status": "error",
                    "runtime_ms": runtime_ms,
                    "error": {
                        "type": "TurnFailure",
                        "message": str(e),
                        "failure": e.failure,
                    },
                }

            except Exception as e:
                # Unexpected error in the backend: recorded (type + message)
                # so the run continues and the report counts it.
                runtime_ms = int((time.perf_counter() - start_time) * 1000)
                stats.add(runtime_ms=runtime_ms, telemetry=None, ok=False)
                consecutive_provider_failures = 0

                record = {
                    "case_id": case.case_id,
                    "status": "error",
                    "runtime_ms": runtime_ms,
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    status = "stopped" if stopped_reason else "completed"
    metadata["execution"] = {
        **limits,
        "status": status,
        "stopped_reason": stopped_reason,
        "valid_as_baseline": status == "completed" and not stats.provider_failures,
        "cases_completed": stats.cases,
        "cases_planned": len(cases),
        "wall_clock_seconds": round(time.perf_counter() - started, 1),
        **stats.summary(),
    }
    save_json(data=metadata, path=metadata_path)

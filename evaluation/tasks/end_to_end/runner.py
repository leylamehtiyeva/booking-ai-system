# evaluation/tasks/end_to_end/runner.py

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.tasks.end_to_end.adapter import EvalTurnFailure, run_case
from evaluation.tasks.end_to_end.dataset import load_end_to_end_dataset
from evaluation.tasks.end_to_end.report import save_json
from evaluation.tasks.end_to_end.run_config import EvalRunConfig, build_run_metadata

RUN_METADATA_FILENAME = "run_metadata.json"


async def run_predictions(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    config: EvalRunConfig | None = None,
    max_llm_calls: int | None = None,
) -> None:
    """
    max_llm_calls is a spend guard: before each case the LLM calls recorded
    so far (from each turn's telemetry, failed attempts included) are
    compared with it and the run stops once it is reached. It is checked
    between cases, so the last case may overshoot it by its own calls.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    config = config or EvalRunConfig()
    metadata_path = output_path.parent / RUN_METADATA_FILENAME

    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_run_metadata(config=config, dataset_path=dataset_path, case_limit=limit)
    metadata["execution"] = {"max_llm_calls": max_llm_calls, "status": "running"}
    # Written before any case runs, so even a partial run says what it was.
    save_json(data=metadata, path=metadata_path)

    llm_calls_used = 0
    completed = 0
    stopped_reason = None

    cases = load_end_to_end_dataset(dataset_path)

    if limit is not None:
        cases = cases[:limit]

    with open(output_path, "w", encoding="utf-8") as f:
        for case in tqdm(cases, desc="Running end-to-end predictions"):
            if max_llm_calls is not None and llm_calls_used >= max_llm_calls:
                stopped_reason = f"max_llm_calls reached ({llm_calls_used} >= {max_llm_calls})"
                break

            start_time = time.perf_counter()

            try:
                prediction = await run_case(case, config=config)
                runtime_ms = int((time.perf_counter() - start_time) * 1000)
                llm_calls_used += _llm_calls(prediction.raw_response)

                record: dict[str, Any] = {
                    "case_id": case.case_id,
                    "status": "ok",
                    "runtime_ms": runtime_ms,
                    "prediction": prediction.model_dump(mode="json"),
                }

            except EvalTurnFailure as e:
                runtime_ms = int((time.perf_counter() - start_time) * 1000)
                llm_calls_used += e.llm_calls

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
            completed += 1

    metadata["execution"] = {
        "max_llm_calls": max_llm_calls,
        "status": "stopped" if stopped_reason else "completed",
        "stopped_reason": stopped_reason,
        "cases_completed": completed,
        "cases_planned": len(cases),
        "llm_calls_used": llm_calls_used,
    }
    save_json(data=metadata, path=metadata_path)


def _llm_calls(response: dict[str, Any]) -> int:
    telemetry = response.get("telemetry") or {}
    return int((telemetry.get("llm") or {}).get("calls_count") or 0)
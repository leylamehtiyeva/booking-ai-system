# evaluation/tasks/end_to_end/runner.py

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evaluation.tasks.end_to_end.adapter import run_case
from evaluation.tasks.end_to_end.dataset import load_end_to_end_dataset


async def run_predictions(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> None:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = load_end_to_end_dataset(dataset_path)

    if limit is not None:
        cases = cases[:limit]

    with open(output_path, "w", encoding="utf-8") as f:
        for case in tqdm(cases, desc="Running end-to-end predictions"):
            start_time = time.perf_counter()

            try:
                prediction = await run_case(case)
                runtime_ms = int((time.perf_counter() - start_time) * 1000)

                record: dict[str, Any] = {
                    "case_id": case.case_id,
                    "status": "ok",
                    "runtime_ms": runtime_ms,
                    "prediction": prediction.model_dump(mode="json"),
                }

            except Exception as e:
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
from __future__ import annotations

import json
from pathlib import Path

from evaluation.tasks.conversation_router.runner import (
    RouterEvalPrediction,
)


def save_evaluation_report(
    *,
    evaluation_result: dict,
    report_path: str | Path,
    failures_path: str | Path,
) -> None:
    report_path = Path(report_path)
    failures_path = Path(failures_path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        key: value
        for key, value in evaluation_result.items()
        if key != "failures"
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = evaluation_result["failures"]

    with failures_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for prediction in failures:
            if isinstance(
                prediction,
                RouterEvalPrediction,
            ):
                row = prediction.model_dump(
                    mode="json"
                )
            else:
                row = prediction

            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
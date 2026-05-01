from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TELEMETRY_DIR = PROJECT_ROOT / "logs" / "telemetry"


def _today_log_path(now: datetime) -> Path:
    date_str = now.strftime("%Y-%m-%d")
    return TELEMETRY_DIR / f"telemetry_{date_str}.jsonl"


def _next_attempt_number(log_path: Path) -> int:
    if not log_path.exists():
        return 1

    with log_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) + 1


def save_telemetry_record(
    *,
    telemetry: dict[str, Any],
    user_message: str,
    source: str,
    top_n: int,
    max_items: int,
    result_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(ZoneInfo("Asia/Baku"))
    log_path = _today_log_path(now)
    attempt_number = _next_attempt_number(log_path)

    record = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timestamp": now.isoformat(),
        "attempt_number": attempt_number,
        "source": source,
        "top_n": top_n,
        "max_items": max_items,
        "user_message": user_message,
        "result_summary": result_summary or {},
        "telemetry": telemetry,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return {
        "log_file": str(log_path),
        "attempt_number": attempt_number,
        "timestamp": record["timestamp"],
    }
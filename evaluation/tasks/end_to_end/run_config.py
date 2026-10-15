"""
Run configuration and metadata for the end-to-end evaluation, so a report
says exactly which backend, models, dataset and settings produced it.

Defaults mirror the production turn (app.schemas.turn.TurnConfig): the
evaluation measures what a user would actually be shown.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config.llm import get_gemini_model, get_llm_profile
from app.config.settings import ROUTER_LLM_PROFILE
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.turn import TurnConfig
from evaluation.core.reference_date import EVAL_REFERENCE_DATE
from evaluation.tasks.end_to_end.contract import CONTRACT_VERSION

_PRODUCTION = TurnConfig()


class EvalRunConfig(BaseModel):
    source: str = _PRODUCTION.source
    top_n: int = _PRODUCTION.top_n
    max_items: int = _PRODUCTION.max_items
    fallback_top_k: int = _PRODUCTION.fallback_top_k

    def fallback_policy(self) -> FallbackPolicy:
        # Same construction as app.services.conversation_service.process_turn.
        return FallbackPolicy(enabled=True, top_k=self.fallback_top_k)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def current_git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run_metadata(
    *,
    config: EvalRunConfig,
    dataset_path: str | Path,
    case_limit: int | None,
) -> dict[str, Any]:
    fallback_policy = config.fallback_policy()
    status = _git("status", "--porcelain", "--untracked-files=no")

    return {
        "contract_version": CONTRACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": current_git_commit(),
            "dirty": bool(status) if status is not None else None,
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(Path(dataset_path)),
            "case_limit": case_limit,
        },
        "reference_date": EVAL_REFERENCE_DATE.isoformat(),
        "run_config": {
            **config.model_dump(),
            "fallback_policy": fallback_policy.model_dump(mode="json"),
        },
        "models": {
            "conversation_router_profile": ROUTER_LLM_PROFILE,
            "conversation_router": get_llm_profile(ROUTER_LLM_PROFILE).model,
            "intent_extraction": get_gemini_model(),
            "intent_update": get_gemini_model(),
            "textual_fallback": fallback_policy.model,
        },
    }


LEGACY_RUN_METADATA = {
    "status": "unavailable",
    "note": (
        "Predictions were produced before run metadata existed (backend "
        "commit, models and top_n unknown; the adapter of that time used "
        "top_n=5). Re-scoring them with the current contract does not "
        "measure the current backend."
    ),
}

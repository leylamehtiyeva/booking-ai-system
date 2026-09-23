from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from app.logic.conversation_router import (
    ConversationRoutingError,
    route_conversation_async,
)
from app.observability.trace import LLMCallTrace, RequestTrace
from app.schemas.conversation_route import ConversationAction
from evaluation.tasks.conversation_router.adapter import build_router_input
from evaluation.tasks.conversation_router.dataset import (
    ConversationRouterEvalCase,
    load_conversation_router_dataset,
)


class RouterEvalPrediction(BaseModel):
    case_id: str
    user_message: str
    category: str

    profile: str
    repeat_number: int

    expected_action: ConversationAction
    predicted_action: ConversationAction | None = None
    predicted_reason: str | None = None

    correct: bool
    success: bool

    latency_ms: float

    model: str | None = None
    llm_attempts: int = 0

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    error: str | None = None

async def _wait_for_profile_slot(
    *,
    profile: str,
    last_started_at: dict[str, float],
    min_interval_seconds: dict[str, float],
) -> None:
    min_interval = min_interval_seconds.get(
        profile,
        0.0,
    )

    if min_interval <= 0:
        return

    previous_started_at = last_started_at.get(
        profile
    )

    if previous_started_at is not None:
        elapsed = (
            time.perf_counter()
            - previous_started_at
        )

        remaining = min_interval - elapsed

        if remaining > 0:
            await asyncio.sleep(remaining)

    last_started_at[profile] = (
        time.perf_counter()
    )

def _sum_optional_int(
    values: list[int | None],
) -> int | None:
    if not values or any(value is None for value in values):
        return None

    return sum(value for value in values if value is not None)


def _sum_optional_float(
    values: list[float | None],
) -> float | None:
    if not values or any(value is None for value in values):
        return None

    return sum(value for value in values if value is not None)


def _get_router_llm_calls(
    trace: RequestTrace,
) -> list[LLMCallTrace]:
    return [
        call
        for call in trace.llm_calls
        if call.step == "conversation_routing"
    ]


def _build_prediction(
    *,
    case: ConversationRouterEvalCase,
    profile: str,
    repeat_number: int,
    latency_ms: float,
    router_calls: list[LLMCallTrace],
    predicted_action: ConversationAction | None,
    predicted_reason: str | None,
    success: bool,
    error: str | None,
) -> RouterEvalPrediction:
    model = (
        router_calls[-1].model
        if router_calls
        else None
    )

    prompt_tokens = _sum_optional_int(
        [call.prompt_tokens for call in router_calls]
    )

    completion_tokens = _sum_optional_int(
        [call.completion_tokens for call in router_calls]
    )

    total_tokens = _sum_optional_int(
        [call.total_tokens for call in router_calls]
    )

    estimated_cost_usd = _sum_optional_float(
        [
            call.estimated_cost_usd
            for call in router_calls
        ]
    )

    return RouterEvalPrediction(
        case_id=case.id,
        user_message=case.user_message,
        category=case.category,
        profile=profile,
        repeat_number=repeat_number,
        expected_action=case.expected_action,
        predicted_action=predicted_action,
        predicted_reason=predicted_reason,
        correct=(
            success
            and predicted_action == case.expected_action
        ),
        success=success,
        latency_ms=round(latency_ms, 2),
        model=model,
        llm_attempts=len(router_calls),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        error=error,
    )


async def run_router_eval_case(
    *,
    case: ConversationRouterEvalCase,
    profile: str,
    repeat_number: int = 1,
) -> RouterEvalPrediction:
    router_input = build_router_input(case)
    trace = RequestTrace()

    started_at = time.perf_counter()

    try:
        decision = await route_conversation_async(
            router_input=router_input,
            trace=trace,
            llm_profile_name=profile,
        )

    except ConversationRoutingError as exc:
        latency_ms = (
            time.perf_counter() - started_at
        ) * 1000

        return _build_prediction(
            case=case,
            profile=profile,
            repeat_number=repeat_number,
            latency_ms=latency_ms,
            router_calls=_get_router_llm_calls(trace),
            predicted_action=None,
            predicted_reason=None,
            success=False,
            error=exc.code,
        )

    except Exception as exc:
        latency_ms = (
            time.perf_counter() - started_at
        ) * 1000

        return _build_prediction(
            case=case,
            profile=profile,
            repeat_number=repeat_number,
            latency_ms=latency_ms,
            router_calls=_get_router_llm_calls(trace),
            predicted_action=None,
            predicted_reason=None,
            success=False,
            error=f"unexpected:{type(exc).__name__}",
        )

    latency_ms = (
        time.perf_counter() - started_at
    ) * 1000

    return _build_prediction(
        case=case,
        profile=profile,
        repeat_number=repeat_number,
        latency_ms=latency_ms,
        router_calls=_get_router_llm_calls(trace),
        predicted_action=decision.action,
        predicted_reason=decision.reason,
        success=True,
        error=None,
    )


async def _wait_for_profile_slot(
    *,
    profile: str,
    last_started_at: dict[str, float],
    min_interval_seconds: dict[str, float],
) -> None:
    min_interval = min_interval_seconds.get(
        profile,
        0.0,
    )

    if min_interval <= 0:
        return

    previous_started_at = last_started_at.get(profile)

    if previous_started_at is not None:
        elapsed = (
            time.perf_counter()
            - previous_started_at
        )

        remaining = min_interval - elapsed

        if remaining > 0:
            await asyncio.sleep(remaining)

    last_started_at[profile] = time.perf_counter()
    
async def run_predictions(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    profiles: list[str],
    repeats: int = 1,
    limit: int | None = None,
    delay_seconds: float = 0.0,
    profile_min_interval_seconds: dict[str, float] | None = None,
    overwrite: bool = False,
) -> int:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not profiles:
        raise ValueError(
            "At least one LLM profile is required"
        )

    if len(set(profiles)) != len(profiles):
        raise ValueError(
            "LLM profiles must be unique"
        )

    if repeats < 1:
        raise ValueError(
            "repeats must be at least 1"
        )

    if delay_seconds < 0:
        raise ValueError(
            "delay_seconds cannot be negative"
        )

    profile_min_interval_seconds = (
        profile_min_interval_seconds or {}
    )

    unknown_profiles = (
        set(profile_min_interval_seconds)
        - set(profiles)
    )

    if unknown_profiles:
        raise ValueError(
            "Rate limits configured for profiles "
            "that are not being evaluated: "
            f"{sorted(unknown_profiles)}"
        )

    for profile, seconds in (
        profile_min_interval_seconds.items()
    ):
        if seconds < 0:
            raise ValueError(
                f"Minimum interval for {profile} "
                "cannot be negative"
            )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Predictions already exist: {output_path}. "
            "Use overwrite=True to replace them."
        )

    cases = load_conversation_router_dataset(
        dataset_path
    )

    if limit is not None:
        if limit < 1:
            raise ValueError(
                "limit must be at least 1"
            )

        cases = cases[:limit]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_predictions = (
        len(cases)
        * len(profiles)
        * repeats
    )

    completed = 0

    last_profile_call_started_at: dict[
        str,
        float,
    ] = {}

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        with tqdm(
            total=total_predictions,
            desc="Running conversation router evaluation",
        ) as progress:

            for case_index, case in enumerate(cases):
                for repeat_number in range(
                    1,
                    repeats + 1,
                ):
                    ordered_profiles = (
                        profiles
                        if (
                            case_index
                            + repeat_number
                        )
                        % 2
                        == 0
                        else list(
                            reversed(profiles)
                        )
                    )

                    for profile in ordered_profiles:
                        await _wait_for_profile_slot(
                            profile=profile,
                            last_started_at=(
                                last_profile_call_started_at
                            ),
                            min_interval_seconds=(
                                profile_min_interval_seconds
                            ),
                        )

                        prediction = (
                            await run_router_eval_case(
                                case=case,
                                profile=profile,
                                repeat_number=repeat_number,
                            )
                        )

                        output_file.write(
                            json.dumps(
                                prediction.model_dump(
                                    mode="json",
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        output_file.flush()

                        completed += 1
                        progress.update(1)

                        if (
                            delay_seconds > 0
                            and completed
                            < total_predictions
                        ):
                            await asyncio.sleep(
                                delay_seconds
                            )

    return completed
"""
Small, explicit bounded-concurrency execution layer for the
soft-evidence pipeline's external API calls (embedding batches, Gemini
verifier calls).

This module owns EXECUTION only. Planning - which calls are allowed,
in what order, subject to which budgets - always happens before any
call in this module is even constructed, in plain deterministic Python
with no concurrency involved at all (see soft_evidence_orchestration.py:
_plan_verifier_tasks, and the embedding batch-plan building in
build_shadow_soft_preference_evidence). Nothing here decides which
tasks run - it only runs an already-decided, ordered list of them,
bounded by a semaphore, and hands results back in that same order.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


async def run_bounded(
    tasks: list[Awaitable[T]],
    *,
    max_concurrency: int,
) -> list[T]:
    """
    Runs `tasks` (already-created coroutine/awaitable objects, in a
    fixed order the caller determined during planning) with at most
    `max_concurrency` running at once.

    Order guarantee: the returned list is positionally aligned with
    `tasks` regardless of actual completion order - this is a native
    asyncio.gather guarantee, not something this function has to track
    manually. Callers restore results into their planned positions by
    zipping the same task-order metadata they used to build `tasks`
    against this function's return value.

    Failure isolation: uses return_exceptions=True, so one task raising
    an unexpected exception never cancels or orphans sibling tasks -
    every task still runs to completion. Expected operational failures
    (API/network errors, parse/schema failures) are already caught
    INSIDE each task (see soft_evidence_retrieval._embed_batch and
    semantic_evidence_verification.verify_evidence_relation, both of
    which return a normal result object rather than raising for those
    cases) - so in practice only a genuine programming/configuration
    bug would ever surface here as an exception. After every task has
    finished, the first such exception (if any) is re-raised, so real
    bugs still propagate loudly rather than being silently swallowed -
    they are just never allowed to cut a sibling task's execution short.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

    if not tasks:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    results = await asyncio.gather(*(_run_one(task) for task in tasks), return_exceptions=True)

    for result in results:
        if isinstance(result, BaseException):
            raise result

    return list(results)

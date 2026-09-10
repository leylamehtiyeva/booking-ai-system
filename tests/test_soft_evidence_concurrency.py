from __future__ import annotations

import asyncio

import pytest

from app.logic.soft_evidence_concurrency import run_bounded


async def test_empty_task_list_returns_empty():
    assert await run_bounded([], max_concurrency=3) == []


async def test_zero_or_negative_concurrency_raises():
    coro = _noop()
    try:
        with pytest.raises(ValueError):
            await run_bounded([coro], max_concurrency=0)
    finally:
        coro.close()  # never scheduled (raises before awaiting) - avoid an "never awaited" warning


async def _noop():
    return "x"


async def test_results_preserve_input_order_regardless_of_completion_order():
    """Tasks with varying delays finish out of order; results must still
    come back positionally aligned with the input task list."""
    delays = [0.03, 0.0, 0.02, 0.01]

    async def _delayed(index: int, delay: float):
        await asyncio.sleep(delay)
        return index

    tasks = [_delayed(i, d) for i, d in enumerate(delays)]
    results = await run_bounded(tasks, max_concurrency=4)

    assert results == [0, 1, 2, 3]


async def test_bounded_concurrency_never_exceeds_max():
    max_concurrency = 3
    active = 0
    peak_active = 0
    lock = asyncio.Lock()

    async def _tracked_task(i: int):
        nonlocal active, peak_active
        async with lock:
            active += 1
            peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return i

    tasks = [_tracked_task(i) for i in range(10)]
    results = await run_bounded(tasks, max_concurrency=max_concurrency)

    assert results == list(range(10))
    assert peak_active <= max_concurrency
    assert peak_active == max_concurrency  # with 10 tasks and cap 3, the cap should actually be reached


async def test_concurrency_of_one_is_effectively_sequential():
    order: list[int] = []

    async def _tracked(i: int):
        order.append(i)
        await asyncio.sleep(0.001)
        return i

    tasks = [_tracked(i) for i in range(5)]
    results = await run_bounded(tasks, max_concurrency=1)

    assert results == [0, 1, 2, 3, 4]
    assert order == [0, 1, 2, 3, 4]  # concurrency=1 forces strict start order


async def test_one_task_exception_does_not_cancel_siblings():
    completed = []

    async def _ok(i: int):
        await asyncio.sleep(0.01)
        completed.append(i)
        return i

    async def _boom():
        await asyncio.sleep(0.001)
        raise RuntimeError("unexpected bug")

    tasks = [_ok(0), _boom(), _ok(2)]

    with pytest.raises(RuntimeError, match="unexpected bug"):
        await run_bounded(tasks, max_concurrency=3)

    # siblings still ran to completion despite the failure
    assert 0 in completed
    assert 2 in completed


async def test_exception_propagates_after_all_tasks_finish():
    """The re-raised exception must be the real one, not swallowed."""
    async def _boom():
        raise ValueError("config bug")

    async def _ok():
        return "fine"

    with pytest.raises(ValueError, match="config bug"):
        await run_bounded([_ok(), _boom(), _ok()], max_concurrency=2)

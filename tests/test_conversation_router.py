from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from app.logic.conversation_router import (
    _collect_final_response_text,
)


class FakeEvent:
    def __init__(
        self,
        *,
        text_parts: list[str | None],
        is_final: bool,
    ) -> None:
        self.content = SimpleNamespace(
            parts=[
                SimpleNamespace(text=text)
                for text in text_parts
            ]
        )
        self._is_final = is_final

    def is_final_response(self) -> bool:
        return self._is_final


async def _event_stream(
    *events: FakeEvent,
) -> AsyncIterator[FakeEvent]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_collect_final_response_ignores_partial_events():
    partial_event = FakeEvent(
        text_parts=[
            '{"route":"search_update",',
            '"reason":"partial"}',
        ],
        is_final=False,
    )

    final_event = FakeEvent(
        text_parts=[
            '{"route":"other",',
            '"reason":"greeting"}',
        ],
        is_final=True,
    )

    result = await _collect_final_response_text(
        _event_stream(
            partial_event,
            final_event,
        )
    )

    assert result == (
        '{"route":"other",'
        '"reason":"greeting"}'
    )
    
    
@pytest.mark.asyncio
async def test_collect_final_response_returns_none_without_final_text():
    partial_event = FakeEvent(
        text_parts=[
            '{"route":"search_update"}',
        ],
        is_final=False,
    )

    empty_final_event = FakeEvent(
        text_parts=[],
        is_final=True,
    )

    result = await _collect_final_response_text(
        _event_stream(
            partial_event,
            empty_final_event,
        )
    )

    assert result is None
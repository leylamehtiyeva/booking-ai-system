from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTrace:
    name: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallTrace:
    step: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    success: bool = True
    error: str | None = None


@dataclass
class ExternalCallTrace:
    step: str
    provider: str
    latency_ms: float
    estimated_cost_usd: float | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[StepTrace] = field(default_factory=list)
    llm_calls: list[LLMCallTrace] = field(default_factory=list)
    external_calls: list[ExternalCallTrace] = field(default_factory=list)

    @contextmanager
    def step(self, name: str, **metadata: Any):
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self.steps.append(
                StepTrace(
                    name=name,
                    latency_ms=round(latency_ms, 2),
                    metadata=metadata,
                )
            )

    def add_llm_call(self, call: LLMCallTrace) -> None:
        self.llm_calls.append(call)

    def add_external_call(self, call: ExternalCallTrace) -> None:
        self.external_calls.append(call)

    def summary(self) -> dict[str, Any]:
        total_latency_ms = sum(step.latency_ms for step in self.steps)

        llm_cost = sum(
            call.estimated_cost_usd or 0.0
            for call in self.llm_calls
        )

        external_cost = sum(
            call.estimated_cost_usd or 0.0
            for call in self.external_calls
        )

        return {
            "trace_id": self.trace_id,
            "latency_ms": {
                "total_observed": round(total_latency_ms, 2),
                "steps": [
                    {
                        "name": s.name,
                        "latency_ms": s.latency_ms,
                        "metadata": s.metadata,
                    }
                    for s in self.steps
                ],
            },
            "cost": {
                "estimated_total_usd": round(llm_cost + external_cost, 6),
                "estimated_llm_usd": round(llm_cost, 6),
                "estimated_external_usd": round(external_cost, 6),
            },
            "scenario": {
                "used_apify": any(c.provider == "apify" for c in self.external_calls),
                "used_fallback": any(c.step == "constraint_textual_fallback" for c in self.llm_calls),
                "used_intent_extraction": any(
                        c.step in {
                            "initial_intent_extraction",
                            "new_search_intent_extraction",
                            "search_intent_extraction",
                        }
                        for c in self.llm_calls
                    ),
                
                "used_conversation_response": any(
                    c.step == "conversation_response_generation"
                    for c in self.llm_calls
                ),
                "used_conversation_response_fallback": any(
                    c.step == "conversation_response_generation"
                    and not c.success
                    for c in self.llm_calls
                ),
                "used_conversation_router": any(c.step == "conversation_routing" for c in self.llm_calls),
                "used_intent_update": any(c.step == "intent_update" for c in self.llm_calls),
                "used_intent_repair": any(c.step == "intent_repair" for c in self.llm_calls),
                "llm_calls_count": len(self.llm_calls),
                "external_calls_count": len(self.external_calls),
            },
            "llm": {
                "calls_count": len(self.llm_calls),
                "calls": [call.__dict__ for call in self.llm_calls],
            },
            "external": {
                "calls_count": len(self.external_calls),
                "calls": [call.__dict__ for call in self.external_calls],
            },
        }
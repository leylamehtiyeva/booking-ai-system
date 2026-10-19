"""
Generate the frontend's mock API responses from the real backend, offline.

Each scenario runs app.services.conversation_service.process_turn on the
fixture listings - real retrieval, matching, ranking and rendering. Only
the LLM boundaries (router, intent extraction/update, factual extraction)
are scripted, and response wording uses the deterministic fallback, so no
API key or network is used. Output: frontend/src/mocks/turns.json, the
exact public TurnResult JSON that POST /api/chat returns.

    uv run python frontend/scripts/generate_mocks.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Hermetic like tests/conftest.py: no .env keys, no network. A call that
# was not scripted fails instead of reaching a paid API.
import dotenv  # noqa: E402
import dotenv.main  # noqa: E402
import socket  # noqa: E402

dotenv.load_dotenv = dotenv.main.load_dotenv = lambda *a, **k: False
for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "APIFY_API_TOKEN"):
    os.environ.pop(key, None)


def _blocked(*args, **kwargs):
    raise RuntimeError(f"network blocked while generating mocks: {args[:1]}")


socket.getaddrinfo = _blocked

from app.agents.intent_router_agent import IntentRoute  # noqa: E402
from app.config import clock  # noqa: E402
from app.llm.call_policy import LlmCallError  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from app.logic import (  # noqa: E402
    constraint_evidence_resolution,
    conversation_flow,
    conversation_response_llm,
    intent_router,
    intent_update,
)
from app.schemas.constraints import (  # noqa: E402
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.conversation_route import (  # noqa: E402
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.fields import Field  # noqa: E402
from app.schemas.filters import PriceConstraint, SearchFilters  # noqa: E402
from app.schemas.intent_patch import SearchIntentPatch  # noqa: E402
from app.schemas.result_query import ResultQuery  # noqa: E402
from app.schemas.session import ConversationSession  # noqa: E402
from app.schemas.turn import TurnConfig  # noqa: E402
from app.services.conversation_service import process_turn  # noqa: E402

OUT = ROOT / "frontend" / "src" / "mocks" / "turns.json"
TODAY = date(2026, 3, 1)
CONFIG = TurnConfig(persist_telemetry=False)

script: dict[str, list] = {"route": [], "intent": [], "patch": [], "factual": []}


def _decision(action, *, scope=ResultScope.NOT_APPLICABLE, target=""):
    return ConversationActionDecision(
        action=action,
        reason="scripted for the frontend mock",
        decision_status=ConversationDecisionStatus.RESOLVED,
        clarification_reason=ClarificationReason.NONE,
        result_scope=scope,
        target_result_id=target,
    )


def _known(text, field, priority=ConstraintPriority.MUST, category=ConstraintCategory.AMENITY):
    return UserConstraint(
        raw_text=text, normalized_text=text, priority=priority, category=category,
        mapping_status=ConstraintMappingStatus.KNOWN, mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _optional(text):
    return UserConstraint(
        raw_text=text, normalized_text=text, priority=ConstraintPriority.NICE,
        category=ConstraintCategory.OTHER, mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[], evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


def _install_fakes() -> None:
    async def route(**kwargs):
        return script["route"].pop(0)

    async def intent(user_text, **kwargs):
        return script["intent"].pop(0)

    async def patch(previous_state, user_message, **kwargs):
        return script["patch"].pop(0)

    async def factual(user_message, **kwargs):
        return script["factual"].pop(0)

    def no_model(*args, **kwargs):
        raise LlmCallError("configuration_error", stage="conversation_response_generation")

    conversation_flow.route_conversation_async = route
    intent_router.route_intent_adk_async = intent
    intent_update.route_intent_update_patch_async = patch
    conversation_flow.extract_factual_constraints_async = factual
    conversation_response_llm.build_adk_model = no_model

    # Textual-evidence fallback: scripted "no evidence found" (-> UNCERTAIN).
    no_evidence = json.dumps({"signals": [], "evidence_missing": True, "reason": "No evidence in the listing text."})

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=no_evidence, usage_metadata=None)

    constraint_evidence_resolution._gemini_client = lambda: SimpleNamespace(models=Models())


async def _turn(session, message):
    with clock.use_reference_date(TODAY):
        turn = await process_turn(session, message, config=CONFIG)
    return turn.session, {"request": message, "response": json.loads(turn.model_dump_json())}


async def main() -> None:
    _install_fakes()
    scenarios: dict[str, list[dict]] = {}

    # 1. Multi-requirement search, then a refinement, then a listing question.
    script["route"].append(_decision(ConversationAction.START_SEARCH))
    script["intent"].append(
        IntentRoute(
            city="Baku", check_in="2026-04-10", check_out="2026-04-15", property_types=["apartment"],
            constraints=[_known("wifi", Field.WIFI), _known("kitchen", Field.KITCHEN), _optional("quiet street")],
            filters=SearchFilters(price=PriceConstraint(max_amount=150, currency="USD", scope="per_night")),
        )
    )
    session, first = await _turn(
        ConversationSession(),
        "Apartment in Baku from April 10 to 15 with WiFi and a kitchen, up to $150 per night. A quiet street would be nice.",
    )
    script["route"].append(_decision(ConversationAction.UPDATE_SEARCH))
    script["patch"].append(SearchIntentPatch(add_constraints=[_known("balcony", Field.BALCONY)]))
    session, second = await _turn(session, "Only show places with a balcony.")
    target = session.shown_result_set.items[0].result_id if session.shown_result_set else ""
    script["route"].append(_decision(ConversationAction.LISTING_QUESTION, scope=ResultScope.SPECIFIC_RESULT, target=target))
    script["factual"].append(ResultQuery(constraints=[_known("parking", Field.PARKING)]))
    session, third = await _turn(session, "Does the first one have parking?")
    scenarios["search_refine_ask"] = [first, second, third]

    # 2. A budget without a scope: the assistant asks before searching.
    script["route"].append(_decision(ConversationAction.START_SEARCH))
    script["intent"].append(
        IntentRoute(
            city="Baku", check_in="2026-04-10", check_out="2026-04-13", property_types=["hotel"],
            filters=SearchFilters(price=PriceConstraint(max_amount=200, currency="$")),
        )
    )
    session, ask = await _turn(ConversationSession(), "A hotel in Baku from April 10 to 13, $200 max")
    script["route"].append(_decision(ConversationAction.UPDATE_SEARCH))
    script["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope="per_night"))))
    session, answer = await _turn(session, "Per night")
    scenarios["budget_clarification"] = [ask, answer]

    assert not any(script.values()), script
    OUT.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())

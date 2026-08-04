import pytest
from pydantic import ValidationError

from app.schemas.conversation_route import (
    ConversationAction,
    ConversationActionDecision,
    ConversationRouteDecision,
    RouterInput,
)
from app.schemas.query import SearchRequest



def test_router_input_accepts_first_message_without_search():
    router_input = RouterInput(
        user_message="Hello",
    )

    assert router_input.user_message == "Hello"
    assert router_input.current_search is None
    assert router_input.latest_result_context is None
    
    
    
    
def test_router_input_accepts_current_search():
    current_search = SearchRequest(
        city="Baku",
    )

    router_input = RouterInput(
        user_message="Add a kitchen",
        current_search=current_search,
    )

    assert router_input.current_search is not None
    assert router_input.current_search.city == "Baku"
    
    
def test_action_decision_accepts_valid_action():
    decision = ConversationActionDecision(
        action=ConversationAction.START_SEARCH,
        reason="The user wants to begin an accommodation search.",
    )

    assert decision.action == ConversationAction.START_SEARCH
    assert decision.action.value == "start_search"
    assert decision.reason == (
        "The user wants to begin an accommodation search."
    )
    
    
def test_action_decision_parses_action_from_string():
    decision = ConversationActionDecision(
        action="general_chat",
        reason="The user is greeting the assistant.",
    )

    assert decision.action == ConversationAction.GENERAL_CHAT
    
    
    
def test_action_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ConversationActionDecision(
            action="search",
            reason="Unknown action.",
        )
        

def test_legacy_route_decision_still_works():
    decision = ConversationRouteDecision(
        route="search_update",
    )

    assert decision.route == "search_update"
        

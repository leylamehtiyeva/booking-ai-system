from app.schemas.conversation_route import RouterInput
from app.schemas.query import SearchRequest

from evaluation.tasks.conversation_router.dataset import (
    ConversationRouterEvalCase,
)


def build_router_input(
    case: ConversationRouterEvalCase,
) -> RouterInput:
    current_search = (
        SearchRequest()
        if case.has_current_search
        else None
    )

    latest_result_context = (
        {"has_shown_results": True}
        if case.has_shown_results
        else None
    )

    return RouterInput(
        user_message=case.user_message,
        current_search=current_search,
        latest_result_context=latest_result_context,
    )
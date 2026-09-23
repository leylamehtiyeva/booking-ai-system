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

    latest_result_context = None

    if case.has_shown_results:
        # Same lightweight projection shape production builds from
        # ShownResultSet (see conversation_flow._build_shown_results_router_context)
        # - position/result_id/title only, never ListingRaw.
        latest_result_context = {
            "has_shown_results": True,
            "shown_results": [
                {
                    "position": position,
                    "result_id": stub.result_id,
                    "title": stub.title,
                }
                for position, stub in enumerate(
                    case.shown_results, start=1
                )
            ],
        }

    return RouterInput(
        user_message=case.user_message,
        current_search=current_search,
        latest_result_context=latest_result_context,
    )
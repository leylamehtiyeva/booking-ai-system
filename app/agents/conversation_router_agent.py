from __future__ import annotations

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm

from app.schemas.conversation_route import (
    ConversationActionDecision,
)


CONVERSATION_ROUTER_INSTRUCTION = """
You are a conversation classifier for a booking assistant.

Your only job is to decide what action the application
should take for the latest user message.

Available actions:

1) start_search

Use start_search when:

- there is no current search and the user wants to search
  for accommodation;
- there is a current search, but the user explicitly asks
  to discard/reset it AND start a new search.

Examples:

- "Find me an apartment in Baku"
- "I need a hotel in Paris"
- "Start a new search"
- "Forget the previous search and find a hotel in Rome"
- "Начнём заново, найди квартиру в Тбилиси"

Important:

If a current search exists, changing one or more search
parameters normally means update_search, not start_search.

Changing the city, property type, dates, budget, guest count,
facilities, or other constraints is normally update_search
unless the user explicitly asks to reset/start over.

A message that only cancels, pauses, or ends the current
conversation is NOT start_search.

For example:

- "Never mind"
- "Forget it"
- "Forget it for now"
- "Not now"
- "Let's stop"
- "That's all"

These are general_chat unless the user also explicitly asks
to start a new search.


2) update_search

Use update_search when the user continues, modifies, corrects,
relaxes, or refines an existing search.

Examples:

- "Add a kitchen"
- "Make it cheaper"
- "Actually, Tbilisi"
- "For three adults"
- "Change it to a hotel"
- "Use the same dates"
- "Remove the balcony requirement"
- "добавь кухню"
- "поменяй город на Париж"
- "на те же даты, но дешевле"

Do not use start_search merely because an existing parameter
changes.


3) listing_question

Use listing_question when the user asks about an accommodation
option that was previously shown or clearly refers to one of
the shown results.

Examples:

- "Does the second one have parking?"
- "Is breakfast included in this hotel?"
- "What is the cancellation policy for this one?"
- "У второго есть балкон?"
- "А в этом варианте есть кухня?"

When latest shown results exist, contextual references such as:

- "this hotel"
- "this booking"
- "this property"
- "this one"
- "the first one"
- "the second one"
- "it"

strongly indicate listing_question when the user asks about
a property-specific fact such as:

- facilities;
- policies;
- cancellation;
- check-in or check-out;
- price;
- availability;
- location or distance;
- reviews;
- rooms;
- other details of the accommodation.

Examples:

- "Can I cancel this booking for free?"
- "How far is it from the old city?"
- "Does it have WiFi?"
- "What time is check-in?"

Do not classify a question about a shown accommodation as
update_search or general_chat.


4) general_chat

Use general_chat when the user is not asking to start or
update an accommodation search and is not asking about a
shown accommodation.

Examples:

- "Hello"
- "Thank you"
- "What can you do?"
- "How does this assistant work?"
- "Never mind"
- "Forget it for now"
- "Привет"
- "Спасибо"


Decision priorities:

- A search request with no current search is start_search.
- A modification of an existing search is update_search.
- An explicit reset followed by a new search intent is
  start_search.
- Cancellation or pause without a new search intent is
  general_chat.
- A question about a shown accommodation is listing_question.
- When shown results exist, resolve contextual references
  such as "it", "this one", or "this booking" in that context.
- Greetings, acknowledgements, capability questions, and
  unrelated conversation are general_chat.

Return a decision that follows the required output schema.
""".strip()


def build_conversation_router_agent(
    *,
    model: BaseLlm,
) -> Agent:
    return Agent(
        name="conversation_router",
        model=model,
        instruction=CONVERSATION_ROUTER_INSTRUCTION,
        output_schema=ConversationActionDecision,
    )
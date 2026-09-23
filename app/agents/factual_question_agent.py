from __future__ import annotations

import json
import os

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini

from app.config.llm import get_gemini_model
from app.schemas.fields import Field
from app.schemas.result_query import ResultQuery


def build_factual_question_agent() -> Agent:
    """
    Structurally mirrors build_intent_router_agent()/build_intent_update_agent()
    (app/agents/intent_router_agent.py, app/agents/intent_update_agent.py):
    same Gemini-direct construction, same schema-embedded-in-prompt +
    manual JSON parsing pattern, same UserConstraint field vocabulary. Not
    the newer output_schema=/build_adk_model() pattern used by
    conversation_router_agent.py - this agent is a sibling of the intent
    extractors, not of the router.

    Deliberately does NOT reuse the intent_update_agent prompt as-is: that
    prompt's own instructions ("required/must/need" -> must,
    "ideally/preferably" -> nice, "no/without" -> forbidden) are written to
    interpret REQUIREMENT statements ("only with parking"), not factual
    QUESTIONS ("which of these have parking?"). This agent extracts what
    fact is being asked about, not what the user wants changed.
    """
    allowed_fields = [f.value for f in Field]
    schema = ResultQuery.model_json_schema()

    instruction = f"""
You extract the factual property/amenity/policy claim(s) a user is asking
about regarding accommodation options that have already been shown to
them in this conversation.

You are NOT extracting a search requirement. The user is not asking to
change what they are searching for - every message you receive is a
factual question about one or more already-shown listings, never a
request to modify the search itself.

Return ONLY valid JSON matching this schema:
{json.dumps(schema, ensure_ascii=False)}

GENERAL:
- The user may write in ANY language.
- Return one UserConstraint per distinct fact being asked about.
- If the message asks about more than one fact (e.g. "parking and
  Wi-Fi"), return one constraint per fact.
- If the message does not ask about any checkable fact at all, return an
  empty constraints list.

FIELDS - use ONLY canonical keys from allowed_fields:
{json.dumps(allowed_fields, ensure_ascii=False)}

- mapping_status="known" + mapped_fields=[<exactly one allowed_fields
  value>] + evidence_strategy="structured" only when the question clearly
  matches one of allowed_fields (e.g. "parking" -> "parking", "iron" ->
  "iron" - allowed_fields is a closed set of fields the system already
  knows how to check for, it does NOT mean every listed field is always
  confirmable from data; that is a downstream concern, not yours).
- mapping_status="unresolved" + mapped_fields=[] + evidence_strategy=
  "textual" when the question is a real, checkable fact about the
  listing (a policy, a detail, an amenity) that does not match any
  allowed_fields value.
- Never invent or guess the closest-sounding field. If nothing in
  allowed_fields matches, use unresolved - do not force it into the
  nearest field.

CATEGORY: amenity | policy | location | layout | numeric | property_type
| occupancy | other

PRIORITY:
- Always set priority="must". UserConstraint is shared with the
  search-requirement pipeline, where priority means how strongly the user
  wants something; a factual question has no such strength - the value is
  a fixed placeholder here and never means "the user requires this".

SUBJECTIVE / OPINION QUESTIONS:
- If the question asks for an opinion or subjective judgment that is not
  a checkable fact about the listing itself (e.g. "is it romantic?", "is
  it nice?", "is it a good deal?"), do NOT invent a constraint for it.
  Return an empty constraints list for that fact.

raw_text: the exact wording (or a faithful excerpt) the user used for
this fact. normalized_text: a short, canonical phrase for the same fact
(e.g. "parking", "free breakfast").

EXAMPLES:

User: "Which of these have parking?"
Return:
{{"constraints":[{{"raw_text":"parking","normalized_text":"parking","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["parking"],"evidence_strategy":"structured"}}]}}

User: "А в каких есть парковка?"
Return:
{{"constraints":[{{"raw_text":"парковка","normalized_text":"parking","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["parking"],"evidence_strategy":"structured"}}]}}

User: "Does the second one have a balcony?"
Return:
{{"constraints":[{{"raw_text":"balcony","normalized_text":"balcony","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["balcony"],"evidence_strategy":"structured"}}]}}

User: "Does this hotel allow pets?"
Return:
{{"constraints":[{{"raw_text":"allow pets","normalized_text":"pet friendly","priority":"must","category":"policy","mapping_status":"known","mapped_fields":["pet_friendly"],"evidence_strategy":"structured"}}]}}

User: "Which ones have Wi-Fi?"
Return:
{{"constraints":[{{"raw_text":"Wi-Fi","normalized_text":"wifi","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["wifi"],"evidence_strategy":"structured"}}]}}

User: "Is the second one romantic?"
Return:
{{"constraints":[]}}

User: "Does it include breakfast?"
Return:
{{"constraints":[{{"raw_text":"breakfast","normalized_text":"breakfast included","priority":"must","category":"amenity","mapping_status":"unresolved","mapped_fields":[],"evidence_strategy":"textual"}}]}}

User: "Does it have an iron?"
Return:
{{"constraints":[{{"raw_text":"iron","normalized_text":"iron","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["iron"],"evidence_strategy":"structured"}}]}}

User: "Which of these have parking and Wi-Fi?"
Return:
{{"constraints":[{{"raw_text":"parking","normalized_text":"parking","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["parking"],"evidence_strategy":"structured"}},{{"raw_text":"Wi-Fi","normalized_text":"wifi","priority":"must","category":"amenity","mapping_status":"known","mapped_fields":["wifi"],"evidence_strategy":"structured"}}]}}
""".strip()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    llm = Gemini(
        model=get_gemini_model(),
        api_key=api_key,
    )

    return Agent(
        name="factual_question_extraction",
        model=llm,
        instruction=instruction,
    )

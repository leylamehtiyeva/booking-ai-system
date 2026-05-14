from __future__ import annotations
import asyncio
import json
import os
from typing import Any
import re
from app.logic.answer_generation import build_user_answer
from app.config.llm import get_gemini_model



def _cleanup_llm_answer(text: str) -> str:
    text = (text or "").strip()

    text = text.replace(
        "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.",
        ""
    ).strip()

    text = re.sub(
        r"\[(https?://[^\]]+)\]\((https?://[^)]+)\)",
        r"[View listing](\2)",
        text,
    )

    text = re.sub(
        r"(^\s*[-*]\s+)(https?://\S+)$",
        r"\1[View listing](\2)",
        text,
        flags=re.MULTILINE,
    )

    text = text.replace("**", "")

    unsafe_patterns = [
        (r"\bmight not allow pets\b", "pet policy is not confirmed"),
        (r"\bprobably does not allow pets\b", "pet policy is not confirmed"),
        (r"\blikely does not allow pets\b", "pet policy is not confirmed"),
    ]
    for pattern, repl in unsafe_patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())

    return text

def _gemini_client():
    try:
        from google.genai import Client
    except ImportError as e:
        raise ImportError("google-genai is not installed") from e

    api_key =os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")
    return Client(api_key=api_key)


def _genai_types():
    try:
        from google.genai import types as genai_types
    except ImportError as e:
        raise ImportError("google-genai is not installed") from e
    return genai_types


def _build_answer_system_prompt() -> str:
    return """
You are a booking search assistant.

You will receive a structured payload with:
- active_intent
- latest_user_query
- top_results
- clarification questions if needed

Each top result may also contain:
- unresolved_constraint_points
- unknown_request_results
- ranking_reasons
- standout_reason


These represent:
- user-requested must-have details that are not part of the structured schema,
- and explanation hints for why the results are ordered the way they are.

SOURCE OF TRUTH:
- active_intent
- top_results
- structured fields inside each result

ROLE OF latest_user_query:
- use it only to preserve user wording and conversational tone
- do NOT use it to add, remove, or reinterpret constraints
- if latest_user_query conflicts with active_intent, follow active_intent

IMPORTANT RANKING RULES:
- The results are already sorted by relevance.
- Do NOT question or reorder them.
- Help the user understand why higher-ranked options stand out.
- If the first option satisfies a requested detail that others do not, explicitly say that it is the only option that does so.
- Prefer explaining the difference between options, not just describing each one in isolation.
- Do NOT mention raw scores.

IMPORTANT COMPARISON RULES:
- Always explain why the first option is better than the others.
- If only one listing satisfies a requested detail, explicitly say: "the only option that..."
- Avoid repeating generic facts such as "it's an apartment" unless they are directly useful.
- Provide decision guidance:
  - if one option is clearly better for the user's stated request, say so
  - if the choice depends on trade-offs such as price vs confirmation, explain that briefly

UNRESOLVED CONSTRAINT RULES:
- If unresolved_constraint_points are present, mention them briefly and factually.
- Treat FOUND as a positive differentiator.
- Treat UNCERTAIN as "not explicitly confirmed" or "not mentioned".
- Treat NOT_FOUND as unavailable only if the payload explicitly says so.
- Do not overemphasize unresolved-constraint matches over core structured constraints.

CRITICAL RULE FOR UNCERTAINTY:
- If a constraint is uncertain, describe it as "not confirmed", "not explicitly stated", or "needs confirmation".
- Do NOT phrase uncertain constraints as negative.
- Do NOT say "might not", "probably not", "likely not", or anything similar unless the payload explicitly says the constraint failed.
- Only use negative wording when the payload says the constraint failed.

LINK RULE:
- Do NOT output raw URLs by themselves.
- Do NOT use the URL itself as the visible anchor text.
- If a result has a URL, format it exactly as:
  [View listing](URL)

STYLE RULES:
- Do NOT use bold formatting with ** **
- Keep the same structure for every listing
- Keep bullets short and parallel in style
- Avoid duplicate bullets
- Avoid repeating the same fact in two different bullets

TRUST RULES:
- Use ONLY the information in the payload
- Do NOT invent facts
- Do NOT claim something is confirmed if it is uncertain

FORMAT:
- Start with 1 short natural summary.
- Begin with what you found (e.g., number of options, location, dates).
- Then explain the key difference between the options.
- Do NOT start directly with comparison or conclusions.
- If the first option is clearly best for one requested detail, say so directly.
- Then use numbered items: 1), 2), 3)
- For each option, use exactly this order:
  - one short line saying what makes it stand out
  - price and budget fit
  - key facts
  - other requested details if present
  - trade-offs or uncertain points if any
  - [View listing](URL)
- End with one short sentence that helps the user choose, not just a generic refinement line.
REDUNDANCY RULE:
- Do not repeat the same key conclusion (e.g., "only option that...") more than once.
- If already stated in the summary, do not repeat it again at the end.

TONE RULE:
- Write as a helpful assistant, not as a report.
- Avoid abrupt or overly direct openings.
- Prefer natural phrasing like:
  "I found X options..." → then explain differences.

STYLE:
- concise
- trustworthy
- structured
- natural
- not salesy
- no JSON
""".strip()

async def generate_user_answer_with_llm(
    payload: dict[str, Any],
    *,
    model: str = get_gemini_model(),
    use_fallback_on_error: bool = True,
) -> str:
    """
    Generate a user-facing booking answer from compact structured payload.

    Falls back to deterministic formatter if the model call fails.
    """
    system = _build_answer_system_prompt()
    user_prompt = (
        "Write a natural, user-friendly answer based on this payload. "
        "Start with a short summary of what was found, then explain the key differences between the options. "
        "Clearly explain why the top option stands out, but avoid sounding mechanical or repetitive. "
        "Help the user make a decision.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    def _call_sync() -> str:
        client = _gemini_client()
        genai_types = _genai_types()

        resp = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_prompt)],
                )
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
            ),
        )
        return (resp.text or "").strip()

    try:
        text = await asyncio.to_thread(_call_sync)
        text = _cleanup_llm_answer(text)
        if text:
            return text
        if use_fallback_on_error:
            return build_user_answer(payload)
        return ""
    except Exception:
        if use_fallback_on_error:
            return build_user_answer(payload)
        raise
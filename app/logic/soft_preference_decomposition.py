"""
Controlled atomic decomposition of UserConstraint -> validated Phase B
atomic claims.

Deliberately NOT an LLM step: a deterministic keyword-family lookup,
the same pattern as app.logic.field_rules.FIELD_RULES, mapping a
constraint's normalized text to zero or more of exactly the 7 validated
semantic families / 12 claim IDs that went through the frozen
evaluation/experiments/verifier_comparison A1/A2/B/C benchmark.

Hypothesis text and retrieval queries below are copied verbatim from:
- evaluation/experiments/verifier_comparison/build_benchmark.py (HYPOTHESES)
- evaluation/experiments/pipeline_cleanup_v2/schema_v3.py (QUIET_RETRIEVAL_QUERIES)
- evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v2.jsonl
  (RW1/RW2a/RW2b/FC1 retrieval_query fields)
No new atomic hypothesis is invented here. FAM1/BQ1/CL1/NL1/BC1 have no
validated retrieval query anywhere in the experiments - for those,
get_retrieval_query() falls back to the frozen hypothesis text itself
(the general-case convention already used for most subclaims in
retrieval_checkpoint_v2.jsonl, not a new invention).

FC2/FC3 (an earlier, superseded family-friendly decomposition) are
intentionally excluded - they never went through the frozen A1/A2/B/C
benchmark. Not reinstated here; a possible later, explicit extension.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.schemas.constraints import ConstraintPriority, UserConstraint

CLAIM_HYPOTHESES: dict[str, str] = {
    "Q1": "The room/property has soundproofing.",
    "Q2": "The immediate surroundings are explicitly described as quiet.",
    "Q3": "Guests report little or no significant noise disturbance.",
    "FC1": "The property's policies formally accommodate children.",
    "FAM1": "The property is suitable for families.",
    "RW1": "The room has a desk or dedicated workspace.",
    "RW2A": "Wi-Fi / internet access is available.",
    "RW2B": "Internet connection is reliable enough for remote work.",
    "BQ1": "The breakfast is good in quality.",
    "CL1": "The property is clean.",
    "NL1": "Nightlife is available nearby.",
    "BC1": "The bed is comfortable.",
}

# Validated short retrieval queries - only exist for these 7 claim IDs.
CLAIM_RETRIEVAL_QUERIES: dict[str, str] = {
    "Q1": "soundproofing",
    "Q2": "quiet surroundings",
    "Q3": "no noise disturbance",
    "RW1": "desk or workspace in the room",
    "RW2A": "wifi available",
    "RW2B": "reliable fast internet connection",
    "FC1": "children welcome policy",
}


def get_retrieval_query(claim_id: str) -> str:
    """Validated short query if one exists, else the frozen hypothesis text."""
    return CLAIM_RETRIEVAL_QUERIES.get(claim_id, CLAIM_HYPOTHESES[claim_id])


# Deterministic family -> claim_ids + trigger keywords, same pattern as
# app.logic.field_rules.FIELD_RULES. Keywords are matched against
# constraint.normalized_text (already lowercased/normalized upstream by
# the intent extraction step, matched case-insensitively here regardless).
SOFT_PREFERENCE_FAMILY_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # family -> (claim_ids, trigger keywords)
    "quiet": (("Q1", "Q2", "Q3"), ("quiet", "silence", "silent", "peaceful", "noise")),
    "remote_work": (
        ("RW1", "RW2A", "RW2B"),
        ("remote work", "work-friendly", "work friendly", "workspace", "desk", "wifi", "wi-fi"),
    ),
    "family_friendly": (
        ("FC1", "FAM1"),
        ("family", "families", "kid-friendly", "kid friendly", "child-friendly", "child friendly", "children"),
    ),
    "breakfast_quality": (("BQ1",), ("breakfast",)),
    "cleanliness": (("CL1",), ("clean", "cleanliness", "spotless", "tidy")),
    "nightlife": (("NL1",), ("nightlife", "bars", "clubs", "party scene")),
    "bed_comfort": (("BC1",), ("comfortable bed", "bed comfort", "mattress")),
}

# Canonical, fixed claim ordering - used by the fair verifier scheduler
# (soft_evidence_orchestration.py), not by decomposition itself.
CANONICAL_CLAIM_ORDER: tuple[str, ...] = (
    "BC1", "BQ1", "CL1", "FAM1", "FC1", "NL1", "Q1", "Q2", "Q3", "RW1", "RW2A", "RW2B",
)


class PreferenceDirection(str, Enum):
    """
    Derived only from ConstraintPriority - no lexical negation parsing.
    FORBIDDEN constraints keep their factual evidence relation unchanged
    (e.g. NL1 SUPPORT is still a correct, true statement even when the
    user forbade nightlife) - this metadata exists so a future ranking
    layer can invert desirability, not so the evidence pipeline does.
    """

    DESIRED = "desired"
    FORBIDDEN = "forbidden"


def _preference_direction_for(priority: ConstraintPriority) -> PreferenceDirection:
    if priority == ConstraintPriority.FORBIDDEN:
        return PreferenceDirection.FORBIDDEN
    return PreferenceDirection.DESIRED


class AtomicClaimAssignment(BaseModel):
    """
    One (claim_id, contributing constraint) pairing, produced by
    decompose_constraints(). Request-level (not per-hotel) - the same
    assignments apply to every shadow hotel in a request.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    hypothesis: str
    family: str
    source_constraint_id: str | None
    source_constraint_text: str
    preference_direction: PreferenceDirection


def decompose_constraints(constraints: list[UserConstraint]) -> list[AtomicClaimAssignment]:
    """
    Maps each constraint to zero or more validated families (a single
    constraint may match multiple families - not stopped at the first
    match), then deduplicates by claim_id: if two constraints both reach
    the same claim_id, only the first assignment for that claim_id is
    kept (first-seen wins - deterministic given `constraints`' order).

    Unsupported constraints (matching no family) contribute nothing -
    no claims are invented or guessed for them.
    """
    assignments: list[AtomicClaimAssignment] = []
    seen_claim_ids: set[str] = set()

    for constraint in constraints or []:
        text = (constraint.normalized_text or constraint.raw_text or "").casefold()
        if not text:
            continue

        direction = _preference_direction_for(constraint.priority)

        for family, (claim_ids, keywords) in SOFT_PREFERENCE_FAMILY_RULES.items():
            if not any(keyword in text for keyword in keywords):
                continue

            for claim_id in claim_ids:
                if claim_id in seen_claim_ids:
                    continue
                seen_claim_ids.add(claim_id)
                assignments.append(
                    AtomicClaimAssignment(
                        claim_id=claim_id,
                        hypothesis=CLAIM_HYPOTHESES[claim_id],
                        family=family,
                        source_constraint_id=constraint.id,
                        source_constraint_text=constraint.normalized_text or constraint.raw_text,
                        preference_direction=direction,
                    )
                )

    return assignments

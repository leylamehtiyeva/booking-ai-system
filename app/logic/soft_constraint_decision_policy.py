"""
Deterministic Python policy that aggregates one hotel's AtomicClaimResult[]
(SoftPreferenceEvidence.claims) into a SoftConstraintDecision for one
original UserConstraint - the missing bridge documented in the Phase C
task: AtomicClaimResult[] -> decision for the original UserConstraint.

No LLM is used here. Every branch below is plain, inspectable Python -
see the Phase C task spec for the exact conservative rules per family
(quiet, remote_work, family_friendly, and four single-claim families).

Two-step design (kept deliberately separate, per the task's "separate
factual state from user desirability" requirement):

1. compute_family_factual_state(): factual, direction-agnostic -
   SATISFIED / VIOLATED / UNRESOLVED for the positive semantic
   proposition a family represents. Never inverted for FORBIDDEN.
2. apply_preference_direction(): applied exactly once, at the
   constraint boundary, turning (factual_state, PreferenceDirection)
   into YES / NO / UNCERTAIN.

Multiple families per constraint are combined conservatively (see
combine_family_states) BEFORE apply_preference_direction runs - so a
constraint is never partially DESIRED and partially FORBIDDEN.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.constraints import ConstraintPriority, UserConstraint
from app.schemas.soft_constraint_decision import (
    ConstraintDecisionValue,
    FactualState,
    SoftConstraintDecision,
    SoftDecisionEvidenceRef,
)
from app.schemas.soft_evidence import (
    AtomicClaimResult,
    ClaimRelation,
    EvidenceResolutionStatus,
    RetrievalStatus,
    SoftPreferenceEvidence,
)
from app.logic.soft_preference_decomposition import (
    SOFT_PREFERENCE_FAMILY_RULES,
    PreferenceDirection,
)

# families_for_constraint() intentionally re-runs the SAME keyword match
# decompose_constraints() uses (not a lookup into its output) - see
# families_for_constraint's docstring for why.
_SINGLE_CLAIM_FAMILIES: dict[str, str] = {
    "breakfast_quality": "BQ1",
    "cleanliness": "CL1",
    "nightlife": "NL1",
    "bed_comfort": "BC1",
}


def families_for_constraint(constraint: UserConstraint) -> list[str]:
    """
    Which validated semantic families this constraint's own text matches -
    independent of app.logic.soft_preference_decomposition.decompose_constraints's
    request-level claim_id dedup (first-seen-wins across constraints).

    Recomputing this directly from the constraint's own text (rather than
    grouping decompose_constraints()'s output by source_constraint_id) is
    deliberate: two different constraints that both name the same family
    (e.g. two separate "quiet" mentions) must both be able to resolve from
    the SAME already-computed claim evidence - dedup only decides which
    constraint's request "owns" the retrieval-triggering assignment, not
    which constraints are semantically supported.

    Returns [] for a constraint with no matched family - the constraint is
    unsupported by the new pipeline (decide_constraint returns None).
    """
    text = (constraint.normalized_text or constraint.raw_text or "").casefold()
    if not text:
        return []
    return [
        family
        for family, (_claim_ids, keywords) in SOFT_PREFERENCE_FAMILY_RULES.items()
        if any(keyword in text for keyword in keywords)
    ]


@dataclass(frozen=True)
class _FamilyOutcome:
    family: str
    state: FactualState
    decisive_claim_ids: tuple[str, ...]
    technical_notes: tuple[str, ...]
    reason_fragment: str


def _claim(claims_by_id: dict[str, AtomicClaimResult], claim_id: str) -> AtomicClaimResult | None:
    return claims_by_id.get(claim_id)


def _relation_for_satisfaction(claim: AtomicClaimResult | None) -> ClaimRelation | None:
    """
    Only a claim whose retrieval technically fully succeeded may vote
    toward a positive SATISFIED conclusion - FAILED obviously cannot,
    and PARTIAL is withheld too (an unseen portion of the pool might
    have contained a contradiction we never got to look at).
    """
    if claim is None or claim.retrieval_status != RetrievalStatus.SUCCESS:
        return None
    return claim.relation


def _relation_for_violation(claim: AtomicClaimResult | None) -> ClaimRelation | None:
    """
    A contradiction found in evidence that WAS successfully retrieved is
    real evidence, even if the retrieval was only PARTIAL - only a fully
    FAILED retrieval (no evidence seen at all) is excluded.
    """
    if claim is None or claim.retrieval_status == RetrievalStatus.FAILED:
        return None
    return claim.relation


def _raw_relation(claim: AtomicClaimResult | None) -> ClaimRelation | None:
    return claim.relation if claim is not None else None


def _technical_note(claim_id: str, claim: AtomicClaimResult | None) -> str | None:
    """
    Preserves technical status/provenance for debugging without ever
    converting a technical failure into a factual verdict (section 4).
    """
    if claim is None:
        return f"{claim_id}: not evaluated"

    if claim.retrieval_status == RetrievalStatus.FAILED:
        errors = "; ".join(claim.retrieval_errors) or "unknown error"
        return f"{claim_id}: retrieval failed ({errors})"

    if claim.retrieval_status == RetrievalStatus.PARTIAL:
        errors = "; ".join(claim.retrieval_errors) or "unknown error"
        return f"{claim_id}: retrieval partial ({errors})"

    unresolved_items = [
        item for item in claim.evidence_items if item.resolution_status != EvidenceResolutionStatus.RESOLVED
    ]
    if unresolved_items:
        statuses = sorted({item.resolution_status.value for item in unresolved_items})
        return f"{claim_id}: {len(unresolved_items)} evidence item(s) unresolved ({', '.join(statuses)})"

    return None


def _notes_for(claims_by_id: dict[str, AtomicClaimResult], claim_ids: tuple[str, ...]) -> tuple[str, ...]:
    notes = [_technical_note(cid, _claim(claims_by_id, cid)) for cid in claim_ids]
    return tuple(n for n in notes if n)


# ==================================================================
# Per-family conservative policies (see the Phase C task spec for the
# exact rules each of these implements verbatim).
# ==================================================================


def _quiet_state(claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    q1, q2, q3 = _claim(claims_by_id, "Q1"), _claim(claims_by_id, "Q2"), _claim(claims_by_id, "Q3")
    notes = _notes_for(claims_by_id, ("Q1", "Q2", "Q3"))

    q2_violation = _relation_for_violation(q2)
    q3_violation = _relation_for_violation(q3)
    if q2_violation == ClaimRelation.CONTRADICT or q3_violation == ClaimRelation.CONTRADICT:
        decisive = tuple(
            cid for cid, rel in (("Q2", q2_violation), ("Q3", q3_violation)) if rel == ClaimRelation.CONTRADICT
        )
        return _FamilyOutcome(
            "quiet", FactualState.VIOLATED, decisive, notes,
            "guest/listing evidence explicitly reports significant noise or non-quiet surroundings",
        )

    # MIXED (or CONTRADICT, already excluded above) on Q2/Q3 must block a
    # positive YES - known live issue: the verifier has been observed
    # mis-scoring "quiet street view" as Q3 SUPPORT (a view claim, not a
    # noise-disturbance report). This policy does NOT special-case that
    # text pattern (would be "hiding the bug with vague aggregation",
    # explicitly disallowed) - see
    # tests/test_soft_constraint_decision_policy.py::test_known_regression_quiet_street_view_false_q3_support
    # for the documented current behavior and app.logic.semantic_evidence_verification
    # for where an eventual verifier-level fix belongs.
    blocked = _raw_relation(q2) in (ClaimRelation.MIXED, ClaimRelation.CONTRADICT) or _raw_relation(q3) in (
        ClaimRelation.MIXED,
        ClaimRelation.CONTRADICT,
    )

    if not blocked and _relation_for_satisfaction(q3) == ClaimRelation.SUPPORT:
        return _FamilyOutcome(
            "quiet", FactualState.SATISFIED, ("Q3",), notes,
            "guests explicitly report little or no significant noise disturbance",
        )

    if (
        not blocked
        and _relation_for_satisfaction(q1) == ClaimRelation.SUPPORT
        and _relation_for_satisfaction(q2) == ClaimRelation.SUPPORT
    ):
        return _FamilyOutcome(
            "quiet", FactualState.SATISFIED, ("Q1", "Q2"), notes,
            "soundproofing is confirmed and the immediate surroundings are explicitly described as quiet",
        )

    return _FamilyOutcome(
        "quiet", FactualState.UNRESOLVED, (), notes,
        "no conclusive evidence establishes whether the property is quiet",
    )


def _remote_work_state(claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    rw1, rw2a, rw2b = (
        _claim(claims_by_id, "RW1"),
        _claim(claims_by_id, "RW2A"),
        _claim(claims_by_id, "RW2B"),
    )
    notes = _notes_for(claims_by_id, ("RW1", "RW2A", "RW2B"))
    claim_pairs = (("RW1", rw1), ("RW2A", rw2a), ("RW2B", rw2b))

    contradicted = tuple(cid for cid, c in claim_pairs if _relation_for_violation(c) == ClaimRelation.CONTRADICT)
    if contradicted:
        return _FamilyOutcome(
            "remote_work", FactualState.VIOLATED, contradicted, notes,
            f"listing evidence explicitly contradicts {'/'.join(contradicted)}",
        )

    satisfactions = {cid: _relation_for_satisfaction(c) for cid, c in claim_pairs}
    raw_relations = {cid: _raw_relation(c) for cid, c in claim_pairs}
    all_support = all(rel == ClaimRelation.SUPPORT for rel in satisfactions.values())
    none_mixed = all(rel != ClaimRelation.MIXED for rel in raw_relations.values())

    if all_support and none_mixed:
        return _FamilyOutcome(
            "remote_work", FactualState.SATISFIED, ("RW1", "RW2A", "RW2B"), notes,
            "desk/workspace, Wi-Fi availability, and internet reliability are all explicitly supported",
        )

    labels = {"RW1": "desk/workspace", "RW2A": "Wi-Fi availability", "RW2B": "internet reliability"}
    missing = [labels[cid] for cid, rel in satisfactions.items() if rel != ClaimRelation.SUPPORT]
    return _FamilyOutcome(
        "remote_work", FactualState.UNRESOLVED, (), notes,
        f"no evidence establishes {', '.join(missing)}",
    )


def _family_friendly_state(claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    fc1, fam1 = _claim(claims_by_id, "FC1"), _claim(claims_by_id, "FAM1")
    notes = _notes_for(claims_by_id, ("FC1", "FAM1"))

    contradicted = tuple(
        cid
        for cid, c in (("FC1", fc1), ("FAM1", fam1))
        if _relation_for_violation(c) == ClaimRelation.CONTRADICT
    )
    if contradicted:
        return _FamilyOutcome(
            "family_friendly", FactualState.VIOLATED, contradicted, notes,
            f"listing evidence explicitly contradicts {'/'.join(contradicted)}",
        )

    if (
        _relation_for_satisfaction(fc1) == ClaimRelation.SUPPORT
        and _relation_for_satisfaction(fam1) == ClaimRelation.SUPPORT
    ):
        return _FamilyOutcome(
            "family_friendly", FactualState.SATISFIED, ("FC1", "FAM1"), notes,
            "policies formally accommodate children and the property is explicitly described as family-suitable",
        )

    return _FamilyOutcome(
        "family_friendly", FactualState.UNRESOLVED, (), notes,
        "no conclusive evidence establishes whether the property is family-friendly",
    )


def _single_claim_state(family: str, claim_id: str, claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    claim = _claim(claims_by_id, claim_id)
    notes = _notes_for(claims_by_id, (claim_id,))

    if _relation_for_violation(claim) == ClaimRelation.CONTRADICT:
        return _FamilyOutcome(
            family, FactualState.VIOLATED, (claim_id,), notes,
            f"{claim_id} is explicitly contradicted by listing/guest evidence",
        )

    if _relation_for_satisfaction(claim) == ClaimRelation.SUPPORT:
        return _FamilyOutcome(
            family, FactualState.SATISFIED, (claim_id,), notes,
            f"{claim_id} is explicitly supported by listing/guest evidence",
        )

    return _FamilyOutcome(
        family, FactualState.UNRESOLVED, (), notes,
        f"no conclusive evidence resolves {claim_id}",
    )


_FAMILY_POLICIES = {
    "quiet": _quiet_state,
    "remote_work": _remote_work_state,
    "family_friendly": _family_friendly_state,
}


def compute_family_factual_state(family: str, claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    policy = _FAMILY_POLICIES.get(family)
    if policy is not None:
        return policy(claims_by_id)

    single_claim_id = _SINGLE_CLAIM_FAMILIES.get(family)
    if single_claim_id is not None:
        return _single_claim_state(family, single_claim_id, claims_by_id)

    raise ValueError(f"No decision policy implemented for family={family!r}")


def combine_family_states(states: list[FactualState]) -> FactualState:
    """
    Section 5: one UserConstraint may match multiple families - combine
    conservatively. ANY violated family wins over any number of
    satisfied ones (a constraint is not "mostly satisfied").
    """
    if not states:
        return FactualState.UNRESOLVED
    if any(state == FactualState.VIOLATED for state in states):
        return FactualState.VIOLATED
    if all(state == FactualState.SATISFIED for state in states):
        return FactualState.SATISFIED
    return FactualState.UNRESOLVED


_DIRECTION_TABLE: dict[tuple[PreferenceDirection, FactualState], ConstraintDecisionValue] = {
    (PreferenceDirection.DESIRED, FactualState.SATISFIED): ConstraintDecisionValue.YES,
    (PreferenceDirection.DESIRED, FactualState.VIOLATED): ConstraintDecisionValue.NO,
    (PreferenceDirection.DESIRED, FactualState.UNRESOLVED): ConstraintDecisionValue.UNCERTAIN,
    (PreferenceDirection.FORBIDDEN, FactualState.SATISFIED): ConstraintDecisionValue.NO,
    (PreferenceDirection.FORBIDDEN, FactualState.VIOLATED): ConstraintDecisionValue.YES,
    (PreferenceDirection.FORBIDDEN, FactualState.UNRESOLVED): ConstraintDecisionValue.UNCERTAIN,
}


def apply_preference_direction(state: FactualState, direction: PreferenceDirection) -> ConstraintDecisionValue:
    """Section 2's table, applied exactly once, at the constraint boundary."""
    return _DIRECTION_TABLE[(direction, state)]


def _preference_direction_for(priority: ConstraintPriority) -> PreferenceDirection:
    if priority == ConstraintPriority.FORBIDDEN:
        return PreferenceDirection.FORBIDDEN
    return PreferenceDirection.DESIRED


def _evidence_refs(
    claims_by_id: dict[str, AtomicClaimResult],
    decisive_claim_ids: tuple[str, ...],
) -> list[SoftDecisionEvidenceRef]:
    refs: list[SoftDecisionEvidenceRef] = []
    for claim_id in decisive_claim_ids:
        claim = _claim(claims_by_id, claim_id)
        if claim is None:
            continue
        for item in claim.evidence_items:
            if item.resolution_status != EvidenceResolutionStatus.RESOLVED or item.relation is None:
                continue
            refs.append(
                SoftDecisionEvidenceRef(
                    claim_id=claim_id,
                    evidence_text=item.evidence_text,
                    relation=item.relation,
                    source_type=item.source_type,
                    source_path=item.source_path,
                )
            )
    return refs


def _build_reason(decision: ConstraintDecisionValue, outcomes: list[_FamilyOutcome]) -> str:
    fragments = [outcome.reason_fragment for outcome in outcomes]
    body = "; ".join(fragments) if fragments else "no supported evidence family produced a conclusive result"
    return f"{decision.value}: {body}."


def decide_constraint(
    constraint: UserConstraint,
    evidence: SoftPreferenceEvidence,
) -> SoftConstraintDecision | None:
    """
    Returns None when the constraint matches no validated semantic
    family (unsupported - caller must fall back to the legacy textual
    resolver for it). Otherwise always returns a decision - UNCERTAIN
    is a real decision, not an absence of one.
    """
    families = families_for_constraint(constraint)
    if not families:
        return None

    claims_by_id = {claim.claim_id: claim for claim in evidence.claims}

    outcomes = [compute_family_factual_state(family, claims_by_id) for family in families]
    combined_state = combine_family_states([outcome.state for outcome in outcomes])

    direction = _preference_direction_for(constraint.priority)
    decision_value = apply_preference_direction(combined_state, direction)

    claim_ids = sorted({cid for family in families for cid in SOFT_PREFERENCE_FAMILY_RULES[family][0]})
    decisive_claim_ids = sorted({cid for outcome in outcomes for cid in outcome.decisive_claim_ids})
    technical_notes = [note for outcome in outcomes for note in outcome.technical_notes]
    evidence_refs = _evidence_refs(claims_by_id, tuple(decisive_claim_ids))

    return SoftConstraintDecision(
        constraint_id=constraint.id,
        raw_text=constraint.raw_text,
        normalized_text=constraint.normalized_text,
        priority=constraint.priority,
        families=families,
        preference_direction=direction,
        factual_state=combined_state,
        decision=decision_value,
        reason=_build_reason(decision_value, outcomes),
        claim_ids=claim_ids,
        decisive_claim_ids=decisive_claim_ids,
        evidence=evidence_refs,
        technical_notes=technical_notes,
    )

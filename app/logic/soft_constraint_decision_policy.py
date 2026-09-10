"""
Deterministic Python policy that aggregates one hotel's AtomicClaimResult[]
(SoftPreferenceEvidence.claims) into a SoftConstraintDecision for one
original UserConstraint - the missing bridge documented in the Phase C
task: AtomicClaimResult[] -> decision for the original UserConstraint.

PRODUCT CONTRACT (read before changing any rule below): this system does
not certify that a hotel objectively has a subjective property like
"quiet" or "good for remote work" - the user still opens the listing and
judges for themselves. The goal is to use available evidence to narrow
the search to good CANDIDATES. Concretely:

- YES means "the available evidence is sufficiently supportive to
  consider this listing a good candidate for the preference" - not "we
  have proven this property objectively satisfies it".
- NO means "the available evidence gives a strong reason NOT to
  consider this listing a candidate".
- UNCERTAIN means "evidence is insufficient or materially conflicting"
  - it is not a failure, and downstream eligibility does not treat it
    as one (see app.logic.listing_evaluation._fails_constraint_resolution
    and _apply_constraint_resolution_scoring: only priority=="must" with
    decision=="NO", or a violating FORBIDDEN decision, cause hard
    exclusion - UNCERTAIN never does).

No LLM is used here. Every branch below is plain, inspectable Python -
see the Phase C task spec for the exact rules per family (quiet,
remote_work, family_friendly, and four single-claim families). Reasons
must describe evidence, never assert a guarantee (no "this hotel is
quiet" - only "soundproofing is listed; no guest noise evidence was
found").

Two-step design (kept deliberately separate, per the task's "separate
factual state from user desirability" requirement):

1. compute_family_factual_state(): factual, direction-agnostic -
   SATISFIED / VIOLATED / UNRESOLVED for "is there enough evidence to
   keep this listing as a useful candidate for the positive semantic
   proposition a family represents" (SATISFIED), "is there a strong
   reason to exclude it" (VIOLATED), or neither (UNRESOLVED). Never
   inverted for FORBIDDEN - see _quiet_state's docstring for why this
   still starts from evidence being read literally.
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


_QUIET_LABELS: dict[str, str] = {"Q1": "soundproofing", "Q2": "quiet surroundings", "Q3": "guest noise reports"}


def _quiet_state(claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    """
    Search-narrowing policy, not a certification of silence (see module
    docstring). Q3's evidence is now attribution-fixed at the routing
    boundary (app.logic.soft_evidence_routing.CLAIMS_REQUIRING_GUEST_REPORTED_EVIDENCE)
    - only genuine guest-review text can ever reach Q3 as SUPPORT/CONTRADICT,
    so "Quiet street view" (a facility/description claim) can no longer
    falsely support it.

    - YES: at least one of Q1/Q2/Q3 is a confirmed positive quiet signal,
      AND neither Q2 nor Q3 is CONTRADICT or MIXED. A single positive
      signal (e.g. soundproofing alone) is a useful candidate signal,
      not proof of silence - the reason says exactly that.
    - NO: Q2 or Q3 is explicitly CONTRADICT AND there is no positive
      quiet signal anywhere to offset it.
    - UNCERTAIN: everything else - including a positive signal that
      coexists with a contradiction/MIXED elsewhere (conflict is
      preserved for the user, not silently resolved to NO), and the
      all-no-evidence case.
    """
    q1, q2, q3 = _claim(claims_by_id, "Q1"), _claim(claims_by_id, "Q2"), _claim(claims_by_id, "Q3")
    notes = _notes_for(claims_by_id, ("Q1", "Q2", "Q3"))
    claim_pairs = (("Q1", q1), ("Q2", q2), ("Q3", q3))

    supporting = tuple(cid for cid, c in claim_pairs if _relation_for_satisfaction(c) == ClaimRelation.SUPPORT)
    has_support = bool(supporting)

    contradicted = tuple(
        cid for cid, c in (("Q2", q2), ("Q3", q3)) if _relation_for_violation(c) == ClaimRelation.CONTRADICT
    )
    blocking = _raw_relation(q2) in (ClaimRelation.MIXED, ClaimRelation.CONTRADICT) or _raw_relation(q3) in (
        ClaimRelation.MIXED,
        ClaimRelation.CONTRADICT,
    )

    if contradicted and not has_support:
        reason = (
            f"{'/'.join(_QUIET_LABELS[c] for c in contradicted)} explicitly report noise; "
            "no positive quiet evidence was found to offset it"
        )
        return _FamilyOutcome("quiet", FactualState.VIOLATED, contradicted, notes, reason)

    if has_support and not blocking:
        reason = f"{'; '.join(_QUIET_LABELS[c] for c in supporting)} support a quiet match; no contradicting evidence was found"
        return _FamilyOutcome("quiet", FactualState.SATISFIED, supporting, notes, reason)

    if has_support and (contradicted or blocking):
        conflicting = tuple(cid for cid in ("Q2", "Q3") if _raw_relation(claims_by_id.get(cid)) in (
            ClaimRelation.MIXED, ClaimRelation.CONTRADICT,
        ))
        decisive = tuple(sorted(set(supporting) | set(conflicting)))
        reason = (
            f"{'; '.join(_QUIET_LABELS[c] for c in supporting)} suggest a quiet match, but "
            f"{'; '.join(_QUIET_LABELS[c] for c in conflicting)} conflicts with it - not enough to "
            "confidently call this a quiet match"
        )
        return _FamilyOutcome("quiet", FactualState.UNRESOLVED, decisive, notes, reason)

    return _FamilyOutcome(
        "quiet", FactualState.UNRESOLVED, (), notes,
        "no conclusive evidence about noise or quiet surroundings was found",
    )


_REMOTE_WORK_LABELS: dict[str, str] = {
    "RW1": "a desk/workspace", "RW2A": "Wi-Fi availability", "RW2B": "connection reliability",
}


def _remote_work_state(claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    """
    RW1 (desk) and RW2A (Wi-Fi) are the essential signals for "good
    candidate for remote work" - RW2B (reliability) is a bonus, never
    required: desk + Wi-Fi with unknown reliability is exactly the kind
    of useful candidate this system should surface, with a reason that
    says reliability was not established (never that it IS reliable).

    - YES: RW1 SUPPORT AND RW2A SUPPORT AND RW2B is not CONTRADICT/MIXED
      (RW2B may be SUPPORT or NOT_ENOUGH_EVIDENCE).
    - NO: RW1 or RW2A explicitly CONTRADICT, AND no positive remote-work
      evidence anywhere to offset it.
    - UNCERTAIN: only one of RW1/RW2A is supported, or there is
      conflicting evidence (e.g. an essential CONTRADICT alongside a
      positive signal elsewhere, or RW2B CONTRADICT/MIXED despite RW1+RW2A
      SUPPORT).
    """
    rw1, rw2a, rw2b = (
        _claim(claims_by_id, "RW1"),
        _claim(claims_by_id, "RW2A"),
        _claim(claims_by_id, "RW2B"),
    )
    notes = _notes_for(claims_by_id, ("RW1", "RW2A", "RW2B"))

    rw1_support = _relation_for_satisfaction(rw1) == ClaimRelation.SUPPORT
    rw2a_support = _relation_for_satisfaction(rw2a) == ClaimRelation.SUPPORT
    rw2b_support = _relation_for_satisfaction(rw2b) == ClaimRelation.SUPPORT
    rw2b_raw = _raw_relation(rw2b)

    if rw1_support and rw2a_support and rw2b_raw not in (ClaimRelation.MIXED, ClaimRelation.CONTRADICT):
        decisive = ("RW1", "RW2A", "RW2B") if rw2b_support else ("RW1", "RW2A")
        if rw2b_support:
            reason = "a desk/workspace, Wi-Fi, and additional evidence of reliable connectivity are all supported"
        else:
            reason = "a desk/workspace and Wi-Fi are supported; connection reliability was not established"
        return _FamilyOutcome("remote_work", FactualState.SATISFIED, decisive, notes, reason)

    essential_contradicted = tuple(
        cid for cid, c in (("RW1", rw1), ("RW2A", rw2a)) if _relation_for_violation(c) == ClaimRelation.CONTRADICT
    )
    support_by_id = {"RW1": rw1_support, "RW2A": rw2a_support, "RW2B": rw2b_support}
    supporting = tuple(cid for cid in ("RW1", "RW2A", "RW2B") if support_by_id[cid])
    has_any_support = bool(supporting)

    if essential_contradicted and not has_any_support:
        reason = f"{'/'.join(_REMOTE_WORK_LABELS[c] for c in essential_contradicted)} is explicitly contradicted; no positive remote-work evidence was found to offset it"
        return _FamilyOutcome("remote_work", FactualState.VIOLATED, essential_contradicted, notes, reason)

    if essential_contradicted:
        reason = (
            f"{'/'.join(_REMOTE_WORK_LABELS[c] for c in essential_contradicted)} is contradicted, but "
            f"{'/'.join(_REMOTE_WORK_LABELS[c] for c in supporting)} is supported - conflicting evidence"
        )
        decisive = tuple(sorted(set(essential_contradicted) | set(supporting)))
        return _FamilyOutcome("remote_work", FactualState.UNRESOLVED, decisive, notes, reason)

    if supporting:
        reason = f"only {'/'.join(_REMOTE_WORK_LABELS[c] for c in supporting)} is confirmed - not enough to confidently call this a good remote-work match"
        return _FamilyOutcome("remote_work", FactualState.UNRESOLVED, supporting, notes, reason)

    return _FamilyOutcome(
        "remote_work", FactualState.UNRESOLVED, (), notes,
        "no evidence establishes a desk/workspace, Wi-Fi, or connection reliability",
    )


def _family_friendly_state(claims_by_id: dict[str, AtomicClaimResult]) -> _FamilyOutcome:
    """
    FC1 (formal policy accommodating children) carries more weight than
    FAM1 (general family-suitability description): an explicit FC1
    CONTRADICT is treated as a clear exclusion signal regardless of
    FAM1, matching how a booking policy that plainly excludes children
    is a stronger signal than descriptive marketing text.

    - NO: FC1 explicitly CONTRADICT.
    - YES: FC1 or FAM1 SUPPORT, with neither MIXED and FAM1 not
      CONTRADICT.
    - UNCERTAIN: everything else (including FAM1 CONTRADICT alone, or
      no evidence at all).
    """
    fc1, fam1 = _claim(claims_by_id, "FC1"), _claim(claims_by_id, "FAM1")
    notes = _notes_for(claims_by_id, ("FC1", "FAM1"))

    if _relation_for_violation(fc1) == ClaimRelation.CONTRADICT:
        return _FamilyOutcome(
            "family_friendly", FactualState.VIOLATED, ("FC1",), notes,
            "listing evidence explicitly indicates children are not accommodated",
        )

    support_by_id = {
        "FC1": _relation_for_satisfaction(fc1) == ClaimRelation.SUPPORT,
        "FAM1": _relation_for_satisfaction(fam1) == ClaimRelation.SUPPORT,
    }
    supporting = tuple(cid for cid in ("FC1", "FAM1") if support_by_id[cid])
    fam1_contradicted = _relation_for_violation(fam1) == ClaimRelation.CONTRADICT
    blocking = _raw_relation(fc1) == ClaimRelation.MIXED or _raw_relation(fam1) in (
        ClaimRelation.MIXED, ClaimRelation.CONTRADICT,
    )

    if supporting and not blocking:
        reason = "formal child-accommodation policy and/or family suitability are supported by listing evidence"
        return _FamilyOutcome("family_friendly", FactualState.SATISFIED, supporting, notes, reason)

    if supporting and blocking:
        conflicting = ("FAM1",) if fam1_contradicted or _raw_relation(fam1) == ClaimRelation.MIXED else ()
        decisive = tuple(sorted(set(supporting) | set(conflicting)))
        return _FamilyOutcome(
            "family_friendly", FactualState.UNRESOLVED, decisive, notes,
            "some family-friendly evidence exists but conflicts with other evidence - not enough to confidently call this a match",
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

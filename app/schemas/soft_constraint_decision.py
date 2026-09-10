"""
Typed deterministic representation of the result of resolving one
original UserConstraint from the new soft-evidence pipeline's
AtomicClaimResult[] (Phase C - the bridge from SoftPreferenceEvidence
to a downstream-consumable decision).

PRODUCT CONTRACT: the END-TO-END OUTCOME (after downstream's existing
priority-based interpretation) is a candidate-matching signal, not a
certification - a kept/rewarded DESIRED preference means "enough
evidence to keep this listing as a good candidate", an excluded/
penalized FORBIDDEN preference means "a strong reason not to consider
this listing a candidate", and neither means "insufficient or
materially conflicting evidence" (not a failure state).

`decision` ITSELF is narrower and direction-agnostic: it always answers
"is the underlying positive proposition confirmed by evidence" -
YES/NO/UNCERTAIN do NOT get pre-inverted for FORBIDDEN here (the
existing downstream contract - app.logic.listing_evaluation.
_fails_constraint_resolution / _apply_constraint_resolution_scoring -
already applies that inversion itself, from `priority`). See
app.logic.soft_constraint_decision_policy's module docstring and
apply_preference_direction's docstring for the full contract, including
the audit trail of a since-fixed double-inversion bug.

Deliberately separate from app.schemas.soft_evidence: that schema
stays factual (evidence relations are never inverted for FORBIDDEN -
see app.logic.soft_preference_decomposition.PreferenceDirection's own
docstring). `preference_direction` is still recorded on
SoftConstraintDecision below for telemetry/UI provenance even though it
does not currently change `decision`'s value.

No probabilistic confidence is ever fabricated here - `confidence` is
always None. The deterministic policy that reaches `decision` is pure
Python (app.logic.soft_constraint_decision_policy), not an LLM.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.logic.soft_preference_decomposition import PreferenceDirection
from app.schemas.constraints import ConstraintPriority
from app.schemas.soft_evidence import ClaimRelation


class FactualState(str, Enum):
    """
    Intermediate, direction-agnostic state of the positive semantic
    proposition a family represents (e.g. Q1/Q2/Q3 -> "the property is
    a good quiet candidate"). PreferenceDirection is applied on top of
    this, once, at the constraint boundary - never before (see
    apply_preference_direction in app.logic.soft_constraint_decision_policy).

    SATISFIED: enough evidence exists to keep this listing as a useful
        candidate for the proposition - not proof the property
        objectively has it.
    VIOLATED: evidence gives a strong reason to exclude this listing
        for the proposition.
    UNRESOLVED: evidence is insufficient or materially conflicting.
    """

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNRESOLVED = "unresolved"


class ConstraintDecisionValue(str, Enum):
    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


class SoftDecisionEvidenceRef(BaseModel):
    """One decisive evidence snippet that drove a family's factual state."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    evidence_text: str
    relation: ClaimRelation
    source_type: str | None = None
    source_path: str | None = None


class SoftConstraintDecision(BaseModel):
    """
    Deterministic decision for one original UserConstraint, derived from
    one hotel's SoftPreferenceEvidence via app.logic.soft_constraint_decision_policy.decide_constraint.

    families / claim_ids describe everything CONSULTED; decisive_claim_ids
    / evidence are the subset that actually determined factual_state -
    the rest were consulted but did not decisively contribute (e.g. an
    UNRESOLVED family, or a claim whose relation didn't match any rule
    branch).
    """

    model_config = ConfigDict(extra="forbid")

    constraint_id: str | None = None
    raw_text: str
    normalized_text: str
    priority: ConstraintPriority

    families: list[str] = Field(default_factory=list)
    preference_direction: PreferenceDirection

    factual_state: FactualState
    decision: ConstraintDecisionValue
    reason: str

    claim_ids: list[str] = Field(default_factory=list)
    decisive_claim_ids: list[str] = Field(default_factory=list)
    evidence: list[SoftDecisionEvidenceRef] = Field(default_factory=list)

    # Provenance for debugging (e.g. "Q2: retrieval failed", "Q3: 1
    # evidence item(s) unresolved (verification_failed)") - never used
    # to silently convert a technical failure into a factual verdict.
    technical_notes: list[str] = Field(default_factory=list)

    # Never fabricated - this is a deterministic decision, not a model
    # score. Kept for downstream schema compatibility only.
    confidence: float | None = None

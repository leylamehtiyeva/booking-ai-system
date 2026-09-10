from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.config.llm import get_semantic_verifier_model


class SemanticVerifierPolicy(BaseModel):
    """
    Configuration for the Gemini semantic evidence verifier
    (app.logic.semantic_evidence_verification). Deliberately separate
    from FallbackPolicy - the old textual fallback and the new evidence
    pipeline are independent until the migration in the README's plan
    reaches the removal step.

    max_calls_per_hotel / max_calls_per_request are constructor-args
    only - not read from any environment variable. They are
    conservative, unvalidated starting points meant to be revised once
    real shadow-mode call volume is observed, not derived from any
    production load data (none exists yet for this pipeline).

    max_calls_per_hotel is the BASE per-hotel budget; the Phase B
    orchestrator (soft_evidence_orchestration.py) computes an
    *effective* per-hotel budget from it - min(12, max(base, number of
    active claims needing Gemini for that hotel)) - so a hotel with
    more than 6 genuinely active claims is not silently starved.
    max_calls_per_request defaults to 60 (5 shadow hotels x 12 possible
    claims each, the true worst case for that effective-budget formula)
    specifically so it functions as a safety ceiling that does not
    itself become an earlier, cross-hotel-starvation-causing cutoff
    under the standard 5-hotel shadow scope.

    semantic_verifier_max_concurrency bounds how many APPROVED Gemini
    verifier calls may be in flight at once - a pure performance knob.
    It is only ever applied to the already-planned, already-budget-
    approved task list (see soft_evidence_orchestration._plan_verifier_tasks) -
    it never changes which candidates get approved vs SKIPPED_CALL_LIMIT,
    the per-hotel/request budgets, or the candidate_rank -> claim ->
    hotel fairness order those budgets were spent in. Conservative
    default, same reasoning as embedding_max_concurrency.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model: str = Field(default_factory=get_semantic_verifier_model)
    temperature: float = 0.0

    max_calls_per_hotel: int = 6
    max_calls_per_request: int = 60
    semantic_verifier_max_concurrency: int = 3

    def normalized_max_calls_per_hotel(self) -> int:
        return max(0, self.max_calls_per_hotel)

    def normalized_max_calls_per_request(self) -> int:
        return max(0, self.max_calls_per_request)

    def normalized_semantic_verifier_max_concurrency(self) -> int:
        return max(1, self.semantic_verifier_max_concurrency)

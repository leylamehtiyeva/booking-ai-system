from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.logic.soft_evidence_retrieval import DEFAULT_EMBEDDING_MODEL


class SoftEvidencePipelinePolicy(BaseModel):
    """
    Configuration for the shadow-mode soft-evidence pipeline itself
    (scope/retrieval), separate from SemanticVerifierPolicy (which is
    scoped to the Gemini verifier specifically).

    shadow_hotel_top_k: how many of the top-ranked listings the pipeline
    runs on - the SAME bounded subset the old textual fallback already
    targets, so old vs new evidence is comparable for the same hotels.

    retrieval_top_k=3 is a provisional shadow-mode value: K=2 is already
    enough to observe MIXED evidence, but heading filtering/dedup run
    AFTER retrieval and can shrink the effective candidate count: K=3
    gives a small amount of headroom without moving to K=5, where the
    research measured more false-SUPPORT exposure.

    embedding_model is fixed to gemini-embedding-001 (the model
    validated by the experiments) - not configurable to a different
    model in this migration.

    enabled defaults to True on a bare SoftEvidencePipelinePolicy() -
    but app.logic.listing_evaluation.evaluate_listings never
    constructs a bare one implicitly; its own internal default
    (_build_soft_evidence_pipeline_policy) is enabled=False, so every
    existing/unaware caller keeps getting zero embedding/Gemini calls
    unless it explicitly opts in.

    embedding_max_concurrency bounds how many embedding batch calls
    (the shared query batch + every hotel's evidence-pool batches) may
    be in flight at once - a performance knob only. It never changes
    which batches exist, their content, or their call count; it only
    changes how many of those already-planned, independent network
    calls run concurrently. Conservative default - not derived from
    any measured production load, since none exists yet for this
    pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    shadow_hotel_top_k: int = 5
    retrieval_top_k: int = 3
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_max_concurrency: int = 3

    def normalized_shadow_hotel_top_k(self) -> int:
        return max(0, self.shadow_hotel_top_k)

    def normalized_retrieval_top_k(self) -> int:
        return max(0, self.retrieval_top_k)

    def normalized_embedding_max_concurrency(self) -> int:
        return max(1, self.embedding_max_concurrency)

"""
Post-retrieval cleanup + strict routing for one atomic claim's
retrieved candidates.

Validated order, ported/adapted from
evaluation/experiments/pipeline_cleanup_v2/{heading_rules,deduplication}.py
(not imported from evaluation/experiments/):

    retrieval -> heading filter -> exact-text dedup -> strict routing

Both filters apply to the already-retrieved top-K candidates for one
claim, not to the raw pre-retrieval pool - this matches the validated
experiment order exactly (confirmed by inspecting
pipeline_cleanup_v2/run_end_to_end_verification_v2.py /
run_strict_structured_routing.py directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.logic.soft_evidence_retrieval import RetrievedEvidence
from app.logic.soft_evidence_routing import resolve_deterministic_claim
from app.schemas.soft_evidence import EvidenceItem


def is_heading(path: str | None) -> bool:
    """Ported unchanged from pipeline_cleanup_v2/heading_rules.py."""
    if not path:
        return False
    return path.endswith(".title") or path.endswith(".header")


@dataclass(frozen=True)
class DedupedEvidence:
    """
    One evidence candidate after exact-text dedup. duplicate_source_paths
    preserves provenance for debugging (which other paths carried the
    same text) without adding a field to the public EvidenceItem schema -
    this is an internal cleanup-stage structure only.
    """

    text: str
    source_type: str
    source_path: str | None
    retrieval_score: float
    duplicate_source_paths: tuple[str, ...] = field(default_factory=tuple)


def filter_headings(candidates: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    return [c for c in candidates if not is_heading(c.source_path)]


def deduplicate_by_text(candidates: list[RetrievedEvidence]) -> list[DedupedEvidence]:
    """
    candidates: assumed already ordered best-first (retrieve_top_k's
    output order, i.e. retrieval_score descending). Keeps the first
    (best-scored) occurrence per exact text as canonical; every other
    occurrence's source_path is recorded in duplicate_source_paths.
    """
    order: list[str] = []
    groups: dict[str, list[RetrievedEvidence]] = {}

    for candidate in candidates:
        if candidate.text not in groups:
            groups[candidate.text] = []
            order.append(candidate.text)
        groups[candidate.text].append(candidate)

    deduped: list[DedupedEvidence] = []
    for text in order:
        group = groups[text]
        canonical = group[0]
        duplicate_paths = tuple(c.source_path for c in group[1:] if c.source_path is not None)
        deduped.append(
            DedupedEvidence(
                text=canonical.text,
                source_type=canonical.source_type,
                source_path=canonical.source_path,
                retrieval_score=canonical.retrieval_score,
                duplicate_source_paths=duplicate_paths,
            )
        )

    return deduped


@dataclass(frozen=True)
class RoutedClaimCandidates:
    """
    Result of clean_and_route_candidates for one claim: candidates
    already resolved deterministically (free - no Gemini call), and
    genuine free-text candidates still needing verification, in
    retrieval-rank order (best-first) ready for the verifier scheduler.
    """

    deterministic_items: list[EvidenceItem]
    free_text_candidates: list[DedupedEvidence]


def clean_and_route_candidates(
    claim_id: str,
    candidates: list[RetrievedEvidence],
) -> RoutedClaimCandidates:
    """
    retrieval (candidates, already top-K) -> heading filter -> dedup ->
    strict routing. Each cleaned candidate is routed independently: a
    deterministic NOT_ENOUGH_EVIDENCE resolution for one candidate never
    prevents a different, genuine free-text candidate for the same
    claim from separately reaching the verifier - resolve_deterministic_claim
    is pure/stateless per candidate (Phase A, unchanged).
    """
    cleaned = deduplicate_by_text(filter_headings(candidates))

    deterministic_items: list[EvidenceItem] = []
    free_text_candidates: list[DedupedEvidence] = []

    for candidate in cleaned:
        deterministic_item = resolve_deterministic_claim(
            claim_id=claim_id,
            source_type=candidate.source_type,
            source_path=candidate.source_path,
            text=candidate.text,
            retrieval_score=candidate.retrieval_score,
        )
        if deterministic_item is not None:
            deterministic_items.append(deterministic_item)
        else:
            free_text_candidates.append(candidate)

    return RoutedClaimCandidates(
        deterministic_items=deterministic_items,
        free_text_candidates=free_text_candidates,
    )

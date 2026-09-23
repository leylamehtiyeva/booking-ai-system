from __future__ import annotations

import numpy as np


def cosine_similarity(
    query: list[float],
    candidates: list[list[float]],
) -> list[float]:
    if not candidates:
        return []

    q = np.asarray(query, dtype=np.float64)
    c = np.asarray(candidates, dtype=np.float64)

    q_norm = np.linalg.norm(q)
    c_norms = np.linalg.norm(c, axis=1)

    denom = c_norms * q_norm
    denom = np.where(denom == 0, 1e-12, denom)

    similarities = (c @ q) / denom
    return similarities.tolist()


def top_k(
    query: list[float],
    candidates: list[list[float]],
    k: int,
) -> list[tuple[int, float]]:
    """
    Return (index, score) pairs for the k most similar candidates,
    ranked highest similarity first.
    """
    similarities = cosine_similarity(query, candidates)
    ranked = sorted(
        enumerate(similarities),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[: max(0, k)]

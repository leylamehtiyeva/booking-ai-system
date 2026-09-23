"""
Heading-detection rule, split into its own zero-heavy-dependency module
so it can be imported from the isolated nli_oracle venv (phase 2,
verification) without pulling in pydantic via atomic_chunking.py's
build_evidence_chunks import chain (phase 1, retrieval, main venv only).
"""

from __future__ import annotations


def is_heading(path: str | None) -> bool:
    if not path:
        return False
    return path.endswith(".title") or path.endswith(".header")

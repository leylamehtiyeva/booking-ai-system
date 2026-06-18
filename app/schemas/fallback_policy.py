from __future__ import annotations

from pydantic import BaseModel

from app.config.llm import get_gemini_model

class FallbackPolicy(BaseModel):
    enabled: bool = True 
    top_k: int = 5
    must_only: bool = True

    run_for_unresolved: bool = True
    run_for_structured_uncertain: bool = True

    max_constraints_per_listing: int = 3

    model: str = get_gemini_model()

    def normalized_top_k(self) -> int:
        return max(0, self.top_k)

    def normalized_max_constraints_per_listing(self) -> int:
        return max(0, self.max_constraints_per_listing)
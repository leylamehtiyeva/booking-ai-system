import recorded from "../src/mocks/turns.json";
import type { TurnResult } from "../src/lib/types";

// Responses recorded from the real backend (scripts/generate_mocks.py).
type Step = { request: string; response: TurnResult };
export const scenarios = recorded as unknown as Record<string, Step[]>;
export const [SEARCH, REFINE, QUESTION] = scenarios.search_refine_ask;
export const [CLARIFY, CLARIFIED] = scenarios.budget_clarification;

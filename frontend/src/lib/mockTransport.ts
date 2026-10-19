// Development / test only: replays responses recorded from the real backend
// (scripts/generate_mocks.py -> src/mocks/turns.json). Never used by a
// production build - see DemoApp's transport selection.
import recorded from "../mocks/turns.json";
import type { ChatTransport } from "./api";
import type { TurnResult } from "./types";

type Step = { request: string; response: TurnResult };
const scenarios = recorded as unknown as Record<string, Step[]>;

const normalize = (text: string) => text.trim().toLowerCase().replace(/\s+/g, " ");

export const MOCK_PROMPTS: string[] = Object.values(scenarios).flat().map((step) => step.request);

export function mockTransport(delayMs = 600): ChatTransport {
  return {
    async send(message) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      const step = Object.values(scenarios)
        .flat()
        .find((s) => normalize(s.request) === normalize(message));
      if (step) return step.response;
      const fallback = Object.values(scenarios)[0][0].response;
      return {
        ...fallback,
        assistant_message:
          "Mock mode only replays recorded example prompts. Run the FastAPI backend to try your own message.",
        outcome_kind: "informational",
        conversation_action: null,
        listing_cards: [],
        inspector: { ...fallback.inspector, results: [], listing_query: [], notes: ["mock mode"] },
      };
    },
  };
}

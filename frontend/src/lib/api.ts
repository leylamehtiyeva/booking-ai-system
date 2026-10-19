import type { TurnResult } from "./types";

export const MAX_MESSAGE_LENGTH = 2000; // app/api/main.py

export type ApiErrorKind =
  | "session_expired" // 404 session_not_found: server restarted or session idle too long
  | "invalid_message" // 422
  | "server_error" // 500 and other non-2xx
  | "network"; // backend unreachable

export class ApiError extends Error {
  constructor(
    readonly kind: ApiErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface ChatTransport {
  send(message: string, sessionId: string | null): Promise<TurnResult>;
}

function detailMessage(body: unknown): string | null {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return null;
}

/** POST /api/chat on the FastAPI backend. */
export function httpTransport(baseUrl = "", fetchImpl: typeof fetch = fetch): ChatTransport {
  return {
    async send(message, sessionId) {
      let response: Response;
      try {
        response = await fetchImpl(`${baseUrl}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sessionId ? { message, session_id: sessionId } : { message }),
        });
      } catch {
        throw new ApiError("network", "The demo server can't be reached right now. Please try again in a moment.");
      }

      if (response.ok) return (await response.json()) as TurnResult;

      const body = await response.json().catch(() => null);
      if (response.status === 404) {
        throw new ApiError(
          "session_expired",
          "This conversation has expired - the demo server keeps conversations in memory only. Your message was not sent.",
        );
      }
      if (response.status === 422) {
        throw new ApiError("invalid_message", `Please write a message of up to ${MAX_MESSAGE_LENGTH} characters.`);
      }
      throw new ApiError(
        "server_error",
        detailMessage(body) ?? "Something went wrong while processing your message. Please try again.",
      );
    },
  };
}

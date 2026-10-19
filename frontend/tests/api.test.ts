import { describe, expect, it, vi } from "vitest";
import { ApiError, httpTransport } from "../src/lib/api";
import { SEARCH } from "./fixtures";

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("httpTransport", () => {
  it("posts the message and returns the TurnResult", async () => {
    const fetchImpl = vi.fn(async () => json(200, SEARCH.response));
    const result = await httpTransport("", fetchImpl).send("hello", null);

    expect(result.session_id).toBe(SEARCH.response.session_id);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/chat");
    expect(JSON.parse(String(init.body))).toEqual({ message: "hello" });
  });

  it("sends the session id to continue a conversation", async () => {
    const fetchImpl = vi.fn(async () => json(200, SEARCH.response));
    await httpTransport("http://api.test", fetchImpl).send("more", "abc");

    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://api.test/api/chat");
    expect(JSON.parse(String(init.body))).toEqual({ message: "more", session_id: "abc" });
  });

  it.each([
    [404, { detail: { code: "session_not_found" } }, "session_expired"],
    [422, { detail: [] }, "invalid_message"],
    [500, { detail: { code: "internal_error", message: "Something went wrong" } }, "server_error"],
    [502, null, "server_error"],
  ])("maps HTTP %s to %s", async (status, body, kind) => {
    const transport = httpTransport("", vi.fn(async () => json(status, body)));
    await expect(transport.send("x", "abc")).rejects.toMatchObject({ kind });
  });

  it("reports an unreachable backend", async () => {
    const transport = httpTransport("", vi.fn(async () => Promise.reject(new TypeError("fetch failed"))));
    const error = await transport.send("x", null).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).kind).toBe("network");
  });
});

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import DemoApp, { STARTER_PROMPTS } from "../src/components/demo/DemoApp";
import { MessageText } from "../src/components/demo/MessageText";
import { ApiError, type ChatTransport } from "../src/lib/api";
import type { TurnResult } from "../src/lib/types";
import { CLARIFY, QUESTION, REFINE, SEARCH } from "./fixtures";

type Reply = TurnResult | ApiError | Promise<TurnResult>;

function scripted(...replies: Reply[]) {
  const calls: { message: string; sessionId: string | null }[] = [];
  const transport: ChatTransport = {
    send: vi.fn(async (message: string, sessionId: string | null) => {
      calls.push({ message, sessionId });
      const reply = replies.shift();
      if (!reply) throw new Error("unexpected call");
      if (reply instanceof ApiError) throw reply;
      return reply;
    }),
  };
  return { transport, calls };
}

async function sendText(text: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Message"), `${text}{Enter}`);
}

describe("DemoApp", () => {
  it("shows a loading state, then the answer and listing cards from the API", async () => {
    let resolve!: (turn: TurnResult) => void;
    const { transport } = scripted(new Promise<TurnResult>((r) => (resolve = r)));
    render(<DemoApp transport={transport} />);

    await userEvent.setup().click(screen.getByRole("button", { name: STARTER_PROMPTS[0] }));

    expect(screen.getByRole("status").textContent).toMatch(/checking listings/i);
    resolve(SEARCH.response);

    const cards = await screen.findAllByRole("article");
    expect(cards).toHaveLength(SEARCH.response.listing_cards.length);
    const first = within(cards[0]);
    const card = SEARCH.response.listing_cards[0];
    expect(first.getByRole("heading", { name: card.title })).toBeTruthy();
    expect(first.getByText(card.price_summary!)).toBeTruthy();
    expect(first.getByText(/Confirmed · /)).toBeTruthy();
    expect(first.getByText(/Not confirmed · /)).toBeTruthy();
    expect(first.getByText("Optional")).toBeTruthy(); // "quiet street" was optional
    expect(first.getByText("Sample listing · no external page")).toBeTruthy();
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("continues the conversation with the session id the API returned", async () => {
    const { transport, calls } = scripted(SEARCH.response, REFINE.response, QUESTION.response);
    render(<DemoApp transport={transport} />);

    await sendText(SEARCH.request);
    await screen.findAllByRole("article");
    await sendText(REFINE.request);
    await waitFor(() => expect(calls).toHaveLength(2));
    await sendText(QUESTION.request);
    await screen.findByText(QUESTION.response.assistant_message);

    expect(calls.map((c) => c.sessionId)).toEqual([null, SEARCH.response.session_id, REFINE.response.session_id]);
  });

  it("explains an expired session and restarts with the same message", async () => {
    const expired = new ApiError("session_expired", "This conversation has expired.");
    const { transport, calls } = scripted(SEARCH.response, expired, CLARIFY.response);
    render(<DemoApp transport={transport} />);

    await sendText(SEARCH.request);
    await screen.findAllByRole("article");
    await sendText("Only show places with a balcony.");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("expired");
    expect(screen.getByText("Not sent")).toBeTruthy();

    await userEvent.setup().click(within(alert).getByRole("button", { name: /new conversation/i }));
    await screen.findByText(CLARIFY.response.assistant_message);
    expect(calls.at(-1)).toEqual({ message: "Only show places with a balcony.", sessionId: null });
  });

  it("shows a server error with a retry that keeps the session", async () => {
    const { transport, calls } = scripted(SEARCH.response, new ApiError("server_error", "Something went wrong."), REFINE.response);
    render(<DemoApp transport={transport} />);

    await sendText(SEARCH.request);
    await screen.findAllByRole("article");
    await sendText(REFINE.request);
    const alert = await screen.findByRole("alert");
    await userEvent.setup().click(within(alert).getByRole("button", { name: "Try again" }));

    await waitFor(() => expect(calls).toHaveLength(3));
    expect(calls[2]).toEqual({ message: REFINE.request, sessionId: SEARCH.response.session_id });
  });

  it("shows a daily limit without offering to resend", async () => {
    const limit = new ApiError("rate_limited", "You have reached the demo's message limit for today.", false);
    const { transport, calls } = scripted(SEARCH.response, limit);
    render(<DemoApp transport={transport} />);

    await sendText(SEARCH.request);
    await screen.findAllByRole("article");
    await sendText(REFINE.request);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("message limit for today");
    expect(within(alert).queryByRole("button")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(SEARCH.response.listing_cards.length); // results kept
    expect(calls).toHaveLength(2);
  });

  it("puts the selected result's decision in the inspector", async () => {
    const { transport } = scripted(SEARCH.response);
    render(<DemoApp transport={transport} />);
    await sendText(SEARCH.request);
    const [first] = await screen.findAllByRole("article");

    await userEvent.setup().click(within(first).getByRole("button", { name: /why this result/i }));

    const inspector = screen.getByRole("complementary", { name: "Decision Inspector" });
    const decision = SEARCH.response.inspector.results[0];
    expect(within(inspector).getAllByText(decision.title).length).toBeGreaterThan(0);
    expect(within(inspector).getByText("Baku")).toBeTruthy();
    expect(within(inspector).getByText(/up to 150 USD per night/)).toBeTruthy();
  });

  it("does not send an empty or too long message", async () => {
    const { transport, calls } = scripted();
    render(<DemoApp transport={transport} />);

    await sendText("   ");
    const user = userEvent.setup();
    const box = screen.getByLabelText("Message");
    await user.click(box);
    await user.paste("x".repeat(2001));

    expect(screen.getByText(/please shorten/i)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true);
    expect(calls).toHaveLength(0);
  });
});

describe("MessageText", () => {
  it("renders links and bold text without injecting HTML", () => {
    const { container } = render(<MessageText text={"**Hi** <img src=x onerror=alert(1)>\n1. [Loft](https://example.org/a) [bad](javascript:alert(1))"} />);

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("strong")?.textContent).toBe("Hi");
    const links = container.querySelectorAll("a");
    expect(links).toHaveLength(1);
    expect(links[0].getAttribute("href")).toBe("https://example.org/a");
  });
});

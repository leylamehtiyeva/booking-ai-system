# Clarification Continuity V1

When the conversation router cannot tell whether a message changes the
search or asks about the shown results ("Only the ones with parking?"), it
abstains and the user is asked to clarify. V1 keeps the original request so
the user's answer can be applied to it on the next turn.

## What V1 supports

- `PendingClarification` (`app/schemas/conversation_route.py`) is application
  state created by Python (`app/logic/clarification_continuation.py`), never
  by the model. It stores the original message, the allowed actions
  (`update_search`, `listing_question`) and the shown result set id, and is
  persisted on `ConversationSession`.
- Created only for `ambiguous_action` when both readings can run (an existing
  search and a non-empty shown result set).
- On the next turn the router receives the pending clarification as input
  (`RouterInput.pending_clarification`) and returns the usual six-field
  decision plus one categorical field, `pending_resolution`
  (`PendingAwareConversationActionDecision`, pending turns only). Everything
  downstream still receives the six-field `ConversationActionDecision`.
  Replies are interpreted by the model; Python reads no message text.
- Continuing an update: Intent Update runs on the original message (as a
  confirmed search change), then on the reply with the original as read-only
  context, so refinements such as "parking is only a preference" apply.
- Continuing a listing question: factual extraction receives the original
  question and the reply as separate labelled inputs, so added facts (e.g. a
  pool) are kept. The answer is the existing per-hotel, per-fact verdict
  table; there is no AND/OR filtering.

## State transitions (`resolve_pending_transition`, `conversation_flow._process_message`)

| Router output on a pending turn | Result | Pending |
|---|---|---|
| `continue_pending` + resolved + allowed action | runs the action on the original request | cleared |
| `continue_pending` + listing + specific hotel that is invalid, missing, or marked `ambiguous_reference` | asks which shown hotel; nothing runs | kept |
| `continue_pending` that is inconsistent (unsure, or action not allowed) | asks again; nothing runs | kept |
| `cancel_pending` + `general_chat` + resolved | acknowledges, nothing changes | cleared |
| `cancel_pending` combined with any other action or status | asks again; nothing runs | kept |
| `unresolved` | asks again; nothing runs | kept |
| `new_request` | handled as an ordinary turn; the original is never applied | cleared, or replaced by a new ambiguous request |
| listing continuation whose result set changed or disappeared | asks the user to ask again | cleared |

## Failure policy

- Routing fails before any decision (timeout, provider error, invalid output,
  or a pending turn without `pending_resolution`): typed routing failure;
  pending, search and shown results are kept.
- Execution fails after a decision (Intent Update, factual extraction,
  retrieval, invalid dates during a continuation; also a failed new request):
  pending is cleared, search and shown results stay at the last accepted
  state, and the user is asked to send the complete request again. Nothing is
  retried with partial meaning and an abandoned request is never revived.

## Running the evaluation

```
uv run python -m evaluation.tasks.clarification_continuity.run --list
uv run python -m evaluation.tasks.clarification_continuity.run --dry-run   # no API calls
GEMINI_MODEL=gemini-2.5-flash uv run python -m evaluation.tasks.clarification_continuity.run --live --repeats 3 --max-calls 200
```

`--live` makes paid Gemini calls for the router, Intent Update, intent
extraction and factual extraction. Retrieval, fact matching and response
wording are faked. Cases: `evaluation/datasets/clarification_continuity/cases_v1.jsonl`
(23 cases, 24 turns). Each failed turn is attributed to the first failing
layer: infrastructure, routing (model), application (Python), extraction
(model). Outputs go to `evaluation/outputs/clarification_continuity/` (not
committed).

## Results

Final acceptance run (current code and prompt), gemini-2.5-flash, full suite,
24 turns x 3 repeats: **72/72 passed** (24/24 in each repeat), with no model,
application or infrastructure failures. 144 real API calls; median latency
4.5 s, p95 8.8 s per turn; estimated cost $0.133, excluding thinking tokens.
Results: `evaluation/outputs/clarification_continuity/live_20261102T144536Z/`.
This is success on the frozen evaluation suite, not a guarantee of
reliability on arbitrary user conversations.

Earlier full-suite runs, before the C14 application fix and the
pending-instruction scope rules, kept for comparison (24 turns x 3 repeats):

| Model | Passed | Failures by layer | Calls | Median / p95 latency per turn | Estimated cost |
|---|---|---|---|---|---|
| gemini-2.5-flash-lite | 63/72 | routing 6, extraction 2, infrastructure 1 | 149 | 2.3 s / 3.8 s | $0.038 |
| gemini-2.5-flash | 69/72 | application 2, infrastructure 1 | 144 | 4.7 s / 8.4 s | $0.129 |

Flash-Lite failures: a Russian "forget X, do Y instead" was treated as a
continuation (3/3); parking `must` became `nice` in 4 of 21 first Intent
Update calls; one unidentified hotel was guessed; one creation turn did not
abstain; one timeout. Flash: no routing or extraction errors; two failures
came from an application defect in C14 (fixed since) and one router timeout.

C14 ("check one of the hotels you showed me", then "Fairmont Baku Flame
Towers"), gemini-2.5-flash, 3 repeats:

| Code version | Passed | Notes |
|---|---|---|
| after the application fix, before the prompt fix | 3/6 | 1 router timeout; 1 widened to all hotels, which also failed turn 2 |
| after the pending-instruction scope rules | 6/6 | no timeouts; turn 1 always asked which hotel; turn 2 checked Fairmont only |

Estimated costs use a characters/4 token estimate and exclude thinking tokens,
so they understate gemini-2.5-flash spend.

## Evaluation limitations

- 24 turns, 3 repeats, mostly one shared fixture (one search, three hotels,
  one pending request about parking). This is an acceptance check, not a
  production reliability estimate.
- Fact concepts are scored by canonical field or `normalized_text` in the
  evaluator only.
- `GEMINI_MODEL` is shared with intent extraction, the textual evidence
  fallback and LLM answer generation; those stages were not evaluated on
  gemini-2.5-flash here. Response wording (`CONVERSATION_RESPONSE_LLM_PROFILE`)
  and the semantic verifier (`SEMANTIC_VERIFIER_MODEL`) are configured
  separately.

## Known limitations

- "Cheaper" without an amount does not reliably lower the budget: C04/C05
  passed only because parking was not carried over. Flash-Lite produced no
  change; Flash sometimes added a textual "cheaper" constraint.
- `start_search` can discard dates and constraints: C06/C09 accept it, and a
  new city then loses the earlier dates, Wi-Fi requirement and budget.
- `ambiguous_reference` does not create pending state on its own; only a
  reference left open inside an `ambiguous_action` continuation is kept.
- After an execution failure the user must restate the whole request.
- Reference selection depends on the model; Python validates that a target
  exists but cannot tell a guessed hotel from an identified one.
- Router latency on gemini-2.5-flash leaves limited headroom under the 15 s
  router timeout (2 timeouts in 78 Flash turns before the scope change;
  none in the final 72-turn run).

## Deferred

- Interpreting relative price changes ("cheaper") without an amount.
- Inheriting dates and constraints when a new search replaces the old one.
- General pending support for `ambiguous_reference`.
- Retrying a confirmed request after an execution failure without restating it.

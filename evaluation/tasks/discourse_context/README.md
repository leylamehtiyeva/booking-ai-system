# Structured Discourse Context V1

Experimental feature behind `TurnConfig.discourse_context` (default `False`:
behaviour, prompts and extractor inputs are then identical to the previous
version).

## What it adds

- **C2 - open reference questions.** A question about one shown listing
  that does not say which ("Does it have a gym?" with two results) becomes a
  `PendingClarification` with `kind=ambiguous_reference`. The answer ("the
  first one") resumes the original question
  (`clarification_continuation.resolve_reference_transition`).
- **C3 - discourse context.** After an answered question about shown
  listings, `DiscourseContext` (`app/schemas/conversation_route.py`) stores
  the result set id, scope, listing ids and asked facts. The router gets it
  as a "Conversation context" prompt section (plus an instruction section
  only on such turns); the factual extractor gets the earlier facts
  (`previous_facts`). Follow-ups that change only the listing or only the
  fact can then be resolved.
- **Lifecycle** (`conversation_flow._next_discourse_context`): created by an
  answered listing question; kept on clarifications, failures and questions
  without checkable facts; cleared by any chat, search change, new search or
  cancellation; dropped when its result set is no longer the shown one or a
  listing in it is missing. Listing ids are always validated against the
  shown set.
- Contracts: `ConversationActionDecision`, `PendingAwareConversationActionDecision`
  and `PendingClarification` are unchanged; `RouterInput` has one optional
  field, `discourse_context`.

## Evaluation

`run.py` runs the 12-dialogue development set
(`evaluation/datasets/discourse_context/dialogues_v1.jsonl`) for variant A
(flag off) and B (flag on) through `process_turn`, with deterministic
retrieval, matching and response wording:

```
python -m evaluation.tasks.discourse_context.run --list
python -m evaluation.tasks.discourse_context.run --dry-run --variant B     # no API calls
python -m evaluation.tasks.discourse_context.run --live --variant A --repeats 2 --max-calls 120
```

Development set, gemini-2.5-flash, 2 repeats: A 38/54 turns, B 54/54;
hypothesis turns A 0/16, B 16/16.

Independent holdout (24 natural dialogues, 110 user turns, full replay with
real state, gemini-2.5-flash, 2 repeats per variant, 801 LLM calls). The
holdout data, annotations and runner are kept outside the repository.

| Metric (2 repeats) | A | B |
|---|---|---|
| Supported C2/C3 target observations (11 scenarios x 2) | 4/22 | 15/22 |
| Supported targets passing in both repeats | 1/11 | 7/11 |
| Control turns | 68/80 | 65/80 (2 timeouts) |
| Safety turns | 8/16 | 10/16 |
| Stale references / wrong targets / wrong factual carryover | 0 | 0 |
| Unsafe outcomes with context or an open reference question as input | - | 0 |

These scores are provisional: the blinded manual review of factual-subject
scoring is still pending. Unsafe outcomes observed in both variants (search
changes on conditional or hypothetical statements, invented dates,
preferences turned into exclusions) come from routing and search extraction,
not from the discourse context.

## Known limitations

- Any intermediate chat or preference statement clears the context, so a
  follow-up after it loses the subject and focus (10 of 24 holdout target
  turns depend on this; 0 succeeded in either variant).
- A search change clears the subject even when it is still meaningful
  (e.g. late check-in after a date change).
- An open reference question is superseded by any unrelated turn; returning
  to it later is not supported.
- Resuming an open reference question was not exercised by the holdout.
- The factual extractor can prefer an incidental fact mentioned in the
  follow-up over the inherited one.
- Not supported: several intents in one turn, summaries, clarifying a topic
  or a missing detail, offer-level facts.

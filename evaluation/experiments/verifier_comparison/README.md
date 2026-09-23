# Semantic Evidence Verification for Soft Hotel Preferences

This document describes an experiment track that explores whether **semantic
evidence retrieval + verification** can improve, or partially replace, an
LLM fallback used to decide whether a hotel property satisfies a *soft*
guest preference (e.g. "quiet", "good breakfast", "good for remote work").

Everything described here lives under `evaluation/experiments/` and is
**research/evaluation code only**. Production code in `app/` was only ever
*read*, via a small number of read-only imports for schema/normalization
reuse (e.g. `app.retrieval.apify.normalize_apify_listing`,
`app.schemas.listing.ListingRaw`) — nothing in `app/` was changed by this
work.

---

## 1. Motivation

The booking assistant resolves two kinds of constraints:

- **Hard constraints** — dates, price, number of guests, property type.
  These are handled deterministically: a date range either overlaps
  availability or it doesn't; a price either fits the budget or it doesn't.
- **Soft preferences** — quiet, good breakfast, good for remote work,
  family-friendly, clean, comfortable beds, nightlife nearby. These cannot
  be resolved by exact matching: they require reading free text (property
  descriptions, policies, guest-review summaries) and judging whether it
  actually supports the claim.

The original system passed a relatively broad slice of property text/context
to an LLM fallback that made one large semantic decision per soft
preference (YES / NO / UNCERTAIN). This experiment track investigates a
narrower, evidence-grounded alternative and evaluates each of its
components in isolation before evaluating them together.

The key distinction this work is built around:

> **Embeddings answer:** "Which text is relevant to this preference?"
> **A verifier answers:** "Does this text actually support the claim?"

For example, *"Continental breakfast included"* is topically relevant to a
"good breakfast" preference — an embedding retriever will correctly surface
it — but it does not by itself support the *quality* claim "the breakfast is
good". Treating topical relevance as evidence is exactly the failure mode
this experiment track set out to measure and reduce.

---

## 2. Architecture evolution

### Original approach

```
broad soft preference → property text/context → LLM fallback → YES / NO / UNCERTAIN
```

Weaknesses observed once this was inspected closely (documented in detail
in later sections, with real examples):

- topic similarity gets confused with evidence ("desk" mentioned once ≠
  workspace suitability confirmed)
- unsupported YES: broad claims accepted on weak or absent textual support
- one LLM call carries the full burden of retrieval, verification, and
  aggregation at once, with no separation of concerns
- structured facts (`facilities[].name` tags) and free-text prose were
  treated the same way, even though the former has a known, controlled
  vocabulary and the latter does not

### New approach (this experiment track)

```
user soft preference
  → atomic decomposition (manual, controlled — see §3)
  → semantic retrieval (embeddings, top-K)
  → candidate evidence
  → heading filtering
  → exact-text deduplication
  → routing by source type
      controlled structured field + known mapping → deterministic SUPPORT
      controlled structured field + no mapping     → NOT_VERIFIED (NLI skipped)
      free natural-language text                    → semantic verifier (NLI / LLM)
  → evidence signals (SUPPORT / CONTRADICT / NOT_ENOUGH_EVIDENCE, or SUPPORT / NOT_VERIFIED)
  → (future) aggregation / ranking — explicitly NOT part of this experiment (see §18)
```

**Embeddings, semantic verification, deterministic routing, and ranking are
four different jobs**, evaluated separately in this track:

| Stage | Question it answers | Section |
|---|---|---|
| Embedding retrieval | Which candidate text is topically relevant to this atomic claim? | §4 |
| Semantic verification (NLI / LLM) | Does this specific candidate text actually support, contradict, or say nothing about the claim? | §5–§7, §11–§14 |
| Deterministic structured routing | Is this candidate a controlled structured tag with a known meaning, so verification can skip the semantic model entirely? | §9 |
| Hotel ranking (future work, not built here) | Given graded evidence across many soft preferences, which properties should be ranked higher? | §17–§18 |

---

## 3. Atomic decomposition

Broad preferences were deliberately decomposed into atomic sub-hypotheses,
each phrased as a single checkable claim. This decomposition was done
**manually/by design**, not learned or automated, so that decomposition
errors would not be mixed into retrieval or verifier error analysis.

Two families illustrate the pattern:

**Remote work:**
- `RW1`: "The room has a desk or dedicated workspace."
- `RW2A`: "Wi-Fi / internet access is available."
- `RW2B`: "Internet connection is reliable enough for remote work."

**Quiet:**
- `Q1`: "The room/property has soundproofing."
- `Q2`: "The immediate surroundings are explicitly described as quiet."
- `Q3`: "Guests report little or no significant noise disturbance."

The Q1/Q2/Q3 and `FC1` (sentence-level) split first appears in the NLI
oracle v2 input build (`evaluation/experiments/nli_oracle/build_input_v2.py`)
and is later **reused verbatim** (imported, not retyped) by the
architecture-v2 schema (`evaluation/experiments/pipeline_cleanup_v2/schema_v3.py`),
which only adds retrieval-query strings on top since the oracle experiments
never ran retrieval.

The full set of frozen hypothesis families used across this track's later
stages (`evaluation/experiments/verifier_comparison/build_benchmark.py`,
`HYPOTHESES` dict):

| ID | Hypothesis |
|---|---|
| Q1 | The room/property has soundproofing. |
| Q2 | The immediate surroundings are explicitly described as quiet. |
| Q3 | Guests report little or no significant noise disturbance. |
| FC1 | The property's policies formally accommodate children. |
| FAM1 | The property is suitable for families. |
| RW1 | The room has a desk or dedicated workspace. |
| RW2A | Wi-Fi / internet access is available. |
| RW2B | Internet connection is reliable enough for remote work. |
| BQ1 | The breakfast is good in quality. |
| CL1 | The property is clean. |
| NL1 | Nightlife is available nearby. |
| BC1 | The bed is comfortable. |

---

## 4. Retrieval experiments

**Scripts:** `evaluation/experiments/evidence_retrieval/run_retrieval_checkpoint.py`,
`run_retrieval_checkpoint_v2.py`
**Embedding model:** `gemini-embedding-001` (`evaluation/experiments/evidence_retrieval/embeddings.py`)
**Evidence chunking:** `evaluation/experiments/evidence_retrieval/chunking.py`
**Data:** 4 DEV properties, all in Baku (`evaluation/experiments/evidence_retrieval/golden/properties/`)
**Results (in repo):** `evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v1.jsonl`,
`retrieval_checkpoint_v2.jsonl`; tracked summary in
`evaluation/experiments/evidence_retrieval/results_summary.json`. **Full
raw reports (local only, gitignored):** `evaluation/outputs/evidence_retrieval_checkpoint_report.json`,
`evidence_retrieval_checkpoint_v2_report.json`.

Two retrieval checkpoints were run, one query per **broad** preference and
one query per **atomic** sub-hypothesis:

| Checkpoint | Query granularity | n | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|---|---|
| v1 | broad preference (1 query per constraint) | 16 pairs | 0.556 | 0.556 | 0.556 |
| v2 | atomic sub-hypothesis (1 query per Q1/Q2/RW1/…) | 32 subclaims | 0.929 | 1.0 | 1.0 |

Recall@1 nearly doubled (0.556 → 0.929) and Recall@3/5 reached 1.0 once
queries were split to match the atomic hypotheses instead of the broad
preference text. This is a purely qualitative-instinct confirmed
quantitatively: a single broad "quiet" query pulls in whichever quiet-adjacent
sentence ranks highest overall, while three atomic queries (soundproofing /
surroundings / guest reports) each retrieve the sentence that actually
matches that specific claim.

Independent of retrieval improving, embeddings alone cannot tell whether a
retrieved sentence is evidence *for* or *against* the claim, or just
*near* it — that judgment is deferred entirely to the verifier stage
(§5 onward). The checkpoint reports (`n_errors`, `score_distributions`) also
record concrete cases where a topically close sentence was retrieved with
high similarity but carried no actual semantic support — the motivating
observation for building a dedicated verifier evaluation.

---

## 5. Oracle NLI experiment

**Purpose:** isolate verification quality from retrieval quality by feeding
**manually-selected** evidence directly to the verifier, bypassing retrieval
entirely.

**Scripts:** `evaluation/experiments/nli_oracle/run_oracle_eval.py` (v1),
`run_oracle_eval_v2.py` (v2), `recompute_baseline_v1.py` (metrics-bugfix
recompute)
**Model:** `GuardrailsAI/finetuned_nli_provenance` (base model
`ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`), wrapped in
`evaluation/experiments/nli_oracle/model.py`. Label order
(`{0: entailment, 1: neutral, 2: contradiction}`) was verified against the
model's real `config.json`, not assumed. Premise/hypothesis order:
`tokenizer(premise, hypothesis, ...)`, `premise = evidence text`,
`hypothesis = subclaim text`, `max_length = 256`.
**Results — tracked summary (in repo):** `evaluation/experiments/nli_oracle/results_summary.json`
(`oracle_v1`, `oracle_baseline_v1_recompute`, `oracle_v2` stages). **Full
raw reports (local only, gitignored):** `evaluation/outputs/nli_oracle_report.json`,
`nli_oracle_baseline_v1_report.json`, `nli_oracle_v2_report.json`.

**oracle_v1** (45 chunk-level examples over the 4-property DEV set, using
the original — not yet atomic — hypothesis phrasing): accuracy 0.756
chunk-level. The model reached `entailment` recall 0.85 (good for SUPPORT)
but **neutral (`RELEVANT_NEUTRAL`) recall was 0.0** — every neutral-gold
example was predicted as `contradiction` (`neutral_to_contradict_rate = 1.0`
in `chunk_level_metrics`/`reliability_critical_metrics`). The DEV set had 0
real CONTRADICT examples at this point, so a true 3-class macro F1 is
undefined (`macro_f1_3class.insufficient_data = true`, documented directly
in the report).

**Macro-F1 bug found and corrected:** the original `oracle_v1` report
computed `macro_f1_observed_classes = 0.9189` — misleadingly high, because
it silently dropped the CONTRADICT class (0 gold examples) *and* treated
`RELEVANT_NEUTRAL`'s undefined precision as excluded rather than 0. The
corrected recompute (`oracle_baseline_v1_recompute`, using the
zero-division-fixed `evaluation/experiments/nli_oracle/metrics.py`)
gives `macro_f1_observed_classes = 0.4595` on the same 45 examples — the
same underlying predictions, a materially different and more honest
picture of quality. This same zero-division-fixed convention is reused
throughout every later report in this track (including the final verifier
comparison, §11–§13).

**oracle_v2** (89 chunk-level examples, atomic Q1/Q2/Q3 + FC1 hypotheses,
structured + free-text evidence mixed): chunk-level accuracy dropped to
0.472. The confusion matrix shows the same pattern in sharper relief: of 43
gold-neutral examples, **all 43** were predicted `contradiction`
(`neutral_to_contradict_rate = 1.0`). This holds even restricted to
structured-field evidence alone (`structured_vs_free_text.structured`:
26/26 SUPPORT correct, but all 17 neutral-gold structured examples
predicted CONTRADICT).

**Finding:** the baseline model is usable for entailment/SUPPORT detection
but is not a trustworthy 3-way classifier — it essentially cannot express
"this is unrelated / insufficient", only SUPPORT or CONTRADICT.

---

## 6. SUPPORT-only reinterpretation

**Script:** `evaluation/experiments/nli_oracle/analyze_support_verifier.py`
**Result — tracked summary (in repo):** `evaluation/experiments/nli_oracle/results_summary.json` →
`support_verifier_v2_analysis`. **Full raw report (local only, gitignored):**
`evaluation/outputs/nli_oracle_support_verifier_v2_analysis.json`.

Given §5's finding, the same already-collected `oracle_v2` predictions were
re-scored as a **binary** verifier instead of a 3-class one:

```
entailment            → VERIFIED_SUPPORT
neutral + contradiction → NOT_VERIFIED
```

Model-predicted `contradiction` was **not** trusted as real negative
evidence (§5 showed the model conflates neutral with contradiction), so
both non-entailment outcomes collapse into a single "not verified" bucket
rather than treating contradiction as an actionable negative signal.

Results (subclaim-level, 22 subclaim groups after excluding 7 with zero
retrieved evidence): **precision = 1.0, recall = 1.0, false_support_rate =
0.0, f1 = 1.0.** Chunk-level reference (89 raw examples, not deduplicated
to subclaims): precision 1.0, recall 0.913, f1 0.9545.

**This looks perfect — and that is exactly why it does not prove
end-to-end quality.** These numbers were computed over **oracle-selected**
evidence: the same evidence a human had manually picked as being
maximally relevant. It says nothing about what happens once retrieval
picks the candidates itself, where noisy, topically-close-but-not-actually-
supportive text becomes possible input (§7).

---

## 7. End-to-end verification experiment

**Scripts:** `evaluation/experiments/nli_oracle/build_retrieval_candidates.py`
(retrieval), `run_end_to_end_verification.py` (verification + aggregation)
**Pipeline:** atomic claim → embedding retrieval (unchanged model) → real
top-K candidates → NLI binary verifier (fresh inference, no oracle reuse)
→ OR aggregation across K
**Data:** `evaluation/experiments/nli_oracle/retrieval_candidates.jsonl`,
checkpoint source `retrieval_checkpoint_v2.jsonl` (32-subclaim schema)
**Results — tracked summary (in repo):** `evaluation/experiments/nli_oracle/results_summary.json` →
`end_to_end_v1`. **Full raw report (local only, gitignored):**
`evaluation/outputs/nli_end_to_end_verification_report.json`.

| K | precision (VERIFIED_SUPPORT) | recall | false_support_rate | accuracy |
|---|---|---|---|---|
| 1 | 0.917 | 0.786 | **0.056** | 0.875 |
| 3 | 0.800 | 0.857 | **0.167** | 0.844 |
| 5 | 0.706 | 0.857 | **0.278** | 0.781 |

**Key observation:** increasing K improves *access* to supporting evidence
(recall climbs slightly) but also increases false SUPPORT, because under OR
aggregation, any single incorrectly-verified candidate among K makes the
whole subclaim positive. False-support rate more than quadrupled from
K=1 (0.056) to K=5 (0.278).

Concrete false positives at K=3 (`false_positives_by_k["3"]`, real
retrieved candidates that were verified SUPPORT but should not have been):

- `"Non-smoking throughout"` → quiet
- `"Continental breakfast included"` → good breakfast
- `"Children and beds"` (a policy-section heading) → family-friendly

These three examples became the seed regression cases carried forward into
architecture v2 (§8–§9).

**Distinguishing failure types**, as this experiment made visible for the
first time:
- **Retrieval failure** — the correct evidence exists but wasn't retrieved
  in top-K (see `no_evidence_case_analysis` for `good_breakfast` and
  `remote_work_reliability_RW2b`: no genuine free-text evidence for these
  claims exists anywhere in the DEV corpus at all — a data gap, not a
  retrieval bug).
- **Input representation failure** — a heading (`"Children and beds"`) or a
  structured facility tag got treated as ordinary prose evidence.
- **Verifier failure** — the text was retrieved correctly but the NLI model
  misjudged the relation (`"Continental breakfast included"` is topically
  on-target for "good breakfast" but says nothing about quality).

---

## 8. Architecture cleanup

Following the §7 failure analysis, four targeted changes were introduced:

| Change | File | Why |
|---|---|---|
| Atomic Q1/Q2/Q3 quiet claims + sentence-level FC1 | `pipeline_cleanup_v2/schema_v3.py` (reuses `nli_oracle/build_input_v2.py`) | A single "quiet" hypothesis conflated three different kinds of evidence (physical soundproofing feature, described surroundings, guest-reported outcome); splitting them lets each be judged on its own literal semantics. |
| Heading filtering | `pipeline_cleanup_v2/heading_rules.py` | `"Children and beds"` (a `.title`/`.header` path) was retrieved and verified as if it were prose evidence. `is_heading()` excludes any candidate whose source path ends in `.title` or `.header`. |
| Exact-text deduplication | `pipeline_cleanup_v2/deduplication.py` | The same sentence (e.g. a repeated Booking template) could appear as multiple separate candidates from different chunk paths, each independently verified — wasting NLI calls and risking double-counted false positives. Keeps the best-ranked representative, records all original `chunk_id`s for traceability. |
| Deterministic mappings for controlled structured fields | `pipeline_cleanup_v2/deterministic_mapping.py` | A `facilities[].name` tag like `"Desk"` or `"Free WiFi"` has a known, controlled meaning — sending it through a semantic model to "guess" is unnecessary and adds a failure mode. `DETERMINISTIC_ALIASES` maps a small, dataset-verified alias list (`RW1: "desk"`, `RW2a: "free wifi"`, `FC2: "family rooms"`, `Q1: "soundproofing"/"soundproof rooms"`) directly to SUPPORT. |

**Results — tracked summary (in repo):** `evaluation/experiments/pipeline_cleanup_v2/results_summary.json`
→ `end_to_end_v2`; retrieval-only recall in
`evaluation/experiments/pipeline_cleanup_v2/retrieval_only_metrics_v2.json`
(recall@1 = recall@3 = 1.0, 40 subclaims) — also tracked, in repo. **Full
raw report (local only, gitignored):** `evaluation/outputs/nli_end_to_end_verification_v2_report.json`.

Candidate funnel (`end_to_end_v2.run_metadata.candidate_counts`): 200 raw
candidates → 189 after heading filter → 137 after dedup → 10 resolved
deterministically → **127 real NLI calls** (73 fewer than the naive 200).

| K | precision | recall | false_support_rate | accuracy |
|---|---|---|---|---|
| 1 | 0.941 | 1.0 | 0.042 | 0.975 |
| 3 | 0.842 | 1.0 | 0.125 | 0.925 |

**Methodological limitation, stated explicitly:** the subclaim schema grew
from 32 (§7, `end_to_end_v1`) to 40 (`end_to_end_v2`) between these two
runs — atomic quiet went from a single hypothesis to three, and other
claims were also refined. **v1 and v2 numbers above are directionally
informative but not a strict apples-to-apples comparison**; this is stated
in the underlying comparison report itself
(`pipeline_cleanup_v2/results_summary.json` → `v1_vs_v2_e2e_comparison`)
and repeated here rather than glossed over.

Within that caveat, the false-support-rate trend at matching K is
consistent with an improvement (K=1: 0.056 → 0.042; K=3: 0.167 → 0.125),
and it comes with a documented mechanism (heading filter + dedup +
deterministic routing), not just a number that moved.

---

## 9. Strict structured routing

**Script:** `evaluation/experiments/pipeline_cleanup_v2/run_strict_structured_routing.py`
**Rule module:** `evaluation/experiments/pipeline_cleanup_v2/structured_tag_rule.py`
**Results — tracked summary (in repo):** `evaluation/experiments/pipeline_cleanup_v2/results_summary.json`
→ `strict_routing`. **Full raw report (local only, gitignored):**
`evaluation/outputs/nli_end_to_end_strict_routing_report.json`.

Architectural principle, tightened further from §8:

```
controlled structured field (path is exactly
  listing.facilities[N].name or rooms[N].facilities[M].name)
  + known alias mapping   → deterministic SUPPORT
  + no mapping             → NOT_VERIFIED, never sent to NLI

free natural-language evidence (everything else: .overview prose,
  room .name, highlights, description, policies, fine_print)
  → semantic verifier
```

The routing check (`structured_tag_rule.is_controlled_structured_tag`) is
by **exact path pattern**, not by the coarser `source_type` bucket used in
§8 — `facilities[].overview` prose and `rooms[].options[].choices`
rate-plan strings live under the same `source_type` as genuine
`facilities[].name` tags but are *not* controlled vocabulary and must still
go through NLI.

Examples of the distinction this enables:
- `"Desk"` (a `facilities[].name` tag) → routes directly to `RW1` SUPPORT,
  no NLI call.
- `"Free WiFi"` (a `facilities[].name` tag) → routes directly to `RW2A`
  SUPPORT — but it does **not** establish `RW2B` (reliability); RW2B has no
  alias entry and any RW2B evidence still goes through NLI.
- `"Non-smoking throughout"` has no entry in `DETERMINISTIC_ALIASES` for
  any quiet subclaim, so it is **not** speculatively mapped to quiet — it
  is either routed to NOT_VERIFIED (if it were itself a controlled tag with
  no mapping) or, as here, treated as ordinary free text sent to NLI, where
  the model must earn its own answer rather than a hand-authored shortcut
  deciding for it.

Results:

| K | precision | recall | false_support_rate | accuracy | NLI calls |
|---|---|---|---|---|---|
| 1 | **1.0** | 1.0 | **0.0** | 1.0 | — |
| 3 | 0.941 | 1.0 | 0.042 | 0.975 | 75 (vs 200 naive, 127 in v2) |

These are counts over this 4-property, 40-subclaim DEV/regression set (16
supported subclaims at K=1) — the K=1 row having 0 false positives here
means 0 were observed on this small set, not that the pipeline is provably
error-free at K=1 in general.

Candidate funnel: 200 → 189 (heading filter) → 137 (dedup) → 62 resolved
deterministically (10 SUPPORT + 52 NOT_VERIFIED-without-NLI) → **75 real
NLI calls**, 125 fewer than the naive 200 and 52 fewer than §8's 127.

Regression checks (`strict_routing.regression_checks`, 8 fixed
canary cases): 7 of 8 pass, including the two seeded from §7
(`"Non-smoking throughout"` correctly stays not-SUPPORT for quiet;
`"Children and beds"` correctly stays not-SUPPORT for family layout). The
one remaining failure, **unresolved and explicitly not claimed fixed**:
`"Continental breakfast included" → good breakfast quality` is still
verified SUPPORT somewhere in the pipeline — a facility-presence-vs-quality
conflation that structured routing does not address (there is no
"good breakfast quality" alias to route deterministically; it is a
genuine semantic-verifier judgment call, not a routing gap).

**Note on the 4-property DEV set:** because architecture v1→v2→v3 was
iterated by directly inspecting failures on these same 4 properties, this
data functions as a **DEV/regression set**, not an unbiased held-out test
set. This is exactly why a separate held-out benchmark was built next (§10).

---

## 10. Held-out semantic challenge benchmark

**Build script:** `evaluation/experiments/verifier_comparison/build_benchmark.py`
**Frozen benchmark:** `evaluation/experiments/verifier_comparison/benchmark_v1.jsonl`
**Held-out property snapshots:** `evaluation/experiments/verifier_comparison/holdout_properties/`
(19 properties: 13 Baku + 6 Barcelona — disjoint from, and including a
second city beyond, the 4-property Baku DEV set used in §4–§9)

**Status: FROZEN.** N=65. Labels were not changed after any model output
was inspected. This is deliberately called a *semantic challenge set*, not
a representative production-traffic sample: properties and evidence
snippets were selected for semantic richness (real CONTRADICT cases, real
aspect-vs-whole-property ambiguity, real template repeats), not sampled
uniformly.

**Annotation semantics — the gold label is purely the semantic relation
between the evidence text and the hypothesis, independent of source
credibility or marketing tone.** For example, *"Delicious Breakfast"*
semantically supports the breakfast-quality hypothesis even though it
comes from marketing copy — source provenance (`source_type`) is recorded
as separate metadata, not folded into the gold relation.

Relation distribution (N=65):

| gold_relation | n | % |
|---|---|---|
| NOT_ENOUGH_EVIDENCE | 29 | 45% |
| SUPPORT | 27 | 42% |
| CONTRADICT | 9 | 14% |

Confidence split (primary vs secondary evaluation, §12–§13):

| annotation_confidence | n |
|---|---|
| high (primary) | 53 |
| medium (secondary) | 12 |

Hypothesis distribution: FAM1=10, BQ1=10, CL1=8, NL1=6, Q1=5, Q2=5, Q3=4,
FC1=4, RW1=4, RW2A=4, BC1=3, RW2B=2.

Source-type distribution: description=23, review_summary=20 (the
Booking-generated `reviewSummary.pros/cons` guest-review aggregate field),
policies=11, highlights=9, other=2.

**`template_group_id`:** groups examples that share the *exact same*
`(evidence_text, hypothesis_id)` pair repeated verbatim across *different*
properties (Booking-generated boilerplate reused as-is) — deliberately
conservative, exact-match only. 3 groups, 7 examples total:

| group | evidence text | hypothesis | gold | members |
|---|---|---|---|---|
| tg_001_q2 | "Quiet street view" | Q2 | SUPPORT | 2 (walker_apartment, city_center_hotel) |
| tg_002_fc1 | "Child policies Children are not allowed." | FC1 | CONTRADICT | 2 (hotel_ambit, azcot_hotel) |
| tg_003_fam1 | "Child policies Children are not allowed." | FAM1 | CONTRADICT | 3 (hotel_ambit, azcot_hotel, apartment_3bedrooms) |

This supports two parallel analyses reported separately in §13: ordinary
example-level metrics (N=53 primary) and a template-deduplicated metric (one
representative per group, N=49) — so a repeated Booking template cannot
inflate the appearance of independent semantic successes.

**Documented limitations of this benchmark** (real data gaps, not padded):
- CONTRADICT coverage capped at 9 real examples after exhaustive search —
  short of an original ~12–15 target, kept as an honest limitation rather
  than synthesized.
- No genuine free-text SUPPORT evidence exists anywhere in the held-out or
  DEV corpus for Q1 (soundproofing as a stated feature) — only structured
  facility tags, which are out of scope for a free-text verifier benchmark.
- No genuine WiFi-reliability (RW2B) language exists anywhere in the
  mined corpus (RW2B has only 2 examples, both NOT_ENOUGH_EVIDENCE).
- Bed-comfort (BC1) coverage is limited (3 examples).

---

## 11. Verifier comparison

Three candidate verifiers were compared on the frozen benchmark (§10).

**A1 — `GuardrailsAI/finetuned_nli_provenance`, honest 3-class.** The same
model evaluated in §5–§9, now on genuinely held-out data with real
CONTRADICT examples for the first time.

**A2 — same model, SUPPORT-only mode.** `entailment → SUPPORT`,
`neutral + contradiction → NOT_ENOUGH_EVIDENCE` — the §6 reinterpretation,
now tested (not just theorized) against real contradiction cases.

**B — `MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli`.** A
specialized NLI model (trained on MNLI+FEVER-NLI+ANLI+LingNLI+WANLI),
wrapped in `evaluation/experiments/verifier_comparison/model_b.py`. Label
order (`{0: entailment, 1: neutral, 2: contradiction}`) verified against
the real `config.json` — same order as A, confirmed rather than assumed.
Same calling convention as A: `tokenizer(premise, hypothesis,
truncation=True, max_length=256)`.

**C — `gemini-2.5-flash`.** Not a dedicated NLI model — a general-purpose
LLM prompted to perform the same semantic-relation classification, wrapped
in `evaluation/experiments/verifier_comparison/verifier_c_llm.py`. Exact
configuration used, from that file:
- `temperature = 0.0`
- structured output: `response_mime_type="application/json"` +
  `response_schema` (`{relation: enum[SUPPORT, CONTRADICT,
  NOT_ENOUGH_EVIDENCE], reason: string}`)
- **no few-shot examples** — fixed system prompt only, written before the
  benchmark run, with no per-example tuning
- system prompt instructs the model to judge only the literal evidence
  text given, not use outside knowledge, and not discount text for reading
  like marketing copy

**Orchestrator:** `evaluation/experiments/verifier_comparison/run_comparison.py`
**Metrics module:** `evaluation/experiments/verifier_comparison/metrics.py`
(same zero-division-fixed convention as §5's corrected metrics: a class
with real gold examples but zero correct predictions gets F1=0.0, not
dropped/undefined)
**Full raw report (local only, gitignored):**
`evaluation/outputs/verifier_comparison_report.json` — per-example raw
probabilities/logits/LLM responses for all 65 examples × 3 methods,
preserved for a **future, separately-scoped** ranking experiment (§18);
not used to design that experiment here.
**Tracked summary (in repo — this comparison's source of truth for this document):**
`evaluation/experiments/verifier_comparison/results_summary.json`

---

## 12. Primary results (high-confidence, N=53)

**These are held-out semantic-challenge-set results, not production
performance and not representative-traffic performance** — see §10 for
what the benchmark deliberately is and is not.

| Method | Accuracy | Macro F1 | SUPPORT P/R/F1 (n_gold=26) | CONTRADICT P/R/F1 (n_gold=9)‡ | NEE P/R/F1 (n_gold=18) |
|---|---|---|---|---|---|
| A1 (baseline, 3-class) | 0.623 | 0.501 | 0.852 / 0.885 / 0.868 | 0.360 / 1.000 / 0.529 | 1.000 / 0.056 / 0.105 |
| A2 (baseline, SUPPORT-only) | 0.698 | 0.501 | 0.852 / 0.885 / 0.868 | — / 0.0 / 0.0† | 0.539 / 0.778 / 0.636 |
| B (DeBERTa-v3-large) | **0.868** | **0.872** | 0.833 / 0.962 / 0.893 | 0.900 / 1.000 / 0.947 | 0.923 / 0.667 / 0.774 |
| C (Gemini 2.5 Flash) | **0.868** | 0.861 | 0.862 / 0.962 / 0.909 | 0.818 / 1.000 / 0.900 | 0.923 / 0.667 / 0.774 |

‡ CONTRADICT recall of 1.000 for B and C means all 9 gold-CONTRADICT
examples were correctly classified — a clean result, but on a base of only
9 examples; read it as "no CONTRADICT recall errors observed on this
9-example slice", not as a precise recall estimate that would hold on a
larger CONTRADICT sample.

† A2 collapses contradiction into NOT_ENOUGH_EVIDENCE by construction, so
it structurally cannot predict CONTRADICT at all — its CONTRADICT
precision is undefined (no predictions made) and recall is 0 by
definition, not by model failure.

Reliability-critical rates (priority order: CONTRADICT→SUPPORT is the most
safety-critical, since it means the verifier asserts a claim the evidence
actively denies). **The benchmark has only 9 high-confidence CONTRADICT
gold examples (§10) — every CONTRADICT→SUPPORT rate below is a rate over
that N=9, not a large-sample estimate:**

| Method | CONTRADICT→SUPPORT (N=9) | NEE→SUPPORT | false_SUPPORT | false_CONTRADICT |
|---|---|---|---|---|
| A1 | **0/9** | 0.222 | 0.148 | 0.364 |
| A2 | **0/9** | 0.222 | 0.148 | 0.0 |
| B | **0/9** | 0.278 | 0.185 | 0.023 |
| C | **0/9** | 0.222 | 0.148 | 0.045 |

**No CONTRADICT→SUPPORT errors were observed on the 9 high-confidence
CONTRADICT examples in this benchmark, for any of the four methods.** This
is the single most important reliability finding here, but with N=9 it
should be read as "no failures observed on the CONTRADICT cases we have",
not as proof that any of these methods will never assert SUPPORT on
contradicting evidence in general — a larger CONTRADICT sample (§10's
documented limitation) could reveal a nonzero rate that 9 examples cannot
rule out.

**Reading the rest of the table:**
- A1 is not a trustworthy 3-class verifier on real held-out contradiction
  data either: CONTRADICT precision is 0.360 (16 false CONTRADICT
  predictions out of 44 non-contradict-gold examples — `false_CONTRADICT
  = 0.364`), and NOT_ENOUGH_EVIDENCE recall is 0.056 (1 of 18) — the same
  neutral-vs-contradiction conflation documented in §5, now confirmed on
  genuinely unseen data with real CONTRADICT examples.
- A2 (SUPPORT-only) trades away true contradiction semantics for a
  materially higher accuracy (0.698 vs 0.623) and a clean false_CONTRADICT
  rate (0.0, since it never predicts CONTRADICT) — usable as a
  conservative one-sided filter, but it can no longer distinguish "this
  evidence actively denies the claim" from "this evidence says nothing
  about the claim", which the product may care about.
- B and C are **substantially stronger** than either baseline mode on every
  axis except A2's structurally-guaranteed false_CONTRADICT of 0.0.
- **B and C have identical accuracy (0.868 = 0.868) on this benchmark.** B
  has a marginally higher macro F1 (0.872 vs 0.861); C has a marginally
  lower false_SUPPORT rate (0.148 vs 0.185) and NEE→SUPPORT rate (0.222 vs
  0.278). **On N=53, these differences are small enough that this should
  not be read as a statistically decisive winner between B and C** — see
  §13 for how the ranking holds (or doesn't) under the secondary set and
  template deduplication.

---

## 13. Secondary and template-deduplicated results

### Secondary (medium-confidence, N=12) — reported separately, not merged into §12

| Method | Accuracy | Macro F1 |
|---|---|---|
| A1 | 0.083 | 0.250 |
| A2 | **0.833** | 0.700 |
| B | 0.750 | 0.429 |
| C | 0.500 | 0.563 |

These 12 examples are exactly the ones annotation flagged as harder/more
ambiguous during labeling — that is *why* they are secondary, not primary.
With N=12, single-example flips move accuracy by ~8 points, so **this
table should not be over-interpreted** as a reliable ranking signal; it is
reported because the freeze agreement required it to be visible, not
because it settles anything. Notably the primary-set ordering does not
hold here — A2 is best and C is worst on this small slice — which is
itself the main takeaway: don't generalize a verifier ranking from N=53
(or N=12) alone.

### Template-deduplicated (primary set, one representative per template group, N=49)

| Method | Accuracy | Macro F1 |
|---|---|---|
| A1 | 0.592 | 0.466 |
| A2 | 0.735 | 0.515 |
| B | 0.857 | 0.862 |
| C | 0.857 | 0.846 |
| *(primary, N=53, for comparison)* | *A1 0.623 / A2 0.698 / B 0.868 / C 0.868* | *A1 0.501 / A2 0.501 / B 0.872 / C 0.861* |

Dropping the 4 duplicate-template instances moves every method's accuracy
down slightly (B and C both from 0.868 to 0.857) but **does not change the
relative ordering** — B and C remain close and both clearly ahead of
A1/A2.

### Template consistency

All 3 template groups × all 4 methods: **100% self-consistent** — every
method predicted the exact same relation for every repeated instance of a
given template (`primary_template_consistency` in `results_summary.json`).
This confirms none of the four methods' apparent successes on repeated
templates come from inconsistent/lucky guessing on identical input.

---

## 14. Failure analysis

Recurring failure patterns for the two strong verifiers (B, C), quoted
directly from `evaluation/outputs/verifier_comparison_report.json`
(`examples_full_raw`, high-confidence subset):

### Aspect vs whole-property inference

Both B and C **systematically** over-generalize from a narrow dining-level
claim to the broad FAM1 hypothesis ("The property is suitable for
families"), on all 3 examples in the benchmark that test this:

| example_id | evidence | gold | A1 | B | C |
|---|---|---|---|---|---|
| vb_fam1_03 | "…features a family-friendly restaurant serving Greek, Italian, and Middle Eastern cuisines." | NOT_ENOUGH_EVIDENCE | SUPPORT | SUPPORT | SUPPORT |
| vb_fam1_04 | "…family-friendly restaurant serving Turkish cuisine…" | NOT_ENOUGH_EVIDENCE | SUPPORT | SUPPORT | SUPPORT |
| vb_fam1_09 | "…shared kitchen, minimarket, coffee shop, and child-friendly buffet." | NOT_ENOUGH_EVIDENCE | SUPPORT | SUPPORT | SUPPORT |

All three are annotated NOT_ENOUGH_EVIDENCE precisely because "family-
friendly restaurant"/"child-friendly buffet" supports only a dining-level
aspect, not the broad property-suitability claim (§10's annotation rule).
Both strong models — and the baseline — miss this distinction identically.
This is a compositional/claim-strength failure, not a topic-matching
failure: the retrieved text is genuinely on-topic.

### Service presence vs outcome

| example_id | evidence | gold | A1 | B | C |
|---|---|---|---|---|---|
| vb_cl1_08 | "…paid shuttle service, lift, 24-hour front desk, minimarket, **daily housekeeping**, hairdresser, tour desk, and luggage storage." (→ CL1: "the property is clean") | NOT_ENOUGH_EVIDENCE | CONTRADICT | SUPPORT | SUPPORT |

Both B and C conflate the *presence* of a cleaning service with an
*observed-outcome* claim about cleanliness — the same presence-vs-outcome
pattern seen with "Continental breakfast included" in §7/§9. (For
contrast, direct outcome-language evidence like "Cleanliness: Spotless and
well-maintained living spaces" is correctly SUPPORT for all three methods —
`vb_cl1_01`–`vb_cl1_05`.)

### Soundproofing ambiguity (Q1)

| example_id | evidence | gold | A1 | B (entailment prob) | C (reason) |
|---|---|---|---|---|---|
| vb_q1_02 | "Room Soundproofing: Disturbance from street traffic, loud music, and internal ventilation noise." | NOT_ENOUGH_EVIDENCE | SUPPORT | **SUPPORT** (0.870) | **CONTRADICT** ("disturbed by noise … contradicts the idea that the room has effective soundproofing") |

This is the clearest case of the two strong verifiers **disagreeing with
each other in opposite directions on the same input**: B anchors on the
literal word "Soundproofing" in the heading and infers SUPPORT with high
confidence; C reasons from the described noise *outcome* to infer the
soundproofing *feature* is absent, landing on CONTRADICT. Both are wrong
relative to the frozen gold (NOT_ENOUGH_EVIDENCE — a guest-reported noise
complaint doesn't strictly confirm or deny a soundproofing feature's
existence). A related case, `vb_q1_01` ("Many guests noted poor
soundproofing with noise from nearby bars"), gets CONTRADICT from *all
three* methods (A1, B, C) against the same NOT_ENOUGH_EVIDENCE gold — a
shared over-inference from an outcome complaint to a feature-absence claim,
this time in one consistent (wrong) direction.

**These remaining errors are compositional and claim-specific, not simple
topic-matching failures** — the retrieved evidence is genuinely on-topic in
every case above; the error is in how strongly a specific phrase should be
read as confirming, denying, or being silent on a specific hypothesis.
**No change was made to the frozen benchmark or to architecture v3 as a
result of these findings** — they are reported as-is, per the freeze
agreement.

---

## 15. Operational comparison

Measured on the same 65-example run (`evaluation/outputs/verifier_comparison_report.json`,
`examples_full_raw[*].{B,C}.latency_ms`, `estimated_cost_usd`). **These
latency numbers are measurements of this specific experimental
environment (a single local machine, sequential calls, whatever hardware
and network conditions happened to hold during this run) — they are not
intrinsic, hardware-independent properties of either model.** DeBERTa's
latency depends strongly on the deployment hardware/runtime it is served
on (CPU vs GPU, batch size, serving framework); Gemini's latency is an
end-to-end measurement that includes remote API, network, and Google's
service-side latency, none of which this experiment controls or isolates.
In this experimental environment, DeBERTa had lower measured mean and p95
latency than the Gemini API:

| | B — DeBERTa-v3-large | C — Gemini 2.5 Flash |
|---|---|---|
| Deployment | local (torch/transformers, isolated venv) | remote API |
| Latency mean | 1255 ms | 1778 ms |
| Latency p50 | 1434 ms | 1595 ms |
| Latency p95 | 1692 ms | 3076 ms |
| Per-call API cost | $0 (no metered API) | ~$0.000164 (65 calls → $0.01069 total) |
| Cost per 1,000 calls | $0 metered | ~$0.1645 |
| Parse failures | n/a (fixed label set, no free-form parsing) | 0 / 65 (structured JSON output worked every time) |

**"$0" for DeBERTa means no per-call API metering fee — it does not mean
free production infrastructure.** Running B in production requires hosting
the model (GPU/CPU inference, RAM for a ~large transformer, runtime/version
management, scaling the service) — real infrastructure cost that this
experiment does not measure.

**Trade-off, as measured here (not a general claim about either model
family, and not a claim about intrinsic model speed — see the latency
caveat above):**
- **DeBERTa (B):** self-hosted, lower mean/p50/p95 latency *in this run on
  this hardware*, no external network dependency or per-call cost — but
  requires model hosting, RAM, and runtime management as an ongoing
  engineering responsibility, and its latency in a different deployment
  (different hardware, batching, GPU vs CPU) could differ materially from
  what was measured here.
- **Gemini (C):** no model hosting to build or maintain, but a network/API
  dependency, a real (if small at this volume) usage cost, and a higher
  p95 tail latency (3076 ms vs 1692 ms) *in this run* — consistent with
  network-bound variance rather than local compute variance, and itself a
  reflection of network conditions on the day of this run rather than a
  guaranteed Gemini API characteristic.

For reference, the current production-path baseline (A, run on the same
isolated venv, same 65 calls, same environment caveat as above) was
markedly slower than either candidate in this run: mean 2765 ms / p50 2872
ms / p95 3117 ms — noted here for context, not as a recommendation, since
A's accuracy (§12) already rules it out as a trustworthy 3-class verifier
regardless of latency.

---

## 16. Main conclusions

1. Embedding similarity is necessary for evidence retrieval but not
   sufficient for evidence validity (§1, §4).
2. Atomic decomposition improved retrieval substantially — Recall@1 rose
   from 0.556 to 0.929, Recall@3/5 reached 1.0 (§4).
3. Oracle verifier results can be misleading: a perfect SUPPORT-only score
   (precision 1.0, recall 1.0, false_support_rate 0.0, §6) was measured on
   human-selected evidence and did not predict real end-to-end behavior,
   where false-support rate reached 0.278 at K=5 (§7).
4. Increasing K increases false SUPPORT under OR aggregation — false
   support quadrupled from K=1 (0.056) to K=5 (0.278) in the same
   experiment (§7).
5. Structured facts should not be passed through a semantic model when
   their meaning is already known — deterministic routing for controlled
   `facilities[].name` tags reduced false SUPPORT at K=1 (0.056 in §7's
   32-subclaim schema, to 0.042 then 0.0 across the 40-subclaim §8→§9
   architecture-v2/v3 progression) and reduced unnecessary NLI calls in
   that same §8→§9 progression (200 naive → 127 → 75) (§8–§9).
6. Strict typed routing (exact path pattern, not coarse source-type
   bucketing) reduced both errors and unnecessary semantic-model calls
   further than source-type-only structured/free-text splitting (§9).
7. The original Guardrails NLI model is not a trustworthy 3-way verifier
   for this domain: on genuinely held-out data with real CONTRADICT
   examples, CONTRADICT precision is 0.360 and NOT_ENOUGH_EVIDENCE recall
   is 0.056 (§12).
8. DeBERTa-v3-large and Gemini 2.5 Flash both perform substantially better
   than the baseline on the held-out semantic challenge set — accuracy
   0.868 vs 0.623/0.698, macro F1 ~0.87 vs ~0.50 (§12).
9. **There is no overwhelming winner between DeBERTa and Gemini on this
   benchmark** — identical accuracy (0.868), close macro F1 (0.872 vs
   0.861), and the small ranking differences that do appear reverse
   direction between the primary and secondary sets (§12–§13). This should
   not be read as a statistically decisive result at N=53.
10. Remaining errors for both strong verifiers concern evidence
    strength/compositionality (aspect-vs-whole-property, presence-vs-
    outcome) rather than pure topical relevance — the retrieved text is
    on-topic in every documented failure case (§14).

---

## 17. Product interpretation

The final booking assistant is a **recommendation/ranking system, not a
guarantee engine**. Hard constraints (dates, price, guest count, property
type) remain strict pass/fail filters. Soft preferences are not held to
the same bar: the system does not need to prove with 100% certainty that a
property satisfies "family-friendly" before it can be useful.

Concretely: `"family-friendly restaurant"` is, per this benchmark's
annotation rule (§10, §14), insufficient evidence for the strict semantic
statement "the whole property is suitable for families" — but it may still
be a legitimate **weak positive ranking signal** worth surfacing among many
signals, rather than a claim requiring binary confirmation.

**Semantic verification quality (this document) and final hotel-ranking
quality (future work, §18) are related but different evaluation
problems.** A verifier being conservative about what counts as SUPPORT is
a *feature* for strict claim-checking and can simultaneously be
*information-losing* for ranking, which might reasonably want to weigh
graded/partial evidence rather than discard it.

---

## 18. Next step

**Next experiment: hotel ranking using evidence strength / verifier
signals.**

The ranking experiment must answer: **do stronger verifier/evidence
signals actually rank more suitable hotels above less suitable hotels?**

This is explicitly **not designed or implemented in this document or in
any code committed here.** The raw per-example logits/probabilities/LLM
responses from the verifier comparison were preserved specifically to make
that future experiment possible
(`evaluation/outputs/verifier_comparison_report.json` →
`examples_full_raw`), but no ranking/aggregation logic has been written
against them.

---

## Result locations

**Every path below is tagged "In repo" or "Local only" so a reader after a
fresh `git clone` can tell, without running anything, which files they
will actually have on disk and which exist only in whoever ran the
experiment's local `evaluation/outputs/` directory (gitignored — see
`.gitignore`'s `evaluation/outputs/*.json` / `*.jsonl` rules, a pre-existing
repo convention, not something introduced by this experiment track).**

| Artifact | Path | In repo? | Contents |
|---|---|---|---|
| DEV property fixtures | `evaluation/experiments/evidence_retrieval/golden/properties/` | **In repo** | 4 Baku properties used throughout §4–§9 |
| Retrieval checkpoints | `evaluation/experiments/evidence_retrieval/golden/retrieval_checkpoint_v1.jsonl`, `retrieval_checkpoint_v2.jsonl` | **In repo** | Per-pair/per-subclaim retrieval results, broad vs atomic queries |
| Retrieval results summary | `evaluation/experiments/evidence_retrieval/results_summary.json` | **In repo** (tracked summary) | Recall@1/3/5, score distributions, error counts for both checkpoints |
| Retrieval full reports | `evaluation/outputs/evidence_retrieval_checkpoint_report.json`, `evidence_retrieval_checkpoint_v2_report.json` | **Local only** (gitignored) | Full per-pair raw results — not present after a fresh clone; regenerate with §"Reproduce commands" below |
| Oracle NLI input/candidates | `evaluation/experiments/nli_oracle/nli_input_v2.jsonl`, `retrieval_candidates.jsonl` | **In repo** | v2 oracle input; v1 end-to-end retrieval candidates |
| Oracle + end-to-end v1 results summary | `evaluation/experiments/nli_oracle/results_summary.json` | **In repo** (tracked summary) | Stages: `oracle_v1`, `oracle_baseline_v1_recompute`, `oracle_v2`, `sanity_check`, `v1_vs_v2_comparison`, `support_verifier_v2_analysis`, `end_to_end_v1` |
| Oracle + end-to-end v1 full reports | `evaluation/outputs/nli_oracle_report.json`, `nli_oracle_baseline_v1_report.json`, `nli_oracle_v2_report.json`, `nli_oracle_sanity_check_report.json`, `nli_oracle_v1_vs_v2_comparison.json`, `nli_oracle_support_verifier_v2_analysis.json`, `nli_end_to_end_verification_report.json` | **Local only** (gitignored) | Full per-example raw results for each stage — not present after a fresh clone |
| Architecture v2 candidates | `evaluation/experiments/pipeline_cleanup_v2/retrieval_candidates_v2.jsonl` | **In repo** | 40-subclaim atomic retrieval candidates |
| Architecture v2 retrieval-only metrics | `evaluation/experiments/pipeline_cleanup_v2/retrieval_only_metrics_v2.json` | **In repo** (tracked) | Recall@1/3 for the 40-subclaim schema |
| Architecture v2 / strict routing results summary | `evaluation/experiments/pipeline_cleanup_v2/results_summary.json` | **In repo** (tracked summary) | Stages: `end_to_end_v2`, `v1_vs_v2_e2e_comparison`, `strict_routing` |
| Architecture v2 / strict routing full reports | `evaluation/outputs/nli_end_to_end_verification_v2_report.json`, `nli_end_to_end_v1_vs_v2_comparison.json`, `nli_end_to_end_strict_routing_report.json` | **Local only** (gitignored) | Full per-subclaim raw results, regression check details — not present after a fresh clone |
| Held-out property snapshots | `evaluation/experiments/verifier_comparison/holdout_properties/` | **In repo** | 19 properties (13 Baku, 6 Barcelona) |
| Frozen benchmark | `evaluation/experiments/verifier_comparison/benchmark_v1.jsonl` | **In repo** | N=65, FROZEN semantic challenge set |
| Verifier predictions + comparison metrics | `evaluation/experiments/verifier_comparison/results_summary.json` | **In repo** (tracked summary) | Run metadata, primary/secondary/template-dedup metrics + confusion matrices for A1/A2/B/C, template consistency |
| Verifier comparison full report | `evaluation/outputs/verifier_comparison_report.json` | **Local only** (gitignored) | Per-example raw predictions/probabilities/logits/LLM responses for all 65 examples × 3 methods — not present after a fresh clone; source for §14's quotes and §15's latency/cost figures, and the input for the future ranking experiment. Regenerate with the `run_comparison` command below (requires `GOOGLE_API_KEY` and re-incurs the small Gemini API cost noted in §15). |

**Rule of thumb applied consistently above:** every `evaluation/experiments/**/results_summary.json`
is a small, tracked, git-committed extract (metrics, confusion matrices,
metadata — no per-example raw predictions); every `evaluation/outputs/*.json`
path is a full raw report that exists only locally for whoever ran the
experiment, is excluded by `.gitignore`, and must be regenerated by
re-running the corresponding script to view again.

---

## Reproducibility

**Two separate Python environments are used:**

1. **Main project venv** (repo root `pyproject.toml`, includes `pydantic`,
   `google-genai` for the embedding API) — required for any script that
   calls the embedding model or touches `app.*` schemas: retrieval
   checkpoints (§4), `nli_oracle/build_retrieval_candidates.py` (§7),
   `pipeline_cleanup_v2/build_retrieval_candidates_v2.py` (§8),
   `verifier_comparison/build_benchmark.py` (§10, no model calls, but
   imports `app.retrieval.apify.normalize_apify_listing` /
   `app.schemas.listing.ListingRaw`).
2. **Isolated experiment venv**
   (`evaluation/experiments/nli_oracle/.venv/`, requirements pinned in
   `evaluation/experiments/nli_oracle/requirements.txt`, **not** part of
   `pyproject.toml`) — required for any script doing NLI model inference
   (`torch`, `transformers`) or calling Gemini as verifier C
   (`google-genai`, `python-dotenv`, `sentencepiece` for the DeBERTa
   tokenizer). Shared by `nli_oracle/`, `pipeline_cleanup_v2/`, and
   `verifier_comparison/`. Setup:
   ```
   python3.11 -m venv evaluation/experiments/nli_oracle/.venv
   evaluation/experiments/nli_oracle/.venv/bin/pip install -r evaluation/experiments/nli_oracle/requirements.txt
   ```

**Models used:**
- Embeddings: `gemini-embedding-001`
- Baseline NLI (A): `GuardrailsAI/finetuned_nli_provenance`
- Candidate NLI (B): `MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli`
- Candidate LLM (C): `gemini-2.5-flash`

**External API requirement:** verifier C (§11) and the embedding-generating
scripts require `GOOGLE_API_KEY` in the environment (loaded via
`python-dotenv`'s `load_dotenv()`; not committed anywhere in this repo).
Running the Gemini-dependent scripts incurs real (small) API cost — see
§15 for the measured total ($0.01069 for the 65-example comparison run).

**Determinism:** verifier C uses `temperature=0.0`; A and B are
deterministic argmax over fixed model weights (no sampling). Embedding
retrieval and NLI inference for A/B are otherwise deterministic given
fixed input.

**Reproduce commands** (each script's own docstring states which venv it
needs; summarized here):

```
# Retrieval checkpoints (main venv)
python -m evaluation.experiments.evidence_retrieval.run_retrieval_checkpoint
python -m evaluation.experiments.evidence_retrieval.run_retrieval_checkpoint_v2

# NLI oracle (isolated venv)
evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.run_oracle_eval
evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.run_oracle_eval_v2

# End-to-end v1: retrieval (main venv) then verification (isolated venv)
python -m evaluation.experiments.nli_oracle.build_retrieval_candidates
evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.run_end_to_end_verification

# Architecture v2 / strict routing: retrieval (main venv) then verification (isolated venv)
python -m evaluation.experiments.pipeline_cleanup_v2.build_retrieval_candidates_v2
evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.pipeline_cleanup_v2.run_end_to_end_verification_v2
evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.pipeline_cleanup_v2.run_strict_structured_routing

# Frozen benchmark build (main venv, no model inference)
python -m evaluation.experiments.verifier_comparison.build_benchmark

# A/B/C verifier comparison (isolated venv, requires GOOGLE_API_KEY)
evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.verifier_comparison.run_comparison
```

All reports write to `evaluation/outputs/` (gitignored — see the tracked
`results_summary.json` files listed in **Result locations** above for the
committed subset of each report).

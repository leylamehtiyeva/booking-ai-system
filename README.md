# Booking AI System

Constraint-aware booking assistant with multi-turn search state, conversation routing, deterministic eligibility decisions, and controlled LLM usage.

<p align="center">
  <img src="assets/booking_ai_system_gif.gif" width="1000"/>
</p>

## Problem

LLM-based systems are inherently non-deterministic and can struggle to reliably enforce strict user requirements
(e.g. "must have WiFi", "no smoking", "under $100").

In a booking assistant:

* hard constraints must be respected
* search state must remain consistent across turns
* decisions should be explainable
* LLM failures must not silently corrupt application state
* expensive or probabilistic components should be isolated and measurable

## Architecture

The system separates conversational routing, structured search state, probabilistic evidence extraction, and deterministic domain decisions.

### High-level flow

```text
User Message
    ↓
Conversation Router
    ↓
Action Dispatch
    ├── start_search
    │      ↓
    │   SearchRequest extraction
    │      ↓
    │   Search pipeline
    │
    ├── update_search
    │      ↓
    │   Existing SearchRequest update
    │      ↓
    │   Search pipeline
    │
    ├── listing_question
    │      ↓
    │   Listing Q&A path
    │
    └── general_chat
           ↓
       Non-search response
```

For search actions:

```text
SearchRequest
    ↓
Constraints
    ↓
Retrieval
    ↓
Matching
    ↓
Deterministic Eligibility Decision
    ↓
Ranking
    ↓
Results
```

### Key principles

* `SearchRequest` is the canonical source of truth for the active search state
* Constraints are the source of truth for semantic user requirements
* LLMs are used for narrow tasks such as routing, intent extraction/update, and textual evidence resolution
* Python controls application flow and state transitions
* Final listing eligibility decisions are deterministic and auditable
* LLM failures are handled separately from domain state updates
* LLM components are evaluated independently before being used in the pipeline

### Matching layer

Matching combines deterministic checks with controlled LLM fallback:

* structured listing data → exact deterministic checks
* textual evidence → rules when possible
* unresolved textual evidence → LLM fallback

The LLM may extract or classify evidence, but the final eligibility decision remains in the deterministic layer.

## How It Works

1. Every user message is classified by the Conversation Router as:

   * `start_search`
   * `update_search`
   * `listing_question`
   * `general_chat`

2. Search actions create or update a structured `SearchRequest`

3. Required search context is validated and missing information is requested when needed

4. User requirements are extracted and normalized into constraints

5. Listings are retrieved

6. Matching is performed using:

   * structured checks
   * textual rules
   * LLM fallback only when evidence cannot be resolved deterministically

7. The deterministic decision layer resolves listing eligibility:

   * YES
   * NO
   * UNCERTAIN

8. Eligible listings are filtered and ranked

9. Structured results and explanations are returned to the UI

## Evaluation

The system is evaluated at multiple independent layers rather than relying only on end-to-end metrics.

### Constraint Extraction (166 cases)

Evaluates whether user requirements are correctly extracted and represented as structured constraints.

* Precision: **0.78**
* Recall: **0.79**
* F1: **0.79**
* Exact constraint set match: **0.80**
* Exact full-case match: **0.66**
* Priority accuracy on matched constraints: **0.92**
* Category accuracy on matched constraints: **0.93**

**Insight:**
Constraint detection is generally reliable, while mixed-priority and multi-constraint requests remain the main source of extraction errors.

### Constraint Resolution (120 cases)

Evaluates the final YES / NO / UNCERTAIN decision for individual constraints based on available listing evidence.

* Valid evaluated cases: **114 / 120**
* Runtime errors: **6**
* Accuracy on valid cases: **0.86**
* YES F1: **0.88**
* NO F1: **0.92**
* UNCERTAIN F1: **0.79**

Critical errors observed in the valid evaluated set:

* NO → YES: **0 cases**
* YES → NO: **0 cases**

**Insight:**
No critical NO → YES or YES → NO errors were observed in the evaluated cases. Most remaining errors involve uncertainty handling rather than direct decision reversals.

### Conversation Router (70 cases)

Evaluates classification of each user turn into:

* `start_search`
* `update_search`
* `listing_question`
* `general_chat`

Each case was executed three times per model to measure both correctness and consistency.

Baseline benchmark before the final routing-boundary prompt fixes:

**Gemini 2.5 Flash Lite**

* Accuracy: **0.986**
* Macro F1: **0.984**
* Majority-vote accuracy: **0.986**
* Consistency: **1.00**
* Median latency: **~1009 ms**

**Groq GPT-OSS 20B**

* Accuracy: **0.976**
* Macro F1: **0.973**
* Majority-vote accuracy: **0.971**
* Consistency: **0.957**
* Median latency: **~548 ms**

Groq provided slightly lower routing accuracy but substantially lower latency and estimated cost, and is currently used as the preferred router model.

After the final routing-boundary prompt changes and structured-output integration, the four previously failing regression cases were rerun three times each on Groq:

* **12 / 12 predictions correct**

A new full 70-case benchmark has not yet been recorded for the updated prompt.

**Insight:**
A lower-cost task-specific model can be sufficient for routing when the action space and output contract are tightly constrained and evaluated independently.

### Ranking Selection (103 cases)

Evaluates deterministic selection and ordering after listing eligibility has already been established.

* Exact ranking match rate: **0.97**
* Selected-set match rate: **1.00**
* Top-1 accuracy: **0.98**
* Ineligible leak rate: **0.00**
* Tier violation rate: **0.019**

**Insight:**
The deterministic ranking layer is highly stable and did not leak ineligible listings in the evaluation set.

### End-to-End (50 cases)

End-to-end evaluation covers the complete pipeline from user query to final listing selection.

The current dataset contains positive, negative, and uncertain scenarios.

**Status: under revision**

An initial evaluation run exists, but its ground truth is not considered reliable enough to use as a project-quality benchmark.

The dataset and evaluation contract are being revised before end-to-end metrics are treated as authoritative.

## Observability

Every request generates a structured telemetry trace that captures execution across the pipeline.

Each trace includes:

* trace ID
* total request latency
* latency for individual pipeline steps
* LLM calls
* token usage
* estimated LLM cost
* external API calls
* fallback usage
* router usage
* execution scenario

### Telemetry Dashboard

The project includes a Streamlit dashboard for inspecting telemetry logs.

The dashboard provides:

* request overview
* latency distribution
* P50 / P95 / P99 latency
* latency by pipeline step
* slowest requests
* trace viewer
* LLM usage and token statistics
* estimated request cost
* fallback / router / Apify usage
* raw telemetry explorer

This makes it possible to inspect individual traces, identify performance bottlenecks, and monitor LLM and external-service usage during development.

## Project Structure

```text
app/
├── agents/          # task-specific LLM agents
├── config/          # application settings and LLM profiles
├── llm/             # provider-independent LLM model construction
├── logic/           # application and deterministic domain logic
├── observability/   # telemetry, latency, cost, and traces
├── retrieval/       # listing retrieval
├── schemas/         # typed application and domain contracts
├── services/        # external integrations
├── tools/           # search pipeline orchestration
└── resources/       # reference data

ui/
├── components/
├── services/
├── state.py
└── streamlit_app.py

dashboards/
└── telemetry_dashboard.py

evaluation/
├── core/
├── datasets/
├── tasks/
└── outputs/

scripts/
tests/
fixtures/
assets/
```

## Tech Stack

* Python 3.11+
* Pydantic
* asyncio
* Google ADK
* LiteLLM
* Gemini API
* Groq API
* Apify
* Streamlit
* Pytest
* uv

## Running Locally

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy the example environment file:

```bash
cp .env.example .env
```

Configure the LLM providers used by the application:

```env
GOOGLE_API_KEY=...
GROQ_API_KEY=...
```

The Conversation Router can use the Groq profile:

```env
ROUTER_LLM_PROFILE=groq_gpt_oss_20b
```

The current Streamlit development/demo flow uses local fixture listings.

Live retrieval through Apify additionally requires the corresponding Apify credentials and actor configuration.

### 3. Run the Streamlit demo

```bash
uv run streamlit run ui/streamlit_app.py
```

### 4. Run the telemetry dashboard

```bash
uv run streamlit run dashboards/telemetry_dashboard.py
```

### 5. Run tests

```bash
uv run pytest
```

## Latency & Cost

Performance and cost are tracked per request through structured telemetry.

Observed characteristics:

* local fixture retrieval is used for fast and predictable development flows
* live Apify retrieval is the main external latency bottleneck and can take tens of seconds
* Groq router median latency in the baseline router evaluation was **~548 ms**
* LLM fallback is invoked only when deterministic or textual rules cannot resolve a constraint
* external retrieval and LLM usage are measured separately

Costs depend on:

* selected LLM provider and model
* number of LLM calls
* whether fallback reasoning is required
* retrieval source

Estimated cost is recorded per trace instead of relying on a single fixed per-request estimate.

## Example

User:

> "Apartment in Barcelona, June 15–20, under $100, must have WiFi"

System:

```text
Conversation Router
→ start_search
→ SearchRequest extraction
→ constraint extraction
→ listing retrieval
→ matching
→ deterministic eligibility decisions
→ ranking
→ results
```

A follow-up such as:

> "Make it cheaper"

is routed as `update_search`, allowing the existing structured search state to be updated instead of rebuilding the request from scratch.

## Current Development Focus

The current system already separates conversational intent routing from the search and matching pipeline.

The next major step is to add a natural-language conversational response layer on top of the controlled domain flow, so the assistant can maintain a natural multi-turn interaction without giving the response-generating LLM control over search state or domain decisions.

## Future Work

* Build the conversational response layer on top of the controlled domain pipeline
* Add explicit conversation and result state for natural multi-turn interactions
* Design reliable listing-reference resolution for follow-up questions
* Rebuild and validate the end-to-end evaluation ground truth
* Improve constraint extraction on mixed-priority and multi-constraint requests
* Evaluate lower-cost models for additional LLM tasks
* Prepare a public deployment for external user testing

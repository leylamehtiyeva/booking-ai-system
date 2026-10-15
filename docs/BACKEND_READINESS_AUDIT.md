# Booking AI Backend — Readiness Audit (v1.0)

**Дата аудита:** 2026-10-15 · **Commit:** `721c63d` (main) · **Scope:** `app/`, `ui/`, `tests/`, `evaluation/`, `scripts/`, `fixtures/`

## Статус выполнения

Ветка `backend-v1-hardening`. Обновлено 2026-10-15.

| Задача | Статус | Commit | Итог |
|---|---|---|---|
| P0-5 Hermetic test suite | ✅ Done | `2077102`, `baef520` | `uv run pytest` без сети и ключей; live — только `BOOKING_AI_LIVE_TESTS=1 uv run pytest -m live` |
| P0-2 Date normalization | ✅ Done | `a2c41f5`, `74b786e` | Reference clock, интерпретация дат как пары, clarification вместо crash; денежные суммы («$2000») не считаются годом |
| P0-3 Fixtures time bomb | ✅ Done | `a2c41f5` | Recurring yearly availability в fixture retriever; данные fixtures не менялись |
| P0-8 Fixture price semantics | ✅ Done | см. `git log` | Fixture retriever пересчитывает цену на запрошенное проживание; данные fixtures не менялись |
| P0-1 Graceful failure handling | ✅ Done | `fba5e0a` | Fatal stage → typed `processing_unavailable`; textual fallback → UNCERTAIN; программные ошибки не скрываются |
| P0-4 Application service | ✅ Done | `3f87a4b` | `process_turn(session, message) -> TurnResult`; Streamlit — тонкий adapter; публичный JSON без server state и debug |
| P0-6 Multi-turn integration tests | ✅ Done | см. `git log` | 7 hermetic сценариев через `process_turn`; mutation check: 7 из 7 дефектов пойманы |
| P0-7 E2E evaluation contract & baseline | ⏳ In progress | — | — |

Offline suite после P0-8: **799 passed, 2 skipped (live)**.

---

Источник истины для этого документа: код и результаты запусков. README использовался только как описание того, как задумано, а не как доказательство того, что это работает. Каждое утверждение ниже помечено evidence. Если что-то не проверялось, это сказано явно.

---

## 0. Что именно было проверено

| Проверка | Как | Результат |
|---|---|---|
| Полный test suite | `uv run pytest` | **628 passed** за 26 s |
| Hermeticity тестов | `pytest` с невалидными `GOOGLE_API_KEY`/`GROQ_API_KEY`/`APIFY_TOKEN` | **5 failed**: 3 теста в `test_reference_resolution.py` случайно вызывают реальный Gemini, 2 `test_live_*` никак не помечены |
| Отключённый test file | `pytest tests/_test_normalize_search_response.py` | **2 из 5 падают** (файл не собирается pytest'ом из-за `_`-префикса) |
| Ranking & selection eval (offline, без LLM) | логика runner'а запущена из scratchpad без перезаписи отчёта | Воспроизводится **1:1** с сохранённым отчётом (top-1 0.981, leak 0/103) |
| End-to-end eval (только `evaluate`, без LLM) | пересчёт сохранённых `predictions.jsonl` (23.09) | decision acc **0.66**, critical false YES **12/25 (48%)** |
| E2E с учётом `match_tier` (offline) | ad-hoc пересчёт тех же predictions | critical false YES **6/25**, decision acc **0.60** |
| Date normalization | прямой вызов с заданным `today` | Подтверждены 3 бага (см. P0-2) |
| Кросс-годовой диапазон дат через `handle_user_message` | router и intent LLM замоканы, без API | **Uncaught `ValidationError`**, turn падает |
| Ошибка LLM в LISTING_QUESTION | тесты с невалидным ключом | **Uncaught `google.genai.errors.ClientError`** из `handle_user_message` |
| Lint | `ruff check app ui` | 27 замечаний (13 E402, 11 F811 duplicate imports, 3 F401) |
| Latency/cost baseline | сохранённый e2e report + `logs/telemetry` | e2e: mean **8.6 s**, max **36 s**, 5.8 LLM calls/turn, ~$0.0009/turn (cost занижен, см. P1-5) |

**Что сознательно НЕ запускалось** (по правилу «никаких платных LLM/API pipelines без разрешения»): `end_to_end run predict`, router eval, reference-resolution eval, constraint extraction/resolution evals, factual-question eval, live Streamlit-сессия, Apify retrieval. Поэтому все LLM-зависимые метрики ниже — это **сохранённые отчёты**, не свежие прогоны. На текущем коде они не воспроизводились.

---

## 1. Executive Summary

Backend архитектурно сильный и в основном уже готов: явное состояние (`SearchRequest` + `ShownResultSet`), patch-based updates, abstention, детерминированная проверка того, на какой listing ссылается LLM, детерминированные factual verdicts. Unit-покрытие отдельных компонентов хорошее: 628 тестов, все зелёные. Это всё сделано всерьёз и хорошо выглядит на интервью.

До v1.0 backend **не готов** по трём причинам. Ни одна из них не требует переписывать архитектуру.

1. **Надёжность turn'а.** Graceful degradation есть только у ошибок Conversation Router. Ошибка или timeout любого другого LLM-вызова (intent extraction, intent update, factual extraction, textual fallback в search) и даже ошибка в датах пользователя роняют весь turn необработанным exception. Обе ситуации воспроизведены.
2. **Даты.** LLM не знает текущую дату. Год без явного указания принудительно ставится текущим, поэтому «April 10–15» в октябре 2026 превращается в прошедшую дату, а «Dec 28 – Jan 3» роняет turn. Fixtures захардкожены на апрель 2026, и demo-сценарий из README перестанет работать 1 января 2027.
3. **Отделение от UI и доверие к метрикам.** Обработка turn'а целиком (генерация ответа, сохранение telemetry, commit state, история) живёт в `ui/services/chat_handler.py` и импортирует `streamlit`. Без UI её не вызвать. При этом единственная system-level метрика (e2e) показывает 48% critical false YES, и её интерпретация зависит от незафиксированного контракта. Multi-turn сценарии не проверяются ни тестами, ни eval'ами.

**Оценка объёма:** 7 P0 задач на ~8–10 рабочих дней плюс выборочные P1. Backend можно закончить за **2.5 недели** и потом остановить разработку.

---

## 2. Architecture Assessment

### Сильные стороны (подтверждены кодом)

- **Typed state contracts.** `SearchRequest` (`app/schemas/query.py`) отвечает за intent, `ShownResultSet` (`app/schemas/shown_result_set.py`) — за то, на какие результаты пользователь может сослаться. Read-only turns по построению не трогают snapshot: `"shown_result_set"` есть в ответе только после выполненного поиска, это проверено тестами.
- **Abstention gate до любых действий.** `decision_status == NEEDS_CLARIFICATION` прерывает обработку до dispatch'а (`app/logic/conversation_flow.py`, `handle_user_message`). Если router вернул выдуманный `target_result_id`, проверка его отклоняет (`_resolve_target_reference`), а не подставляет ближайший результат.
- **Semantic no-op guard** для `UPDATE_SEARCH`: при `state == previous_state` новый поиск не запускается и `ShownResultSet` не заменяется.
- **LISTING_QUESTION pipeline.** Candidate selection, затем LLM extraction, затем structured matcher, затем textual fallback и детерминированный рендер. Verdict никогда не проходит через response LLM (`conversation_response_llm.py`, short-circuit для `ListingQueryConversationOutcome`).
- **Router hardening.** Timeout, retry только для retryable-ошибок, классификация ошибок, user-safe ответ, неизменённый state.
- **Typed response layer** (`app/schemas/conversation_response.py`). Discriminated union outcomes, deterministic fallback для каждого outcome.
- **Ranking/selection eval** воспроизводится offline и бесплатно.

### Слабые стороны

| # | Слабость | Evidence |
|---|---|---|
| A1 | Error handling непоследовательный: router защищён, остальные стадии нет | `conversation_flow.py:556` ловит только `ConversationRoutingError`; вызовы на `:409`, `:713`, `:730`, `:802` не обёрнуты |
| A2 | Turn orchestration в UI-слое | `ui/services/chat_handler.py:115` `process_user_message`: response generation, `save_telemetry_record`, `set_search_state`, history. Импортирует `streamlit` |
| A3 | Бизнес-презентация в `ui/` | `ui/formatters.py` (`build_display_answer`, `build_result_links`) нужна и API, и eval |
| A4 | Application result — нетипизированный `Dict[str, Any]` с ~10 shapes, различаемыми по `response_type`/`need_clarification` | `handle_user_message` → `Dict[str, Any]`; типизация появляется только в `conversation_response_adapter.py` |
| A5 | Четыре копии ADK-runner boilerplate (`_strip_json_fence` ×4, `_ensure_gemini_key` ×3), у каждой своя (или никакой) политика timeout/retry/telemetry | `intent_router.py`, `intent_update.py`, `factual_question_extraction.py`, `conversation_router.py` |
| A6 | Нет понятия «сегодня» (clock) в domain logic | `date_normalization.py:39,80`; в prompts intent/update agents нет текущей даты (`grep today` пусто) |
| A7 | Experimental soft-evidence pipeline (~2k LOC) встроен в production path `evaluate_listings` (по умолчанию выключен) | `listing_evaluation.py:880+`, `_build_soft_evidence_pipeline_policy()` → `enabled=False` |

Архитектура в целом **правильная**, её не нужно переделывать. Нужно закрыть A1, A2/A3/A4 (одним service-слоем) и A6.

---

## 3. Feature Completeness Matrix

Легенда: ✅ реализовано и проверено · 🟡 реализовано частично · 🔵 реализовано, не проверено end-to-end · 🔴 работает неправильно · ⚪ устарело/не используется

| Компонент | Статус | Комментарий / evidence |
|---|---|---|
| Conversation Router (4 actions + abstention) | ✅ / 🔵 | Детерминированный слой покрыт тестами (`test_conversation_router.py`, `test_reference_resolution.py`). Router eval сохранён (98.6% action accuracy, gemini-2.5-flash-lite), но **скорит только `action`**, а не `decision_status`/`result_scope`/`target_result_id`. Prompt менялся после отчёта (`151a7aa`) |
| Router error handling | 🟡 | Классифицированные ошибки → `routing_unavailable`. Но `except Exception: routing_error_code = "unexpected_error"; raise` (`conversation_router.py:307`) пробрасывает **не** `ConversationRoutingError`, и flow его не ловит |
| Router context для multi-turn references | 🟡 | Router видит только `current_search` + список shown results (`_build_router_prompt`, `:159`). Ни истории, ни «последнего обсуждаемого listing'а», поэтому «а у него есть парковка?» после вопроса о втором отеле разрешить нечем. Live-поведение не проверялось |
| START_SEARCH: intent extraction | ✅ / 🔴 | Eval F1 ≈ 0.79 (сохранённый). Нет timeout; retry только по строке `"429"`; ошибки пробрасываются наружу. Telemetry не учитывает system instruction |
| UPDATE_SEARCH: patch + apply | ✅ / 🔴 | `apply_intent_patch` хорошо покрыт тестами. `route_intent_update_patch_async` без timeout/retry/обработки ошибок; telemetry пишется **до** вызова (`intent_update.py:126`, `success=True`, `response_text=None`) |
| No-op update guard | ✅ | `test_update_search_no_op_guard.py` |
| Required-context clarification (city/dates) | ✅ / 🔴 | Работает для отсутствующих полей. Для `check_out <= check_in` вместо вопроса падает `ValidationError` (`SearchRequest.validate_date_range`, `query.py:31`) |
| Date normalization | 🔴 | Прошедшие даты, кросс-годовые диапазоны, нет reference date в prompts (P0-2) |
| Retrieval — fixtures | ✅ / 🔴 | Работает, но все `available_dates` в 2026 → time bomb (P0-3) |
| Retrieval — Apify | 🔵 | Код и unit-тесты на нормализацию есть, в UI выключено (`SOURCE_NAME = "fixtures"`). Live не проверялось |
| Retrieval — `apify_fast.py` | ⚪ | 635 LOC, в `app/` не используется (только tests/benchmarks) |
| Structured matching + numeric/price filters | ✅ | Хорошее unit-покрытие; price semantics (total vs per night) разобрана в e2e README |
| Textual LLM fallback (search) | 🟡 | Работает, но **последовательно** (top_k 5 × до 3 constraints) и **без обработки ошибок**: один сбой Gemini роняет весь поиск. В LISTING_QUESTION тот же fallback уже деградирует до UNCERTAIN (`result_query_matching.py`) — нужный паттерн в кодовой базе уже есть |
| Ranking & selection (strong/partial/weak tiers) | ✅ | Offline eval воспроизводится 1:1 |
| LISTING_QUESTION (selection → extraction → match → verdict → render) | ✅ / 🔵 | Хорошие unit-тесты. Нет scored eval (factual extraction set — 14 кейсов, 1.0, нерепрезентативно). Ошибка extraction LLM не обрабатывается |
| Legacy listing-question path | ⚪ | `_answer_listing_question` (`conversation_flow.py:454`) + параметр `shown_listing` — нигде не вызываются |
| Response generation (LLM + deterministic fallback) | ✅ | Timeout 10 s, fallback, source tracking. Post-generation проверки фактов для search wording нет (README честно это пишет) |
| `ConversationFailureOutcome` | 🟡 | Схема и рендер есть, но adapter **никогда его не создаёт** (`routing_unavailable` → `None` → passthrough). Typed failure path построен наполовину |
| Legacy answer generation | ⚪ | `app/logic/answer_generation_llm.py` (229 LOC) — не импортируется в `app/`/`ui/`, только в своём тесте |
| Observability: trace, per-step latency, cost | 🟡 | Работает, но: intent-стадии недооценивают input tokens (prompt без system instruction); `intent_update` логируется до вызова; LISTING_QUESTION matching не обёрнут в step; `total_observed` — сумма steps, а не wall-clock |
| Telemetry dashboard | 🔵 | Есть тест на soft-evidence часть; вручную не проверялся |
| Soft-evidence / embedding / NLI pipeline | 🔵 (experimental) | Выключен в runtime; research track. Для v1.0 **не требуется** |
| Отделение от Streamlit / API readiness | 🔴 | См. раздел 3a |

### 3a. Готовность к FastAPI

**Где UI связан с application logic**

| Что | Где сейчас | Куда должно уйти |
|---|---|---|
| Turn orchestration: load state → `handle_user_message` → typed response → LLM wording → result links → telemetry snapshot → persist telemetry → commit state | `ui/services/chat_handler.py::process_user_message` | `app/services/conversation_service.py::process_turn` |
| `build_display_answer`, `build_result_links` | `ui/formatters.py` | `app/logic/` (рядом с `answer_generation.py`) |
| `build_recent_conversation_messages` | `ui/services/chat_handler.py` | service layer |
| Session state (messages, search_state, shown_results) | `st.session_state` (`ui/state.py`) | `ConversationSession` model + `SessionStore` |
| `asyncio.run` дважды за turn | `chat_handler.run_async` | Service полностью `async`; Streamlit вызывает `asyncio.run(process_turn(...))` один раз |

**Минимальный контракт** (без новых frameworks, только Pydantic, который уже используется):

```python
# app/schemas/session.py
class ConversationSession(BaseModel):
    session_id: str
    search_request: SearchRequest | None = None
    shown_result_set: ShownResultSet | None = None
    messages: list[ConversationMessage] = []          # видимая история
    last_referenced_result_id: str | None = None      # P1-1

# app/schemas/turn.py
class ListingCard(BaseModel):          # для frontend cards
    result_id: str; position: int; title: str; url: str | None
    price_summary: str | None; match_tier: str | None
    matched: list[str]; uncertain: list[str]; failed: list[str]

class DecisionInspector(BaseModel):    # для Decision Inspector
    router_decision: dict              # action, decision_status, scope, target, reason
    effective_action: str | None
    search_request: SearchRequest | None
    constraints: list[dict]            # normalized_text / priority / mapping_status
    per_result_evidence: list[dict]    # constraint_resolution_results / ResultQueryMatch
    response_source: Literal["llm", "deterministic", "deterministic_fallback", "direct"]

class TurnResult(BaseModel):
    assistant_message: str
    outcome_kind: str                  # search | listing_query | clarification | informational | general_chat | failure
    conversation_action: ConversationAction | None
    listing_cards: list[ListingCard] = []
    inspector: DecisionInspector
    telemetry: dict                    # trace.summary()
    session: ConversationSession       # уже обновлённая

# app/services/conversation_service.py
async def process_turn(session: ConversationSession, user_message: str,
                       *, config: TurnConfig = TurnConfig()) -> TurnResult: ...
```

**Хранение state между запросами.** Для v1.0 достаточно `SessionStore` protocol с `InMemorySessionStore`, где `session_id` хранится на клиенте. `ShownResultSet` содержит полный `ListingRaw` (включая `raw` provider payload), поэтому отдавать его клиенту и принимать обратно не стоит: это лишний объём и возможность подменить state. Redis/SQLite — P2.

**Атомарность.** `process_turn` должен коммитить обновлённую сессию **только** после успешного turn'а или typed failure. Сейчас это обеспечивается случайно: при exception в Streamlit `set_search_state` просто не выполняется.

---

## 4. Evaluation & Testing Gaps

### 4.1 Состояние существующих evaluations

| Evaluation | n | Сохранённый результат | Актуальность | Надёжность метрики |
|---|---|---|---|---|
| Constraint extraction | 166 | F1 ≈ 0.79 | 🟡 отчёт от 28.04; intent agent с тех пор менялся | Средняя: не перезапускалась на текущем коде |
| Constraint resolution (textual) | 120 (114 valid) | acc ≈ 0.86, 0 YES↔NO flips | 🟡 отчёт от 29.04 | Средняя |
| Ranking & selection | 103 | top-1 0.981, leak 0 | ✅ **воспроизведено сегодня** | Высокая (детерминированно), но тестирует только `select_ranked_items` на синтетических items |
| Conversation router | 70 × 3 runs | action acc 0.986 | 🔴 prompt менялся после (`151a7aa`); скорится только `action` | Низкая для abstention/reference |
| Reference resolution minimal pairs | 19 | — | 🔴 runner есть (`scripts/evaluation/run_reference_resolution_eval.py`), сохранённого отчёта нет | Нет данных |
| Factual question extraction | 14 | 1.0 по всем метрикам | 🟡 | Низкая (n=14, отмечен как experimental set) |
| End-to-end | 50 | acc 0.66, critical false YES 48% | 🔴 см. ниже | **Низкая: контракт не зафиксирован** |
| Experiments (NLI, verifier, embeddings) | — | много отчётов | Research, не v1.0 | Не входят в v1.0 claims |

### 4.2 Почему метрикам e2e пока нельзя доверять

- `_infer_decision` (`evaluation/tasks/end_to_end/adapter.py:41`) считает **любой** непустой результат за YES. При этом система по замыслу показывает и `partial`/`weak` matches с UNCERTAIN constraints. Пересчёт с учётом `match_tier` даёт 6/25 critical false YES вместо 12/25 — то есть метрика наполовину меряет определение, а не систему.
- Predictions собраны с `top_n=5`, а приложение использует `TOP_N_DEFAULT = 2`.
- Все 50 кейсов однотуровые (`previous_state=None`). Multi-turn поведение не измеряется вообще.
- Модель, на которой собраны predictions, в отчёте не зафиксирована. `.env.example` указывает `gemini-2.0-flash`, код по умолчанию — `gemini-2.5-flash`, router report — `gemini-2.5-flash-lite`.
- В `evaluation/tasks/end_to_end/README.md` перечислены **известные баги приложения** (property type filter, кейс 0031; subjective constraints → YES вместо UNCERTAIN). Они не исправлены и не отслеживаются.

### 4.3 Пробелы в тестах

| Пробел | Почему критично |
|---|---|
| **Нет multi-turn integration tests**: ни один тест не проводит несколько turns подряд через сохраняемый state (search → listing question → update → listing question по новому `ShownResultSet`) | Ровно это показывают на demo, и именно здесь живут ошибки state transitions |
| **Нет fault-injection тестов** для intent extraction / update / factual extraction / textual fallback / retrieval | Обе воспроизведённые проблемы (crash) не пойманы 628 тестами |
| **Нет тестов на даты относительно «сегодня»** (прошлое, переход через год) | Подтверждённый баг |
| 3 теста не hermetic (реальный Gemini), 2 live-теста без маркера | Каждый `pytest` тратит деньги, тесты flaky без сети |
| `tests/_test_normalize_search_response.py` — 444 строки, отключён, 2 из 5 тестов падают | Контракт `normalize_search_response` фактически не защищён |
| Нет теста, что ошибка в стадии оставляет `SearchRequest`/`ShownResultSet` неизменными | Гарантия state для API |
| `tests/test_chat_handler_response.py` тестирует бизнес-логику через monkeypatch `streamlit` | Исчезнет после выноса service layer |

### 4.4 Какие evaluations действительно нужны перед v1.0

Минимальный набор. Всё остальное — P2.

1. **E2E с зафиксированным decision contract** + свежий прогон на финальном коде (платно, с разрешения).
2. **Multi-turn scenario eval** — 12–15 сценариев. Scoring: action на каждом turn, diff `SearchRequest`, валидность ссылки на listing, final verdict, отсутствие лишних мутаций.
3. **Router eval v2**: скорить `decision_status` / `result_scope` / `target_result_id`, слить minimal pairs, перезапустить на текущем prompt.
4. **LISTING_QUESTION scored eval** — ~30 вопросов по fixtures (structured + textual + unsupported), метрики: verdict accuracy и critical YES↔NO flips.

---

## 5. Prioritized Backlog

Сложность: **S** ≤ 0.5 дня · **M** 1–2 дня · **L** 3+ дня.

### P0 — Must fix

#### P0-1. Unhandled failures роняют turn — typed failure outcome для всех стадий
> **✅ Done (`fba5e0a`).** Реализован не через `StageFailure`, а через существующие механизмы плюс `app/llm/call_policy.py`.
>
> | Стадия | Класс | Timeout / retry | Поведение при сбое |
> |---|---|---|---|
> | Conversation router | fatal | 15 s, 2 попытки (без изменений) | `routing_unavailable` (как раньше); теперь также классифицирует `litellm.ServiceUnavailableError` / `Timeout` и httpx transport errors |
> | Intent extraction (START) | fatal | `LLM_STAGE_*`: 20 s, 3 попытки, backoff 1→2 s | `processing_unavailable`, `stage=search_intent_extraction`, прежний state |
> | Intent update (UPDATE) | fatal | то же | `processing_unavailable`, `stage=search_state_update`, прежний state |
> | Factual extraction (LISTING_QUESTION) | fatal | то же | `processing_unavailable`, `stage=listing_question_factual_extraction` |
> | Textual evidence fallback | recoverable | то же | Провал по отдельному constraint → UNCERTAIN (`reason: Textual evidence check unavailable (<code>)`), turn завершается |
> | Retrieval (Apify) | fatal | без retry (платный вызов), 180 s urlopen | `RetrievalError(code)` → `processing_unavailable`, `stage=retrieval`, прежний state (новый SearchRequest не сохраняется) |
> | Response generation | recoverable | 10 s (без изменений) | Deterministic fallback; ошибка в telemetry — только код |
>
> - Retry только для `rate_limited` / `provider_unavailable` (тот же набор, что был у router).
> - Не ретраятся: `timeout`, `authentication_error`, `provider_error`, `configuration_error`, `empty_response`, `invalid_response`.
> - Нераспознанные exceptions пробрасываются как есть. Catch-all в LISTING_QUESTION matching убран.
> - `processing_unavailable` превращается в `ConversationFailureOutcome` и рендерится без response LLM.
> - Telemetry: каждая неудачная попытка записывается только с кодом ошибки. `intent_update` больше не пишет «успешный» вызов до самого вызова.
> - Тесты: `tests/test_failure_handling.py` (66 fault-injection тестов).
>
> **Известные ограничения:**
> - Thread с вызовом textual fallback (sync google-genai client) при timeout не отменяется, а отбрасывается.
> - Если fallback-провайдер «висит», сбои по constraints копятся последовательно: до `top_k × 3 constraints × timeout`. Circuit breaker на запрос не делался; это связано с P1-6.
> - При parse failure токены не учитываются: записывается только failed call.
> - Ошибки конфигурации ADK-агентов без ключа (`Gemini` строится напрямую, не через `model_factory`) не классифицируются как `configuration_error` (P1-8).
> - Программная ошибка всё ещё роняет turn. Последний рубеж (лог + generic ответ) — задача service layer (P0-4).
- **Проблема.** Любая ошибка или timeout вне router'а пробрасывается из `handle_user_message`. В Streamlit пользователь видит traceback и не получает ответа ассистента.
- **Файлы.** `app/logic/conversation_flow.py` (`handle_user_message`, `_answer_listing_query`); `app/logic/intent_router.py::_route_intent_via_adk`; `app/logic/intent_update.py::route_intent_update_patch_async`; `app/logic/factual_question_extraction.py::extract_factual_constraints_async`; `app/logic/constraint_evidence_resolution.py::resolve_listing_constraints_with_fallback`; `app/logic/conversation_router.py:307`; `app/logic/conversation_response_adapter.py`.
- **Evidence.** (a) `pytest tests/test_reference_resolution.py` с невалидным ключом → `google.genai.errors.ClientError` выходит из `handle_user_message`. (b) Mocked START_SEARCH с датами 2026-12-28 → 2027-01-03 → uncaught `ValidationError`. (c) В коде нет `try` вокруг `:409`, `:713`, `:730`, `:802`.
- **Почему важно.** API-эндпоинт, который отдаёт 500 при rate limit Gemini, не годится для demo. Это и есть «обработка ошибок LLM» из задачи.
- **Минимальный fix.**
  1. Небольшой helper `app/llm/adk_call.py`: timeout + retry только на retryable-коды + классификация ошибок. Логику уже написанного router'а переиспользовать, а не дублировать. Перевести на него 3 оставшихся ADK-вызова.
  2. В `handle_user_message` — `StageFailure(stage, code)` → ответ `response_type="processing_unavailable"` с `previous_state`, без `shown_result_set`. Тот же паттерн, что `routing_unavailable`.
  3. Adapter: `routing_unavailable`/`processing_unavailable` → `ConversationFailureOutcome` (вместо `None`), чтобы у failure был typed путь.
  4. Textual fallback в search: ошибка по одному constraint → UNCERTAIN с `record_llm_call_failed` (как уже сделано в `result_query_matching.py`), а не отказ всего поиска.
  5. `conversation_router.py:307`: оборачивать в `ConversationRoutingError("unexpected_error")`.
- **Acceptance criteria.** Parametrized fault-injection test: для каждой стадии {router, intent, update, factual, fallback, retrieval} × {exception, timeout} `handle_user_message` возвращает typed failure, `state == previous_state`, ключа `shown_result_set` нет, trace содержит failed LLM call. Ошибка fallback по одному constraint даёт UNCERTAIN, поиск не падает.
- **Сложность.** M (1.5–2 дня).

#### P0-2. Даты: нет reference date, прошлые даты, кросс-годовой crash
> **✅ Done (`a2c41f5`).**
> - `app/config/clock.py::today()`: приоритет `use_reference_date()` → `APP_REFERENCE_DATE` → реальная дата.
> - `resolve_stay_dates` нормализует диапазон как пару, одинаково для START и UPDATE. Без года — ближайшая будущая дата; `check_out` выводится относительно `check_in`, а при частичном update — относительно существующего `check_in`. Если выведенный `check_out` дальше 90 ночей, это считается перевёрнутым диапазоном.
> - Явные прошедшие, перевёрнутые и невозможные даты, а также `nights <= 0` → `InvalidStayDatesError` → clarification. START сохраняет остальной запрос, UPDATE — предыдущий state.
> - Reference date добавлена в prompts START и UPDATE.
> - Тесты: `tests/test_stay_dates.py`.
> - **Известные ограничения (не менялись):**
>   - при update одна дата заезда по prompt'у → 1 ночь;
>   - month inheritance выключен;
>   - «явный год» определяется regex'ом и может сработать на числе вроде «budget 2000» — тогда дата от LLM берётся без пересчёта года.
- **Проблема.** Prompts intent/update не содержат текущую дату, а `normalize_*_dates` принудительно ставят текущий год.
- **Файлы.** `app/logic/date_normalization.py:39,41,80,82`; `app/agents/intent_router_agent.py`, `app/agents/intent_update_agent.py` (instruction); `app/logic/intent_router.py::build_search_request_adk_async`; `app/schemas/query.py:31`.
- **Evidence** (запуск с `today=2026-10-15`):
  - `"April 10-15"` → `2026-04-10..2026-04-15` (в прошлом), без clarification;
  - `"Dec 28 to Jan 3"` → `2026-12-28..2026-01-03` → clarification «Check-out must be later», а в полном flow — uncaught `ValidationError`;
  - относительные даты («next weekend») LLM вычисляет без знания текущей даты.
- **Почему важно.** Неправильная дата = неправильный поиск при живом Apify. Кросс-годовой диапазон роняет turn.
- **Минимальный fix.** Один источник «today» (`app/config/clock.py::today()`, переопределяемый env `APP_REFERENCE_DATE` для demo/eval). Добавить «Today is YYYY-MM-DD» в prompts intent и update. Правило нормализации: если год не указан явно → ближайшая будущая дата (≥ today); если `check_out < check_in` → перенести `check_out` на следующий год. Явно прошедшие даты → clarification. `build_search_request_adk_async` не должен конструировать невалидный `SearchRequest`: невалидный диапазон → clarification до конструирования.
- **Acceptance criteria.** Unit-тесты с инжектированным `today` на 6 кейсов (без года в прошлом/будущем; переход через год; явный год в прошлом; одна дата; nights). Invalid range возвращает clarification, а не exception.
- **Сложность.** S–M (1 день).

#### P0-3. Fixtures: time bomb для demo
> **✅ Done (`a2c41f5`), но не так, как предлагалось ниже.** Переписывать `available_dates` нельзя: `price` — это total за всё окно, а 42 из 50 e2e-запросов явно указывают 2026 год.
> - Окно доступности трактуется как ежегодно повторяющееся (сравнение по месяцу и дню, поддержаны окна через New Year). Граница 29 февраля в невисокосном году сужается: начало → 1 марта, конец → 28 февраля.
> - Tests и eval adapters закрепляют дату 2026-03-01 (`evaluation/core/reference_date.py`).
> - Проверено offline: 49 из 49 e2e-кейсов с датами дают те же даты, набор кандидатов из fixtures совпадает 1:1, ranking eval совпадает 1:1. Demo из README возвращает одинаковые результаты при «сегодня» = 2026-03 … 2028-01.
- **Проблема.** Все `available_dates` в `fixtures/listings_sample.json` лежат в 2026 (апрель + одна запись август–сентябрь). После P0-2 «April 10–15» в октябре 2026 станет 2027 → `no_results`. Без P0-2 demo сломается 1 января 2027.
- **Файлы.** `fixtures/listings_sample.json`; `app/retrieval/fixtures.py::_covers_dates`.
- **Минимальный fix.** Расширить окна доступности на 2026–2028 и сохранить исключения там, где кейсам e2e/тестам нужен date-based отказ. Либо: demo/eval запускаются с `APP_REFERENCE_DATE`. Рекомендация — **оба** варианта: данные устойчивы к времени, eval воспроизводим.
- **Acceptance criteria.** Сценарий из README даёт те же 2 результата при `APP_REFERENCE_DATE` = 2026-10-15 и = 2027-06-01. Offline-тесты retrieval зелёные. Пересчёт e2e expectations (offline audit script из e2e README) без расхождений.
- **Сложность.** S.

#### P0-4. Вынести turn orchestration из Streamlit: `process_turn` + typed contracts
> **✅ Done (`3f87a4b`).**
> - **Service:** `app/services/conversation_service.py::process_turn(session, message, config=TurnConfig()) -> TurnResult`.
> - **Contracts:** `app/schemas/session.py::ConversationSession` (session_id, SearchRequest, ShownResultSet, история ≤ 20) и `app/schemas/turn.py`:
>   - `TurnResult` — assistant_message, outcome_kind, conversation_action (`None` при routing failure), listing_cards, `DecisionInspector`, failure `{stage, code}`, telemetry summary;
>   - `session` и `debug` — `exclude=True`, т.е. доступны только in-process и в публичный JSON не попадают.
> - `ui/formatters.py` перенесён в `app/logic/display_answer.py`.
> - Streamlit — тонкий adapter: unexpected error логируется на UI boundary и показывается безопасным сообщением; session не коммитится. Следующий turn использует последнюю валидную session (проверено тестом).
> - **Проверки:**
>   - старый handler и `process_turn` дают идентичные ответы, `SearchRequest` и `ShownResultSet` в 7 сценариях;
>   - публичный `model_dump(mode="json")` проверен рекурсивно по ключам: нет `ListingRaw`, server session, debug records, provider payload;
>   - `streamlit.testing.AppTest` прогоняет полный пользовательский turn с mocks.
> - **Исправлено попутно:**
>   - UPDATE_SEARCH no-op ронял Streamlit (`update_no_op` не было в direct response types);
>   - при невалидном JSON textual fallback писал сырой ответ модели в `reason`, который виден в Decision Inspector.
> - **Ограничения / для API-слоя:**
>   - нужен `SessionStore` с lock на `session_id`: два одновременных turn'а по одной сессии — побеждает последняя запись;
>   - `listing_cards` возвращаются только в turn'ах с поиском;
>   - telemetry пишется в файл синхронно;
>   - `routing_unavailable` типизирован в service; adapter по-прежнему возвращает для него `None`.
- **Проблема.** См. 3a. Без этого FastAPI придётся дублировать `chat_handler.py`, а backend нельзя тестировать без UI.
- **Файлы.** `ui/services/chat_handler.py::process_user_message`, `build_assistant_response`, `build_recent_conversation_messages`; `ui/formatters.py`; `ui/state.py`. Новые: `app/services/conversation_service.py`, `app/schemas/session.py`, `app/schemas/turn.py`.
- **Минимальный fix.** Перенести логику **без изменения поведения**: `process_turn(session, message) -> TurnResult`. `ui/formatters.py` → `app/logic/display_answer.py` (в `ui/` оставить re-export или обновить импорты). `chat_handler.process_user_message` превращается в ~20 строк: session из `st.session_state` → `asyncio.run(process_turn)` → session обратно. `TurnResult` содержит `listing_cards` и `inspector`, собранные из уже существующих данных (`answer_payload.top_results`, `parsed_intent.router`, `constraint_resolution_results`, `listing_query_result`). Telemetry сохраняется внутри сервиса.
- **Acceptance criteria.** `grep -r "streamlit" app/` пуст. `process_turn` вызывается из теста без `streamlit`. Существующие `test_chat_handler_response.py` переписаны на `process_turn` и зелёные. UI работает как раньше (ручной smoke). `TurnResult.model_json_schema()` генерируется (готовый OpenAPI response model).
- **Сложность.** M (2 дня).

#### P0-5. Hermetic test suite
> **✅ Done (`2077102`, `baef520`).**
> - Изоляция работает уже при импорте `tests/conftest.py`, т.е. до того, как `litellm` загрузит `.env`: dotenv отключён, credentials заменены заглушками, локальный LLM-конфиг удалён.
> - Сеть к не-loopback адресам заблокирована. Любая заблокированная попытка валит тест, даже если код её проглотил.
> - Маркер `live` включается только через `BOOKING_AI_LIVE_TESTS=1`.
> - Remote cost map `litellm` отключена.
> - Self-тесты: `tests/test_offline_isolation.py`.
- **Проблема.** `import app...` подтягивает `.env` (через google-adk), и 3 теста реально вызывают Gemini; 2 live-теста не помечены.
- **Файлы.** `tests/conftest.py`; `tests/test_reference_resolution.py` (3 LISTING_QUESTION теста — не замокан `extract_factual_constraints_async`); `tests/test_embedding_google_genai_compat.py`, `tests/test_semantic_verifier_google_genai_compat.py`; `pytest.ini`.
- **Evidence.** Прогон с невалидными ключами → 5 failed.
- **Минимальный fix.** Autouse fixture в `conftest.py`: подменять `GOOGLE_API_KEY`/`GEMINI_API_KEY`/`GROQ_API_KEY`/`APIFY_TOKEN` на заглушки и блокировать сокеты (на stdlib, без новой зависимости). Marker `live` + `addopts = -m "not live"`. Замокать factual extraction в 3 тестах.
- **Acceptance criteria.** `pytest` зелёный без сети и без ключей. `pytest -m live` запускает только live-тесты.
- **Сложность.** S.

#### P0-6. Multi-turn integration tests (hermetic)
> **✅ Done.** `tests/integration/test_multi_turn_scenarios.py` — 7 сценариев из списка ниже через `process_turn`.
> - **Настоящие:** flow, state transitions, fixture retrieval, matching, ranking, reference validation, deterministic verdicts и rendering.
> - **Scripted — только LLM-границы.** Router разрешает «first/second» по полученному context. Очереди фейков обязаны быть израсходованы ровно, т.е. лишний LLM-вызов роняет тест.
> - **Mutation check.** Все 7 намеренных дефектов пойманы:
>   - snapshot не заменяется;
>   - доверие выдуманному id;
>   - нет no-op guard;
>   - пропуск истории;
>   - устаревший router context;
>   - потеря constraints при update;
>   - сброс state при failure.
> - Багов не найдено.
> - **Не покрыто:** pronoun follow-ups (P1-1), бюджет/цены (P0-8), живой router (P1-3), LLM-формулировка ответа в цепочке.
- **Проблема.** См. 4.3. Нет ни одного теста цепочки turns через сохраняемый state.
- **Зависит от.** P0-4 (тесты пишутся против `process_turn`), P0-5.
- **Минимальный fix.** `tests/integration/test_multi_turn_scenarios.py`: scripted fake LLM layer (очередь заранее заданных `ConversationActionDecision`/`IntentRoute`/`SearchIntentPatch`/`ResultQuery`), реальные fixtures, реальный matcher/ranking. Сценарии:
  1. search → «does the second one have WiFi?» → verdict для правильного listing'а;
  2. search → update budget → `ShownResultSet` заменён, `result_set_id` новый → «the first one» указывает на **новый** первый;
  3. search → update без изменений → no-op, snapshot тот же;
  4. listing question с выдуманным `target_result_id` → clarification, state не изменён;
  5. search без дат → clarification → «April 10–15» → поиск выполнен;
  6. сбой LLM на 3-м turn → failure outcome, следующий turn работает с прежним state;
  7. general chat посреди сценария → state и snapshot не изменены.
- **Acceptance criteria.** 7 сценариев зелёные, без сети, < 5 s.
- **Сложность.** M (1.5 дня).

#### P0-7. Зафиксировать e2e decision contract и пересчитать baseline
- **Проблема.** См. 4.2.
- **Файлы.** `evaluation/tasks/end_to_end/adapter.py::_infer_decision`, `run.py`, `metrics.py`; `evaluation/tasks/end_to_end/README.md`.
- **Минимальный fix.** Документировать и реализовать контракт: нет результатов → NO; top-1 `strong` → YES; только `partial`/`weak` → UNCERTAIN; clarification → UNCERTAIN. Сохранить старую метрику как `decision_any_result` для сравнения. `top_n` брать из `app.config.settings`. Записывать в report модель/профиль и git SHA. Сначала **бесплатно** пересчитать сохранённые predictions; после P0-1…P0-3 — **один** платный `predict` (только с разрешения).
- **Acceptance criteria.** Report содержит обе метрики, модель и SHA. README показывает e2e с явной формулировкой контракта. Оставшиеся critical false YES разобраны поимённо (bug / dataset / accepted limitation).
- **Сложность.** S (код) + M (triage, ограничить 1 днём).

#### P0-8. Fixture `price` — total за всё окно доступности, а не за запрошенное проживание
> **✅ Done.**
> - **Формула.** `FixturesRetriever` отдаёт копии с ценой `window_total × requested_nights / window_nights`. Используется длина исходного окна, поэтому ставка за ночь сохраняется в другом году, через Новый год и при сужении окна из-за 29 февраля.
> - **Нет данных — нет цены.** Без известной длины окна или проживания — `price=None`, т.е. UNCERTAIN downstream, а не выдуманная цена.
> - **Проверено.** `apt-large-good`, 5 ночей: $350, `$100/night` → matched, «70 per night», listing card «350.0 USD total for 5 night(s)». Тесты: `tests/retrieval/test_fixture_stay_price.py`.
> - **Влияние на e2e** (offline, сохранённые intents, fallback выключен) — меняется 9 кейсов:
>   - 0010 исправлен (было NO, стало правильное YES);
>   - 0026, 0028, 0032 (label NO) теперь показывают listing с неподтверждённым must-have — баг цены маскировал известное ограничение «отсутствие данных ≠ отказ»;
>   - 0006: top-1 зависит от textual fallback (breakfast).
>
>   Labels не менялись; разбор переносится в P0-7.
- **Проблема.** В `fixtures/listings_sample.json` `price` = ставка × ночи **всего окна** `available_dates` (пересчёт 2026-08-27, см. `evaluation/tasks/end_to_end/README.md`). Но `match_price_filters` сравнивает его с бюджетом за **запрошенные** ночи.
- **Путь цены.** `FixturesRetriever.get_candidates` (`ListingRaw.price` без изменений) → `listing_evaluation._rank_structured` → `evaluate_numeric_filters` → `match_price_filters(total_price=listing.price, check_in/check_out запроса)`. Бюджет `per_night` × ночи запроса сравнивается с total окна. Display: `normalize_search_response` выводит «per night» как `price / ночи запроса`.
- **Evidence** (offline). `apt-large-good` ($3150 за окно в 45 ночей, $70/ночь), запрос на 5 ночей:
  - `max $100 per_night` → **NO** («3150 > 500»), хотя реально 5 × $70 = $350;
  - `max $500 total_stay` → **NO**;
  - в ответе «per night» = **$630**.
- **Масштаб.** В e2e golden set у 136 из 248 пар (кандидат, кейс) запрос короче окна. В **8 из 50 кейсов** меняется price-verdict, всегда NO→YES:
  - expected YES: `0006`, `0007`, `0021`, `0023`;
  - expected UNCERTAIN: `0041`, `0043`, `0049`, `0050`.
- **Классификация.** Старый баг, не регрессия P0-3: при датах 2026 поведение то же. Ошибка всегда пессимистична (ложный NO / лишний отсев), ложных YES не создаёт. С Apify не воспроизводится: там `price` — total за запрошенное проживание.
- **Почему важно.** Fixtures — это источник для demo и всех e2e evals. Любой бюджетный запрос короче окна отсеивает подходящие варианты, а цена за ночь показывается неверно.
- **Минимальный fix.** В `FixturesRetriever` пропорционально пересчитывать цену на запрошенное проживание: `price × nights(request) / nights(window)`. Тогда `ListingRaw.price` везде означает «total за запрошенное проживание», как у Apify. Файл fixtures, matcher и display не меняются.
- **Зависимости.** Меняет e2e baseline (8 кейсов), поэтому делать **вместе с P0-7**, до платного e2e-прогона: offline-пересчёт, затем поимённый разбор этих 8 кейсов в e2e README.
- **Acceptance criteria.** Тест: `apt-large-good`, 5 ночей, `max $100/night` → YES, «per night» = $70. Запрос на всё окно даёт ту же цену, что и раньше. 8 кейсов разобраны в e2e README.
- **Сложность.** S (код) + S (разбор кейсов).

### P1 — Should fix

| ID | Проблема | Файлы / evidence | Минимальный fix | Acceptance criteria | Сложн. |
|---|---|---|---|---|---|
| P1-1 | Router не знает, какой listing обсуждался последним: «а у него есть парковка?» после вопроса про второй отель нечем разрешить | `conversation_router.py:159` `_build_router_prompt` — только state + shown list | `last_referenced_result_id` в `ConversationSession` (ставится после успешного SPECIFIC_RESULT) + 2–4 последних сообщения в router context. Проверка `target_result_id` остаётся | Hermetic test, что context передаётся; live minimal pairs для pronoun follow-ups (в router eval v2) | S–M |
| P1-2 | Multi-turn **live** scenario eval | нет | `evaluation/tasks/multi_turn/`: 12–15 сценариев в JSONL, runner поверх `process_turn`, scoring per turn | Report с per-turn action acc, state-diff acc, reference acc, verdict acc, cost/latency | M |
| P1-3 | Router eval устарел и скорит только action | `evaluation/tasks/conversation_router/*`, отчёт до `151a7aa` | Добавить scoring для `decision_status`/`result_scope`/`target_result_id`; объединить с `reference_resolution_minimal_pairs.jsonl`; перезапустить (платно) | Report на текущем prompt, метрики abstention precision/recall | M |
| P1-4 | LISTING_QUESTION без scored eval | README, `factual_question_experimental_set.jsonl` (n=14) | ~30 golden вопросов по fixtures → verdict per (listing, constraint); runner через `_answer_listing_query` | Verdict acc + critical flips в README | M |
| P1-5 | Telemetry неточна | `intent_update.py:126` (лог до вызова); `intent_router.py:100` и factual extraction: `prompt_text` без system instruction → input tokens сильно занижены; matching в LISTING_QUESTION без `trace.step`; `total_observed` = сумма steps | Логировать после вызова из `usage_metadata` (как в `conversation_response_llm.py`); step `listing_question_matching`; `wall_clock_ms` в `RequestTrace` | Тест: для каждой LLM-стадии есть ровно один record с tokens > 0 и корректным success | S–M |
| P1-6 | Последовательный textual fallback: mean 8.6 s, max 36 s в e2e | `listing_evaluation.py::_apply_constraint_fallback_layer`, `resolve_listing_constraints_with_fallback` | Bounded concurrency (переиспользовать `soft_evidence_concurrency.py`), семафор 4–5. Модель брать из policy, а не из default arg `get_gemini_model()` (вычисляется при импорте, `:453`) | p50/p90 latency e2e снижаются, результаты идентичны (тест на детерминированный порядок) | S–M |
| P1-7 | Dead/legacy code | `_answer_listing_question` + `shown_listing` (`conversation_flow.py:454`); `answer_generation_llm.py` + тест; `route_intent_adk`/`build_search_request_adk` (sync); no-op `_inherit_month_from_previous_state`; `tests/_test_normalize_search_response.py` (2 failing); ruff F811/F401 | Удалить. Отключённый test file починить (если контракт актуален) или удалить. `ruff --fix` для импортов | `ruff check app ui` без F811/F401; нет файлов `tests/_test_*` | S |
| P1-8 | Конфигурация расходится | `.env.example`: `GEMINI_MODEL="gemini-2.0-flash"` vs default `gemini-2.5-flash` vs router report `flash-lite`; `FX_CACHE_PATH` `data/` vs `app/resources/`; `tenacity`/`google.api_core` используются в eval, но не объявлены в `pyproject.toml` | Выровнять `.env.example` с defaults; объявить зависимости; одна таблица «стадия → модель» в README | Свежий clone + `.env.example` → приложение и eval используют задокументированные модели | S |
| P1-9 | Triage известных багов из e2e README (property type filter, кейс 0031; subjective constraints → YES) | `evaluation/tasks/end_to_end/README.md` | **Только** после P0-7; timebox 1 день; каждый кейс: fix / accepted limitation | Каждый оставшийся critical false YES имеет статус в README | M (timeboxed) |
| P1-10 | Репозиторная гигиена | 18 файлов `logs/` tracked вопреки `.gitignore` (сырые Apify-ответы + telemetry); `evaluation/experiments/nli_oracle` — 725 MB на диске; untracked experiment-файлы; пустые `evaluation/README.md`, `evaluation/core/json_report.py`, `paths.py` | `git rm --cached logs/...` (оставить 1 анонимизированный sample, если нужен для документации); короткий `evaluation/README.md` «core vs experiments» | `git ls-files logs` пуст (или только sample); у evaluation есть индекс | S |

### P2 — Nice to have (сознательно за рамками v1.0)

- Включение soft-evidence/embedding/NLI pipeline в runtime (сейчас это research, им и останется).
- Post-generation factual verifier для LLM-wording поисковых ответов.
- Многоязычные deterministic ответы (clarification/listing verdicts сейчас только на английском).
- Live Apify как default source; `apify_fast.py` (либо интегрировать, либо перенести в `scripts/benchmarks`).
- Persistent session store (SQLite/Redis) — для v1.0 хватит in-memory.
- Streaming ответов, async telemetry writer, concurrency-safe нумерация telemetry.
- Вынести soft-evidence hooks из `evaluate_listings` в отдельный слой (A7).
- Блокирующий `urlopen` FX API внутри async-пути при устаревшем кэше (`currency_rates.py`) → `asyncio.to_thread`.
- Полная типизация `handle_user_message` (`Dict` → union моделей). Для v1.0 достаточно, что typed boundary проходит в `process_turn`/`TurnResult`.

---

## 6. Backend v1.0 Definition of Done

**Booking AI Backend v1.0 is complete and ready for API integration**, когда выполнены все пункты:

**Reliability**
- [x] Ни одна ошибка или timeout LLM/retrieval не выходит из `process_turn` как exception: только typed `failure` outcome, и state с snapshot не изменены (fault-injection test по всем стадиям). *(P0-1 + P0-4: `TurnResult.failure`, тесты provider/routing failure через `process_turn`)*
- [x] У каждого LLM-вызова есть timeout; retry только на retryable-ошибки. *(P0-1)*
- [x] Невалидные/кросс-годовые/прошедшие даты дают корректную дату или clarification, а не crash и не тихий поиск в прошлом. *(P0-2)*
- [x] Demo-сценарий из README работает при любом `APP_REFERENCE_DATE` в 2026–2028. *(P0-3, `test_readme_demo_search_is_stable_regardless_of_today`)*
- [x] Цена в fixtures соответствует запрошенному проживанию. *(P0-8)*

**Architecture / API readiness**
- [x] `app/services/conversation_service.py::process_turn(session, message) -> TurnResult` — единственная точка входа на один turn. *(P0-4)*
- [x] В `app/` нет импортов `streamlit`; Streamlit — тонкий adapter над `process_turn`. *(P0-4, проверяется тестом)*
- [x] `ConversationSession` и `TurnResult` — Pydantic-модели с JSON schema; `TurnResult` содержит `assistant_message`, `listing_cards`, `inspector`, `telemetry`. *(P0-4)*
- [ ] Dead/legacy code из P1-7 удалён.

**Testing**
- [x] `uv run pytest` зелёный без сети и без API-ключей; live-тесты только через `-m live`. *(P0-5)*
- [x] 7 hermetic multi-turn сценариев (P0-6) зелёные. *(P0-6)*
- [x] Unit-тесты на даты с инжектированным `today`. *(P0-2)*

**Evaluation**
- [ ] E2E decision contract задокументирован; свежий e2e report на финальном коде содержит модель, SHA и `top_n`; оставшиеся critical false YES разобраны поимённо.
- [ ] В README только метрики, которые воспроизводятся текущим кодом (или помечены датой и SHA).
- [ ] *(если выполнен P1)* router eval v2 и LISTING_QUESTION eval с сохранёнными отчётами.

**Observability**
- [ ] Каждый turn даёт один telemetry record; у каждой LLM-стадии корректные success/failure, tokens и cost (P1-5); есть wall-clock latency.

**Freeze**
- [ ] README «Status / Limitations» обновлён; tag `backend-v1.0`. После этого в backend только bug fixes, нужные API/frontend.

---

## 7. Implementation Roadmap (≈ 2.5 недели)

```
Неделя 1 — Надёжность и фундамент
  D1   P0-5 hermetic tests ─────────────┐ (safety net для всего остального)
  D1-2 P0-2 даты + P0-3 fixtures ────────┤
  D2-4 P0-1 typed failures + adk_call ───┤
  D4-5 P0-4 process_turn / TurnResult ◄──┘ (после P0-1, чтобы failure outcome сразу попал в контракт)

Неделя 2 — Проверяемость
  D6-7 P0-6 multi-turn hermetic tests (зависит от P0-4)
  D6   P0-7 e2e contract + бесплатный re-score       ║ параллельно
  D7   P1-7 dead code, P1-8 config, P1-10 hygiene     ║ параллельно (S-задачи)
  D8   P1-5 telemetry, P1-6 параллельный fallback
  D9   P1-1 router context (last_referenced_result_id)

Неделя 3 — Измерения и freeze (платные прогоны только с разрешения)
  D10  P1-3 router eval v2 + P1-4 listing-question eval (подготовка датасетов offline)
  D11  ОДИН платный прогон: e2e + router v2 + listing-question (+ P1-2 multi-turn, если готов)
  D12  P1-9 triage (timebox 1 день)
  D13  README / метрики / DoD checklist → tag backend-v1.0
```

**Зависимости.** P0-5 → всё остальное · P0-2 ↔ P0-3 (делать вместе: P0-2 без P0-3 сразу ломает demo) · P0-1 → P0-4 → P0-6 / P1-1 / P1-2 · P0-1…P0-3 + P0-8 → платный e2e-прогон (P0-7) → P1-9. P0-8 делать вместе с offline-частью P0-7.

**Что можно параллелить.** P0-7 (offline re-score), P1-7, P1-8, P1-10 не зависят от service-рефакторинга. Подготовку датасетов P1-3/P1-4 можно делать в любой момент.

**Что сознательно вне v1.0.** Весь P2: soft-evidence в runtime, NLI/embeddings, live Apify по умолчанию, persistent store, streaming, multilingual, verifier для search wording.

**Когда backend считается завершённым.** Когда закрыт DoD-checklist из раздела 6 (ориентир: конец 3-й недели). Если P1-2/P1-3/P1-4 не успеваются, v1.0 закрывается без них, а в README честно пишется, что не измерено. Новые фичи после этого не начинаются.

---

## 8. First Recommended Task

**P0-5 — hermetic test suite, и сразу вслед за ним «красные» тесты на подтверждённые баги P0-1/P0-2.**

Почему именно это:
1. **Дёшево и сразу полезно.** Полдня. Сейчас каждый `pytest` тратит деньги и зависит от сети: 3 теста незаметно ходят в Gemini.
2. **Страховка для всех следующих изменений.** P0-1 и P0-4 — рефакторинг потока обработки turn'а. Без надёжного offline suite нельзя доказать, что поведение не изменилось.
3. **Баги сначала фиксируются тестом.** Fault-injection тест («LLM factual extraction бросает `ClientError`») и тест на даты («Dec 28 – Jan 3») сначала добавляются красными. Потом P0-1 и P0-2 делают их зелёными. Так видно, какая задача какой баг закрыла: это хороший материал для интервью.

Конкретный первый шаг: autouse fixture в `tests/conftest.py`, который заменяет API-ключи на заглушки и блокирует сеть, плюс marker `live` в `pytest.ini`. Затем замокать `extract_factual_constraints_async` в трёх тестах `tests/test_reference_resolution.py` и убедиться, что `pytest` зелёный без сети.

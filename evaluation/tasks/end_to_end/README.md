# End-to-end evaluation

Тестирует полный пайплайн (`app.logic.conversation_flow.handle_user_message`) на golden-датасете
`evaluation/datasets/end_to_end/end2end_golden_set.jsonl`: пользовательский запрос → YES/NO/UNCERTAIN
+ выбранный listing. Источник listing'ов — `fixtures/listings_sample.json` (`source="fixtures"`).

Запуск:
```
python -m evaluation.tasks.end_to_end.run predict [--limit N] [--top-n N] [--max-llm-calls N]  # ПЛАТНО; пишет predictions.jsonl + run_metadata.json
python -m evaluation.tasks.end_to_end.run evaluate                        # бесплатно: пересчёт сохранённых predictions по текущему контракту
python -m evaluation.tasks.end_to_end.run replay [--top-n N]              # бесплатно: сохранённые intents через текущий backend, без LLM
python -m evaluation.tasks.end_to_end.run all [--limit N] [--top-n N] [--max-llm-calls N]      # ПЛАТНО: predict + evaluate
```

`--max-llm-calls` — ограничитель расходов: перед каждым кейсом сравнивается число LLM-вызовов из
telemetry (включая неудачные попытки), и прогон останавливается при достижении лимита (последний кейс
может превысить его на свои вызовы). Итог пишется в `run_metadata.json` → `execution`.

Настройки прогона по умолчанию совпадают с production turn (`TurnConfig`): `top_n=2`,
`max_items=40`, textual fallback `top_k=5`, `source=fixtures`, reference date `2026-03-01`.
`--output-dir` (глобальный флаг, до команды) задаёт папку прогона; по умолчанию `evaluation/outputs/end_to_end`.

CLI загружает `.env` до импорта `app` (`env.py`), как `ui/streamlit_app.py`. Раньше `.env` подхватывался
позже, и часть настроек «замерзала» на значениях по умолчанию: router шёл через `gemini_default` вместо
production `groq_gpt_oss_20b`, а textual fallback — через `gemini-2.5-flash` вместо `GEMINI_MODEL`.
Прогон 2026-09-23 почти наверняка выполнен на такой смеси моделей.

**Архив прогона 2026-09-23:** `evaluation/outputs/end_to_end/legacy_2026-09-23/` — исходные
`predictions.jsonl`, оригинальные v1-отчёты (`*_v1_original.*`), пересчёт по v2 (`report.json`) и replay.

## Decision contract v2 (`contract.py`, P0-7)

Три уровня, которые раньше смешивались:

1. **Eligibility** — показывать ли вариант. Listing скрывается только при *подтверждённом* нарушении
   (structured NO, явное отрицательное evidence, NO от fallback, несовпадение property type, цена NO).
   Система работает в **open-world**: если удобства нет в данных, оно «не подтверждено», а не «отсутствует».
2. **Constraint satisfaction** — по каждому требованию: matched / uncertain / failed.
3. **Overall decision** — что система вправе утверждать пользователю (по top-1 показанному результату;
   `strong` всегда ранжируется первым):

| Ответ системы | Решение | basis |
|---|---|---|
| top-1 `match_tier == strong` (все must, property type и числовые фильтры подтверждены) | YES | `strong_match` |
| top-1 `partial` / `weak` | UNCERTAIN | `partial_match` / `weak_match` |
| уточняющий вопрос | UNCERTAIN | `clarification` |
| ничего не показано | NO | `no_results` |
| typed failure (provider / retrieval) | ошибка, не решение | `turn_failure` |

Решения пересчитываются из сохранённого `raw_response`, поэтому любой старый прогон можно бесплатно
переоценить. Старый контракт («любой результат = YES») оставлен в отчёте как блок `legacy`.

**Метрики:** `decision.accuracy` (строгое совпадение), `critical_false_yes` (gold NO/UNCERTAIN → YES —
главная метрика безопасности), `false_no` (gold YES → ничего не показано), `conservative`
(UNCERTAIN вместо YES/NO — ошибка, но без ложного утверждения), top-1 / top-k / MRR, `legacy.*`.
Каждый отчёт содержит `run_metadata` (git SHA + dirty, модели по стадиям, `top_n`, fallback policy,
sha256 датасета, reference date) и `evaluation_metadata`.

**Чего контракт не утверждает:** `strong` покрывает только *извлечённые* требования. Требование,
пропущенное intent extraction или сохранённое как nice-to-have (например, субъективное), не мешает
`strong` — такие случаи видны как critical false YES и разбираются ниже, а не маскируются контрактом.

**Семантика golden labels:** NO / YES в датасете размечены в **closed-world** — удобство, которого нет в
`facilities` listing'а, считается отсутствующим. Это расходится с open-world поведением системы и
объясняет большую часть `conservative` ошибок (см. triage). Labels не менялись.

## Эксперимент 2026-08-27: аудит датасета и фикстур

**Повод:** подозрение, что golden-датасет некорректен — метрики (в частности
`critical_false_yes_rate`) выглядели неправдоподобно то ли завышенными, то ли заниженными.

### Методология

1. Детерминированный аудит-скрипт (без LLM): для каждого кейса проверял, действительно ли
   какой-то listing во `fixtures/listings_sample.json` satisfies его `hard_constraints`
   (city/property_type/currency/budget/dates/must_have/must_not_have/pets/smoking/size), и
   сравнивал результат с `expected_decision`/`expected_selected_id`/`acceptable_ids`.
2. Реальный прогон pipeline (`run all`) — для проверки поведения живой системы, не только
   статических данных.
3. Разбор конкретных расхождений на уровне `raw_response` (`request_summary`, `matched_constraints`,
   `debug_notes`) — чтобы отличить баг датасета от бага приложения.
4. Проверка реальных сырых ответов Apify (`logs/apify_raw/*.json`) для установления фактической
   семантики поля `price`.

### Найденные и исправленные проблемы датасета/фикстур

- **`fixtures/listings_sample.json`: дубликаты id.** `hotel-large` использовался для двух разных
  listing'ов (Баку и Тбилиси) → переименовал тбилисский в `tbilisi-hotel-large`. `manila-apt-1` был
  побайтово продублирован → удалил дубль.
- **`e2e_final_0001`**: `expected_selected_id` был `baku-apt-11` ($140/ночь), хотя `budget_max=130`
  и поле `target_listing_id` в той же строке верно указывало `baku-apt-8`. Исправлено на
  `baku-apt-8`, `baku-apt-11` убран из `acceptable_ids`.
- **`e2e_final_0012`, `e2e_final_0014`**: в `acceptable_ids` не хватало `baku-apt-smoking` /
  `baku-apt-nonsmoking` — эти фикстуры были добавлены позже (для юнит-тестов smoking-policy), но
  реально удовлетворяют constraints этих кейсов и не были учтены при скоринге. Добавлены.
- **`e2e_final_0034`, `e2e_final_0036`, `e2e_final_0039`** (labeled `NO`): реально существовал
  listing, полностью satisfying все hard_constraints — т.е. кейсы были неверно размечены. Ужесточил
  constraints (min_size_sqm / budget_max) так, чтобы ни один listing не подходил и NO стал
  действительно верным, а не искусственным.

### Ключевая находка: семантика цены `price`

`app/logic/numeric_filters.py::match_price_filters` при `scope="per_night"` умножает
`budget_max` на число ночей и сравнивает с `listing.price` — то есть код предполагает, что
`listing.price` = **total за весь стей**, а не ставка за ночь. Проверка реальных логов Apify
(`logs/apify_raw/*.json`, десятки запросов на Баку с 2–14 ночами) подтвердила: подразумеваемая
цена за ночь (`price / nights`) стабильна (~$80–130) независимо от длины стея, а сырое `price`
растёт пропорционально числу ночей → **это действительно total, код написан правильно**.

Фикстуры же были заполнены ставками **за ночь** — несоответствие между кодом (написан под total) и
тестовыми данными (per-night) обесценивало почти все негативные тесты с длинным стеем: бюджет
раздувался в N раз и почти никогда не отсекал listing по цене.

**Исправление:** пересчитаны все 48 цен в `fixtures/listings_sample.json`:
`price_total = исходная_ставка_за_ночь × nights(available_dates)`.

### Метрики до/после фикса цены (тот же прогон pipeline, одинаковый датасет)

| метрика | price = за ночь (баг) | price = total (исправлено) |
|---|---|---|
| `decision_accuracy` | 0.469 | **0.660** |
| `critical_false_yes_rate` | 87.5% (21/24) | **48%** (12/25) |
| `top1_selection_accuracy` | 0.80 | **0.84** |
| `NO → YES` (confusion matrix) | 14 | **4** |
| errors | 1 | **0** |

Разница подтверждает: искажённая семантика цены в фикстурах была главным источником неправдоподобных
метрик, а не реальное качество системы.

### Метрика `critical_false_yes_rate` — исправлен знаменатель

В `evaluation/tasks/end_to_end/metrics.py` знаменатель считался как `len(ok_rows)` (все кейсы), а
не как количество кейсов, где ложный YES вообще возможен (`expected_decision in {NO, UNCERTAIN}`).
Добавлено поле `rejection_total`, знаменатель исправлен на него.

### Осознанно НЕ исправлено: двусмысленность текста запроса

4 кейса (`e2e_final_0005`, `0010`, `0011`, `0029`) дают false negative/false positive из-за того,
что *сам текст пользовательского запроса* двусмысленный (например "cost under $480" без "per
night", или "in April 2026" без точных дат). Первая попытка — переписать текст запроса, чтобы убрать
двусмысленность — была **отменена**: реальные пользователи не всегда формулируют запросы идеально,
и датасет должен тестировать, как система ведёт себя именно с такими, естественно двусмысленными
формулировками, а не только с "причёсанными". Тексты запросов возвращены в исходный вид.

Это значит: указанные 4 кейса — не баг датасета, а зафиксированный сигнал о том, как система
интерпретирует неоднозначность (per-night vs total-stay scope, отсутствие точных дат) — что дальше
с этим делать (улучшать extraction, задавать уточняющий вопрос) — вопрос уже к логике приложения,
а не к датасету.

### Оставшиеся находки — баги приложения, не датасета

Обнаружены при разборе failures после фикса цены (**НЕ исправлялись** в рамках этой сессии —
изменения датасета их не касаются):

- **Property type filter иногда не срабатывает** (`e2e_final_0031`): запросили `hotel`, система
  вернула listing с `property_type=apartment`; в `matched_constraints` нет вообще записи про
  property_type.
- **Must-have constraint без evidence тихо пропускается вместо отказа** (`e2e_final_0037`,
  `0038`, и ранее `0039` до фикса): `_fails_must()` в `app/logic/listing_evaluation.py` отклоняет
  listing только при явном `Ternary.NO`; если данных о поле просто нет (`UNCERTAIN`), constraint
  считается выполненным. Также встречается case, где ограничение вообще не смаплено на известное
  поле (`mapped_fields: []`, например "no pets", "sea view") — физически нечем проверить.
- **UNCERTAIN-детекция не срабатывает на субъективные формулировки**: 8 из 9 uncertain-кейсов
  система уверенно вернула YES вместо того чтобы распознать субъективную/неподтверждаемую часть
  запроса ("romantic atmosphere", "feels safe and premium", "good for remote work") и уйти в
  уточняющий вопрос.

## Датасет: описание полей

- `hard_constraints` — эталонный разбор запроса (city/property_type/budget_max/…), используется
  только для оффлайн-аудита; в сам pipeline не передаётся (pipeline парсит `user_query` заново
  через LLM).
- `expected_selected_id` / `acceptable_ids` — какой id (или один из нескольких) допустим как top-1
  ответ; используется только когда `expected_decision == "YES"`.
- `target_listing_id` — присутствует в каждой строке, но нигде не используется в
  `comparator.py`/`metrics.py`. Стоит либо начать использовать как sanity-check (например,
  `assert target_listing_id in acceptable_ids` при загрузке датасета — фикс кейса 0001 показал,
  что расхождение между этими двумя полями как раз и было багом), либо убрать как мёртвое поле.


## P0-7: triage ошибок (2026-10-15)

Источник: бесплатный re-score сохранённых predictions (`legacy_2026-09-23/`, backend до P0-1…P0-8,
`top_n=5`) по контракту v2 и replay сохранённых intents через текущий backend (без LLM, textual fallback
выключен; после исправлений D3/breakfast: critical false YES 4/25, единственное изменение — 0037).
Re-score: decision accuracy **0.62**, critical false YES **6/25**, false NO **4/25**,
conservative YES→UNCERTAIN 6, NO→UNCERTAIN 2; legacy-блок воспроизводит прежние 0.66 / 12.

| Кейс | Gold → система | Причина | Категория | Текущий backend |
|---|---|---|---|---|
| 0029 | NO → **YES** | «it would be great if smoking was allowed» — явное nice-to-have, извлечено верно; по D3 оно не блокирует YES. Gold считает smoking hard | label расходится с продуктовой политикой (D3) | остаётся — ожидаемо |
| 0037 | NO → **YES** | «I really want a sea view» извлечено как nice (должно быть must); «I won't be bringing any pets» превращено в must «no pets» (это не требование к жилью), которое к тому же игнорировалось (known без mapped_fields) | intent extraction (priority, лишний constraint) + runtime-баг unmapped must | replay → UNCERTAIN (из-за «no pets»); исправление priority — промпт, проверка в платном прогоне |
| 0041, 0047, 0050 | UNCERTAIN → **YES** | прямая просьба («looking for … nice atmosphere», «Do you have anything with…», «Could you find … trustworthy») извлечена как nice вопреки правилу промпта | intent extraction (priority) | промпт уточнён примерами; проверка в платном прогоне |
| 0049 | UNCERTAIN → **YES** | «I'd really prefer … local» — мягкая формулировка → nice; по D3 не блокирует YES | label расходится с политикой D3 (пограничный) | остаётся — ожидаемо |
| 0002, 0004, 0012, 0014, 0018, 0025 | YES → UNCERTAIN | отрицательное требование («no gym / breakfast / pool / conference room») нельзя подтвердить в open-world; gold — closed-world. В 0002 также «breakfast» → `coffee_machine`, «balcony» как nice | расхождение open/closed-world (+ intent mapping в 0002) | без изменений |
| 0031, 0038 | NO → UNCERTAIN | must-have не подтверждён — показан как partial/weak | расхождение open/closed-world | 0031 → NO (safety net по property type) |
| 0026, 0028, 0032, 0034 | NO → UNCERTAIN (после P0-8) | исправленная цена больше не отсеивает listing с неподтверждённым must-have; в 0034 также «breakfast» → `coffee_machine` | расхождение open/closed-world, ранее скрытое багом цены | новое поведение |
| 0005, 0011 | YES → NO | «under $480», «$400 maximum» за месяц → `total_stay`; gold считает per-night | неоднозначный запрос (scope бюджета) | без изменений |
| 0045 | UNCERTAIN → NO | то же (Dubai, $400 → total) | неоднозначный запрос | без изменений |
| 0008 | YES → NO | город извлечён как «Batum» | intent extraction (нормализация города) | без изменений |
| 0010 | YES → NO | цена за окно (P0-8) + «April 2026» → 1 ночь | backend-баг | исправлено (replay: YES) |
| 0006 | YES → YES | после P0-8 top-1 зависит от textual fallback (breakfast) | ranking + fallback | проверить в платном прогоне |
| 0023 | YES → YES | в gold `property_type=apartment`, в запросе его нет | label с допущением (на top-1 не влияет) | — |

Систематические проблемы intent extraction: «breakfast» маппится на `coffee_machine` (нет canonical
поля — LLM выбирает ближайшее); жёсткие требования понижаются до nice-to-have.

### Продуктовые решения (приняты 2026-10-15)

- **D1 — open-world для must-have.** Неподтверждённый must-have → listing можно показать как partial, решение
  UNCERTAIN; отсутствие информации не трактуется как NO.
- **D2 — open-world для запретов.** Отсутствие информации о запрещённом свойстве не доказывает, что запрет
  соблюдён → UNCERTAIN. Closed-world ради метрик не вводится.
- **D3 — нет безоговорочного YES при неподтверждённом обязательном требовании.** Неподтверждённый MUST
  (в т. ч. субъективный) не даёт `strong`; явное nice-to-have не блокирует YES, если все hard-требования
  подтверждены. Ключевое vs необязательное определяется priority из intent extraction («must» для прямой
  просьбы, «nice» только при явном смягчении).

Как это реализовано (classifier tiers не менялись — их семантика уже соответствовала D1–D3, что
подтверждено `tests/test_requirement_semantics.py`):

- `UserConstraint` (`app/schemas/constraints.py`) — инварианты маппинга при создании/загрузке constraint:
  - meal plan («breakfast», «half board», «all inclusive»…) никогда не маппится на amenity-поле —
    раньше «breakfast» → `coffee_machine`, и listing с кофемашиной «подтверждал» завтрак; теперь проверка
    идёт по тексту listing'а;
  - `known` без единого mapped field (кроме категорий со своим слотом: property type, occupancy, numeric)
    → unresolved/textual; раньше такой MUST/FORBIDDEN молча игнорировался.
- Промпт intent extraction дополнен примерами: прямые субъективные просьбы → must, явное смягчение → nice,
  факты о путешественнике («I won't be bringing any pets») — не constraint.


## Процедура платного прогона

1. Чистое дерево (`run_metadata.git.dirty == false`).
2. Smoke в отдельную папку, без полного прогона:
   ```
   uv run python -m evaluation.tasks.end_to_end.run --output-dir evaluation/outputs/end_to_end/smoke_<date> predict --limit 3 --max-llm-calls 40
   uv run python -m evaluation.tasks.end_to_end.run --output-dir evaluation/outputs/end_to_end/smoke_<date> evaluate --limit 3
   ```
3. Проверить `run_metadata.json` (SHA, модели, `top_n`, reference date, `execution`) и ошибки.
4. Полный прогон — только после отдельного решения: `--output-dir evaluation/outputs/end_to_end/baseline_<date> all --max-llm-calls 600`.

Predictions пишутся построчно с flush, `run_metadata.json` — до первого кейса, поэтому прерванный прогон
не теряет уже обработанные кейсы. Лимит LLM-вызовов считает все попытки, которые делает наш код
(router, intent, update, factual, textual fallback, включая повторы); внутренние HTTP-повторы SDK, если они
есть, в telemetry не видны.


## Baseline 2026-10-15 — до targeted fixes

Артефакты: `evaluation/outputs/end_to_end/baseline_gemini_router_2026-10-15/` (`predictions.jsonl`,
`run_metadata.json`, `report.json`, `failures.jsonl`; не изменялись после прогона, секретов не содержат).

- **Конфигурация:** commit `961cef8` (`dirty=false`); все стадии — `gemini-2.5-flash-lite`
  (`ROUTER_LLM_PROFILE=gemini_default`); `top_n=2`, `max_items=40`, fallback `top_k=5`; reference date `2026-03-01`.
- **Выполнение:** 50/50 кейсов, `valid_as_baseline=true`, 0 provider failures, 1 `invalid_response`
  (0002, intent extraction), 289 LLM-вызовов (router 50, intent 50, textual fallback 189), latency на кейс
  mean 8.3 s / p90 14.9 s / max 17.1 s, оценка стоимости по telemetry $0.053 (занижена, P1-5).

| Метрика (контракт v2, исходные labels) | Значение |
|---|---|
| Decision accuracy | 28/49 = 0.571 |
| Critical false YES | 6/25 |
| False NO | 5/24 |
| Conservative | YES→UNCERTAIN 6, NO→UNCERTAIN 3 |
| Top-1 acceptable | 19/24 (каждый промах — кейс без результатов) |
| Legacy-контракт | accuracy 0.592, critical false YES 13 |

Предыдущий baseline на Groq-router (`baseline_2026-10-15`, не сохранён в Git) невалиден: 46/50 кейсов
упали на лимите Groq 8K tokens/minute.

### Разбор ошибок (22 кейса)

| Категория | Кейсы |
|---|---|
| Scope бюджета: «$200 max» на месяц → `total_stay`, gold — per night | 0005, 0011, 0012, 0025 (false NO), 0045 |
| Даты: «in April 2026» → выезд 1 мая; только дата заезда → 1 ночь; 90-ночная защита без переноса года | 0010, 0028, 0032 |
| Open-world, отрицательные требования (D2) | 0004, 0014, 0018 |
| Open-world, неподтверждённый must (D1) | 0026, 0038 |
| Субъективный must не подтверждён (D3; нестабильно между прогонами) | 0001, 0003, 0023 |
| Label расходится с D3 (явное nice-to-have) | 0029, 0043, 0049 |
| Ошибка golden label (`pets_allowed:false` из «I won't be bringing pets») | 0037 |
| Textual fallback подтвердил субъективное требование косвенной ассоциацией («quiet» → «nice atmosphere») | 0041 |
| Невалидный JSON от intent extraction | 0002 |

Critical false YES: 0028, 0029, 0037, 0041, 0043, 0049. Из них ошибки системы — 0041 (evidence) и 0028
(молчаливая 1 ночь; кроме того, gold придумал дату выезда); 0049 — пробел UX: неподтверждённое
nice-to-have не показывается пользователю.

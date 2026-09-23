# End-to-end evaluation

Тестирует полный пайплайн (`app.logic.conversation_flow.handle_user_message`) на golden-датасете
`evaluation/datasets/end_to_end/end2end_golden_set.jsonl`: пользовательский запрос → YES/NO/UNCERTAIN
+ выбранный listing. Источник listing'ов — `fixtures/listings_sample.json` (`source="fixtures"`).

Запуск:
```
python -m evaluation.tasks.end_to_end.run predict   # прогоняет pipeline, пишет predictions.jsonl (стоит денег/времени — LLM-вызовы)
python -m evaluation.tasks.end_to_end.run evaluate   # только сравнение predictions.jsonl с датасетом, без API-вызовов
python -m evaluation.tasks.end_to_end.run all
```

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

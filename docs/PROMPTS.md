# PROMPTS.md — плейбук для Claude Code

Як користуватись: відкрий термінал у корені репо → `claude` → копіюй промпти
по черзі. Один промпт = одна сесія/задача. Для задач, позначених [PLAN],
спочатку увімкни Plan mode (Shift+Tab) і затверди план перед написанням коду.
Після кожної задачі: перевір результат → "запусти тести" → "закоміть зміни".

---

## Етап 1 — Каркас

1.1 [PLAN]
Прочитай CLAUDE.md і docs/SPEC.md. Створи каркас бекенда: структуру папок з
розділу 2 специфікації, backend/requirements.txt з усіма залежностями зі
стека, app/config.py на pydantic-settings (усі налаштування з .env, додай
.env.example), app/db/session.py (async SQLAlchemy, SQLite за замовчуванням,
готовність до PostgreSQL через DATABASE_URL).

1.2
Реалізуй усі моделі SQLAlchemy за схемою з розділу 3 SPEC.md у
app/db/models.py. Налаштуй Alembic і згенеруй першу міграцію. Напиши
pytest-тести: створення кожної моделі, унікальність tg_id і client_order_id,
JSON-поля.

1.3
Створи app/main.py: FastAPI з healthcheck GET /health, CORS для домену
фронтенда з конфіга, structlog з фільтром секретів (будь-який рядок, схожий
на API-ключ або токен, маскується). Тест на healthcheck.

1.4
Налаштуй GitHub Actions (.github/workflows/ci.yml): на push і PR —
pytest з pytest-cov, ruff check, mypy. Налаштуй pyproject.toml для ruff
і mypy (строгість розумна, не максимальна). Додай у README.md бейджі
CI і coverage. Поріг coverage: завал білда при < 80%.

## Етап 2 — Дані

2.1 [PLAN]
Реалізуй app/collectors/market_ws.py: підписка через ccxt на OHLCV для
пар і таймфреймів з конфіга (15-20 пар; 15m/1h/4h). Вимоги: auto-reconnect
з exponential backoff, heartbeat, після реконекту — gap-fill пропущених
свічок через REST, запис у candles (upsert по PK). Окремий entrypoint
для запуску як systemd-сервіс. Тести з моками ccxt.

2.2
Створи scripts/backfill.py: завантаження історії OHLCV за N днів (аргументи
CLI: --symbols, --tf, --days) через ccxt REST з повагою до rate limits.
Прогрес-бар, ідемпотентність (повторний запуск не дублює).

2.3
Реалізуй app/collectors/derivatives.py: кожні 5 хв (APScheduler) тягне
funding rate, open interest, long/short ratio з Futures API біржі для
наших пар; зберігає останні значення (окрема таблиця derivatives_snapshot —
додай модель і міграцію). Retry і логування за правилами CLAUDE.md.

## Етап 3 — SMC

3.1 [PLAN]
Реалізуй app/analysis/smc.py як обгортку над бібліотекою smartmoneyconcepts:
функції, що на DataFrame свічок повертають структуру (BOS/CHoCH), order
blocks, FVG, liquidity sweeps, equal highs/lows, premium/discount зони.
Уніфікований вихідний формат zones JSON з розділу 3 SPEC.md (signals.zones).
Тести на синтетичних свічках з відомими патернами.

3.2
app/analysis/indicators.py на pandas-ta: ATR (для відстані стопа), обсягові
метрики, EMA 50/200 для фільтра тренду. Тести.

3.3
Створи тимчасову сторінку scripts/preview_chart.html + невеликий ендпоінт
GET /debug/zones?symbol&tf, щоб очима перевірити розмітку зон на
Lightweight Charts поверх реальної історії з БД. (Видалимо після Етапу 8.)

## Етап 4 — Новини

4.1
app/collectors/news.py: CryptoPanic API (ключ з .env) + RSS (CoinDesk,
Cointelegraph, The Block) через feedparser + Fear & Greed
(api.alternative.me/fng/). Кожні 10 хв, дедуплікація по url, мапінг
новина→symbols за тикерами/назвами монет. Запис у news_items.

4.2 [PLAN]
app/analysis/sentiment.py: виклик Gemini API для нових news_items —
промпт вимагає СУВОРО JSON {"sentiment": -10..10, "importance": 1..5}.
Парсинг зі зняттям можливих ```json огорож, retry, фолбек: при відмові
Gemini новина лишається без сентименту (не блокує пайплайн).
Економний батчинг (кілька новин одним запитом).

4.3
Макрокалендар: статичний конфіг найближчих FOMC/CPI дат (config або
таблиця macro_events) + функція is_macro_window(now) -> bool (±30 хв).

## Етап 5 — Сигнали і бот

5.1 [PLAN]
app/analysis/scoring.py і engine.py: повний пайплайн з розділу 5 SPEC.md,
ваги з розділу 6 у конфізі (не хардкод). Створення signal у БД.
Тести: синтетичні сценарії → очікуваний score.

5.2
app/bot/: aiogram 3, хендлери /start (з обовʼязковим дисклеймером і
збереженням disclaimer_accepted_at), /signals (останні активні), /stats,
/help. Webhook або long polling — обери і обґрунтуй у docs/DECISIONS.md.

5.3
app/bot/alerts.py: при новому сигналі — рендер PNG графіка (mplfinance:
свічки + зони + рівні entry/SL/TP) + текст формату з розділу 1 SPEC.md
(score + історичний win rate, БЕЗ "шансів у %") + кнопка-deep-link у
Mini App. Розсилка юзерам за тарифом (Free: затримка 15 хв, тільки BTC/ETH).

5.4
Gemini-пояснення сигналу: app/analysis/engine.py після створення сигналу
викликає sentiment.py/explain(signal_json) з жорстким промптом (тільки
передані факти, 4-6 речень, українською), кешує в signals.ai_explanation.

## Етап 6 — Бектест

6.1 [PLAN]
scripts/backtest.py на vectorbt: проганяє пайплайн по історії з candles
(ТІЛЬКИ закриті свічки, перевір відсутність lookahead — додай тест, який
ловить використання даних з майбутнього), симулює угоди за правилами
розділів 6-7. Звіт: win rate, profit factor, max drawdown, розбивка по
score-бакетах 70-79/80-89/90+, по символах. Експорт у JSON для /stats.

6.2
Walk-forward: розбиття історії на вікна, оптимізація ваг скорингу на
train-вікні, перевірка на test-вікні. Підсумкові ваги → конфіг.

## Етап 7 — Paper trading

7.1
Режим paper: positions з exchange='paper', виконання за цінами з candles
(вхід по торканню зони, вихід по SL/TP), без жодних викликів бірж.
Щотижневий автозвіт у бот адміну і в публічну статистику GET /stats.

## Етап 8 — Mini App

8.1 [PLAN]
frontend/: каркас Mini App (vanilla JS), telegram-web-app.js, екран стрічки
сигналів з даними з GET /signals, авторизація: initData у заголовку
X-Telegram-Init-Data → POST /auth/telegram → JWT. Дизайн: темна тема,
акуратно, без перевантаження.

8.2
frontend/src/chart.js: Lightweight Charts, свічки з GET /chart, поверх —
зони (прямокутники OB/FVG), лінії entry/SL/TP з підписами, мітки
BOS/CHoCH/sweep. Перемикач 4h/15m. Оновлення раз на 15 сек.

8.3
Екран деталей сигналу: графік + AI-пояснення + фактори списком + статус.
Екрани: статистика (з /stats), налаштування (монети, TF, мін.score),
тарифи. Обробка startapp=signal_<id> deep link.

8.4
Бекенд-ендпоінти: /auth/telegram (HMAC-валідація initData + auth_date<1h,
JWT 15 хв), /signals, /signals/{id}, /chart, /stats. slowapi rate limits.
Тести валідації initData (валідний/підроблений/протухлий).

8.5
Деплой фронтенда на Cloudflare Workers (wrangler.toml, інструкція в
docs/DEPLOY.md), бекенд: deploy/*.service systemd-юніти (api, collector,
bot) з hardening з розділу 12, інструкція установки на Ubuntu 22.04.

8.6
Демо-режим: кнопка "Demo" на стартовому екрані Mini App (і окремий
web-доступ без Telegram) — інтерфейс з 5-10 реальними історичними
сигналами (read-only, дані з спец-ендпоінта GET /demo/signals без auth,
з rate limit). Мета: рекрутер/клієнт бачить продукт за 30 секунд без
реєстрації.

## Етап 9 — Білінг

9.1 [PLAN]
Stars-підписка: createInvoiceLink(currency="XTR",
subscription_period=2592000) для Pro і Auto; хендлери pre_checkout_query
і successful_payment (включно з is_recurring → продовження plan_until =
subscription_expiration_date); запис у payments. Команда /cancel →
editUserStarSubscription. APScheduler щогодини: plan_until < now()-24h →
plan='free' + повідомлення з кнопкою продовження.

9.2
Крипто-оплата через NOWPayments: інвойс на 30 днів зі знижкою з конфіга,
webhook підтвердження → plan_until += 30 днів. Нагадування за 3 дні.

## Етап 10 — Автотрейдинг (тільки після 4+ тижнів paper trading!)

10.1 [PLAN]
app/security/crypto.py: AES-256-GCM (cryptography), майстер-ключ з
secrets.env, унікальний nonce на запис. Тести: шифрування/розшифрування,
зіпсований nonce → виняток.
api/keys.py: POST /keys → перевірка прав ключа через API біржі
(якщо є withdrawal — HTTP 422 з поясненням) → шифрування → БД;
відповідь містить ТІЛЬКИ label_mask. DELETE /keys/{id}.

10.2 [PLAN]
trading/risk.py: розрахунок розміру позиції і всі ліміти з розділу 7.
Максимально простий код + вичерпні тести крайніх випадків.
Познач REVIEW-коментарями.

10.3 [PLAN]
trading/executor.py (спершу ТІЛЬКИ Bybit testnet, ключ з .env):
лімітний вхід з clientOrderId=UUID → після філу нативні біржові SL/TP →
audit_log на кожен крок. api/trade.py: POST /trade/confirm (тільки plan
auto, тільки активний не-протухлий сигнал), POST /trade/panic.
Тести з моками ccxt: ідемпотентність (повторний confirm не створює
другий ордер), відмова на протухлий сигнал.

10.4
trading/monitor.py: відстеження позицій (поллінг біржі), TP1 → стоп у
беззбиток, пуші в бот, закриття → pnl у positions. Глобальний kill switch
адміна (команда в боті + флаг у БД, який executor перевіряє перед кожним
ордером).

## Етап 11 — Презентація (портфоліо)

11.1
Перепиши README.md (англійською, + README.uk.md українською): одне речення
суті → секція скріншотів/GIF (плейсхолдери, я додам файли сам) → Mermaid-
діаграма архітектури (collectors → analysis engine → bot/api → mini app;
окремо trading flow) → "Key engineering decisions" (non-custodial,
exchange-side stops, walk-forward backtesting, lookahead-bias protection,
AES-256-GCM key storage) → стек → живі метрики (посилання на /stats) →
бейджі CI/coverage.

11.2
Створи docs/CASE-STUDY.md: проблема → аналіз аналогів (таблиця з SPEC) →
архітектура → 3 найскладніші технічні рішення з фрагментами коду
(WebSocket reconnect+gap-fill, lookahead-захист у бектесті, шифрування
ключів) → результати (метрики paper trading) → що б покращив далі.
Стиль: технічна стаття, англійською.

11.3
Підготуй docs/INTERVIEW-NOTES.md (тільки для мене, у .gitignore приватної
частини): розповідь на 3 хв про walk-forward + lookahead bias простими
словами, типові питання співбесіди по проєкту і відповіді.


# SmartFlow — повна специфікація

## 1. Що це
SaaS крипто-аналітики: Telegram-бот + Telegram Mini App (кросплатформність
через Telegram: ПК/Android/iOS одним кодбейзом).

Бекенд безперервно збирає ринкові дані з бірж, детектить сетапи за Smart
Money Concepts, скорить їх за конфлюенсом факторів (структура, ліквідність,
деривативи, новинний сентимент), шле алерти в бот. У Mini App — інтерактивний
графік з розміченими зонами, AI-пояснення, статистика. На тарифі Auto —
виконання угод на біржі користувача через його API-ключі (non-custodial).

ВАЖЛИВО (позиціювання і юридично): продукт НЕ "предсказує ринок" і НЕ дає
фінансових порад. Він показує сетапи, score та ІСТОРИЧНУ статистику
(win rate за N угод). Формат сигналу:
  "ETH/USDT — лонг-сетап | Score 87/100 | історично score 85+ на ETH:
   62% win rate (247 угод бектесту)"
Заборонено формулювання виду "виросте з шансом 89%".

## 2. Структура репозиторію
smartflow/
├── CLAUDE.md
├── docs/ (SPEC.md, PROMPTS.md, DECISIONS.md)
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint
│   │   ├── config.py          # pydantic-settings, читає .env
│   │   ├── db/ (models.py, session.py)
│   │   ├── collectors/
│   │   │   ├── market_ws.py   # WebSocket OHLCV з біржі
│   │   │   ├── derivatives.py # funding, OI, liquidations (REST, кожні 5 хв)
│   │   │   └── news.py        # RSS + Fear&Greed + CoinGecko + макрокалендар
│   │   ├── analysis/
│   │   │   ├── smc.py         # обгортка над smartmoneyconcepts
│   │   │   ├── indicators.py  # pandas-ta
│   │   │   ├── sentiment.py   # Gemini: сентимент новин
│   │   │   ├── scoring.py     # конфлюенс → score 0-100
│   │   │   └── engine.py      # оркестратор пайплайна
│   │   ├── trading/
│   │   │   ├── risk.py        # калькулятор позиції, ліміти (REVIEW людиною)
│   │   │   ├── executor.py    # ccxt: ордери, біржові SL/TP (REVIEW людиною)
│   │   │   └── monitor.py     # супровід позицій, беззбиток, пуші
│   │   ├── bot/ (bot.py, handlers/, alerts.py)
│   │   ├── api/ (auth.py, signals.py, chart.py, keys.py, trade.py, billing.py)
│   │   └── security/ (crypto.py, ratelimit.py)
│   ├── tests/
│   ├── requirements.txt
│   └── deploy/ (systemd units: api, collector, bot)
├── frontend/                  # Mini App → Cloudflare Workers
│   ├── index.html
│   ├── src/ (app.js, chart.js, api.js, screens/)
│   └── wrangler.toml
└── scripts/ (backfill.py, backtest.py)

## 3. Схема БД
users        (id PK, tg_id UNIQUE, username, plan ENUM(free,pro,auto),
              plan_until TIMESTAMP NULL, risk_pct FLOAT DEFAULT 1.0,
              max_positions INT DEFAULT 3, autotrade_paused_until NULL,
              disclaimer_accepted_at, created_at)

api_keys     (id PK, user_id FK, exchange ENUM(binance,bybit),
              key_encrypted BLOB, secret_encrypted BLOB, nonce BLOB,
              label_mask TEXT,          -- "****a3f9" для UI
              perms_checked_at, created_at)

candles      (symbol, timeframe, ts, o, h, l, c, v,
              PRIMARY KEY(symbol, timeframe, ts))

signals      (id PK, symbol, side ENUM(long,short), timeframe,
              score INT, entry_low, entry_high, sl, tp1, tp2, rr FLOAT,
              factors JSON,    -- {"sweep":true,"ob_retest":true,"funding":-0.018,...}
              zones JSON,      -- [{"type":"OB","price_from":..,"price_to":..,
                               --   "time_from":..,"time_to":..}, ...]
              ai_explanation TEXT NULL, news_context JSON,
              status ENUM(active,expired,tp,sl),
              created_at, resolved_at NULL)

positions    (id PK, user_id FK, signal_id FK, exchange,
              client_order_id UUID UNIQUE, qty, entry_price, sl, tp1, tp2,
              status ENUM(pending,open,closed,cancelled),
              pnl_usd NULL, pnl_pct NULL, opened_at, closed_at NULL)

audit_log    (id PK, user_id, action TEXT, payload JSON,
              exchange_response JSON, ts)   -- append-only

payments     (id PK, user_id FK, provider ENUM(stars,crypto),
              amount, currency, plan, status, external_id,
              is_recurring BOOL, created_at)

news_items   (id PK, source, title, url, symbols JSON,
              sentiment INT NULL, importance INT NULL, published_at)

## 4. Джерела даних
- OHLCV: Binance або Bybit через ccxt; WebSocket для real-time +
  REST backfill. 15-20 пар, таймфрейми 15m / 1h / 4h.
- Деривативи (Binance/Bybit Futures REST, кожні 5 хв): funding rate,
  open interest, long/short ratio; кластери ліквідацій.
- Новини (без API-ключів): RSS через feedparser — CoinDesk, Cointelegraph,
  The Block, Decrypt, Bitcoin Magazine; Fear & Greed (api.alternative.me/fng/);
  CoinGecko free tier — trending coins, BTC dominance; економкалендар
  (FOMC/CPI). Сентимент усіх джерел — через Gemini API.
- Gemini API: (а) сентимент новини -10..+10 + важливість 1..5, JSON-only
  відповідь; (б) пояснення сигналу — ТІЛЬКИ на основі переданого JSON
  факторів, без вигадок, 4-6 речень, структура: що сталося → чому сетап →
  що інвалідовує.

## 5. Пайплайн аналізу (тригер: закриття свічки 15m/1h/4h)
1. Оновити candles
2. smc.py: структура (BOS/CHoCH), order blocks, FVG, liquidity sweeps,
   equal highs/lows, premium/discount. Контекст 4h, вхід 15m/1h.
3. Деривативи: funding, ΔOI, найближчі кластери ліквідацій
4. Новини за 4 год по символу + прапор макроподії (±30 хв FOMC/CPI)
5. scoring.py → score 0-100
6. score >= 70 і немає макропрапора → створити signal
7. Gemini-пояснення (кешується в signals.ai_explanation, 1 раз на сигнал)
8. alerts.py: PNG графіка (mplfinance/plotly) + текст → юзерам за тарифом

## 6. Скоринг (стартові ваги, калібруються бектестом)
liquidity sweep +25 | retest order block +20 | FVG у зоні входу +10 |
збіг напряму 4h і 1h структури +15 | екстремальний funding проти
напряму товпи +10 | ΔOI підтверджує +5 | сентимент новин узгоджений +10 |
discount для лонга / premium для шорта +5.
R:R < 2.0 → сигнал відкидається незалежно від score.

## 7. Ризик-движок (правила незмінні юзером)
- розмір позиції = (депозит * risk_pct) / |entry - sl|
- risk_pct: 0.5..3.0 (%), max 3 одночасні позиції
- денний збиток >= 5% депозиту → autotrade_paused_until = завтра 00:00 UTC
- заборона входу ±30 хв від макроподій
- сигнал старший 2 год АБО ціна пішла > 0.5% від зони входу → expired

## 8. Виконання угод (executor.py)
1. Лімітний ордер на вхід, clientOrderId = UUID (ідемпотентність)
2. Після філу — НЕГАЙНО біржові SL (stop-market) і TP; на Bybit —
   нативний position TP/SL. ЗАБОРОНЕНО програмні стопи.
3. monitor.py: TP1 (50% обсягу) → перенести стоп у беззбиток; пуш у бот
   про кожну подію (вхід/TP1/TP2/SL); усе в audit_log.
4. /panic: скасувати всі ордери юзера, закрити позиції маркетом
   (з підтвердженням), вимкнути автотрейдинг.

## 9. API (FastAPI; усе під JWT після валідації initData)
POST /auth/telegram          initData → перевірка HMAC + auth_date<1h → JWT (15 хв)
GET  /signals?status=active&symbol=
GET  /signals/{id}           деталі + zones + ai_explanation
GET  /chart?symbol&tf&from   свічки + зони активних сигналів (JSON)
POST /keys                   {exchange, key, secret}: перевірка прав
                             (відмова якщо є withdrawal) → шифрування → БД
DELETE /keys/{id}
POST /trade/confirm          {signal_id, risk_pct} → ордер
POST /trade/panic
GET  /stats                  win rate, profit factor, drawdown, історія
                             (публічна версія без auth — для маркетингу)
POST /billing/stars/webhook  (фактично через бот: successful_payment)

## 10. Mini App (frontend)
- Lightweight Charts: свічки 4h і 15m (перемикач), real-time через
  WebSocket/поллінг до бекенда
- Поверх графіка: прямокутники Order Block і FVG, зони ліквідності,
  лінії entry/SL/TP1/TP2 з цінами і R:R, мітки BOS/CHoCH/sweep
- Екрани: стрічка сигналів → деталі сигналу (графік + AI-пояснення +
  кнопка угоди для Auto) → статистика → налаштування (монети, TF,
  мін. score, risk_pct, ключі бірж) → тарифи
- Екран підтвердження угоди показує: розмір позиції (рахує бекенд),
  ризик у $ і %, потенційний прибуток/збиток, чекбокс "розумію ризики"
- Deep link із сигналу в боті: t.me/<bot>/app?startapp=signal_<id>

## 11. Білінг
- Telegram Stars: createInvoiceLink(currency="XTR",
  subscription_period=2592000) — нативна щомісячна підписка;
  successful_payment (включно з is_recurring) → plan_until =
  subscription_expiration_date
- Крипта (NOWPayments): одноразово 30 днів зі знижкою ~20%,
  нагадування за 3 дні до кінця
- APScheduler щогодини: plan_until < now() - 24h(grace) → plan='free',
  повідомлення з кнопкою продовження
- Тарифи: Free (1-2 сигнали/день із затримкою 15 хв, BTC+ETH) |
  Pro ~$15 (все миттєво, 20+ монет, AI) | Auto ~$45 (+автотрейдинг,
  калькулятор ризику)
- /start → обовʼязковий дисклеймер ("аналітичний інструмент, не фінансова
  порада...") → disclaimer_accepted_at

## 12. Безпека (чекліст-інваріанти)
- AES-256-GCM для ключів; майстер-ключ у /etc/smartflow/secrets.env (600)
- Перевірка apiRestrictions ключа: withdrawal → відмова
- Ключі ніколи: у логах, у відповідях API (тільки label_mask)
- initData HMAC на кожен запит; JWT 15 хв
- slowapi rate limiting на всі ендпоінти
- UFW: тільки 443 + SSH(ключі, без паролів), fail2ban
- Cloudflare proxy перед origin; origin приймає тільки IP Cloudflare
- systemd: окремий юзер без sudo, ProtectSystem=strict, PrivateTmp=true
- Щоденний gpg-шифрований бекап БД в інше сховище
- Глобальний kill switch адміна (вимкнути всі автотрейди)

## 13. Бектест (scripts/backtest.py)
- vectorbt на 2-3 роках історії; ТІЛЬКИ закриті свічки (без lookahead)
- walk-forward: оптимізація ваг на періоді A, перевірка на B;
  out-of-sample період не чіпається до фіналу
- Метрики: win rate, profit factor, max drawdown, по score-бакетах
  (70-79 / 80-89 / 90+) — ці числа і показуються юзерам

## 14. Етапи розробки
1. Каркас: структура, config, моделі БД, Alembic + CI (GitHub Actions:
   pytest, ruff, mypy, coverage badge)
2. Колектор OHLCV (WS + auto-reconnect + gap-fill) + scripts/backfill.py
3. SMC + індикатори, перевірка зон на тест-графіку
4. Деривативи + новини + Gemini-сентимент
5. Скоринг + signals + бот з алертами (PNG)
6. Бектест + walk-forward + калібрування ваг
7. Paper trading (4-8 тижнів) + публічна статистика
8. Mini App: графік із зонами + всі екрани + ДЕМО-режим (перегляд
   інтерфейсу з історичними сигналами без логіна) + деплой CF Workers
9. Білінг: Stars-підписка + крипта + крон тарифів
10. Автотрейдинг: keys → risk → executor → monitor
    (Bybit testnet → 5-10 бета-юзерів з мінімальними депозитами)
11. Презентація (портфоліо): README EN/UA з Mermaid-діаграмою архітектури,
    скріншоти/GIF, демо-відео 60-90 с, кейс-стаді (проблема → аналоги →
    архітектура → складні рішення → результати з живими метриками /stats)

## 15. Портфоліо-вимоги (наскрізні)
- Проєкт — флагман портфоліо автора. Якість коду, тестів, комітів і
  документації — на рівні продакшн open-source.
- Живі метрики /stats (аптайм, к-сть проаналізованих свічок, сигналів,
  win rate) — публічні, це головний доказ для роботодавців і клієнтів.
- "Глибока" фіча для технічних співбесід: walk-forward оптимізація +
  захист від lookahead bias — реалізувати і задокументувати особливо
  ретельно (окремий розділ у кейс-стаді).
- Публічність коду вирішується на Етапі 11: або публічне ядро без ваг
  скорингу і прод-конфігів, або приватний репо + публічний кейс-стаді.

# DECISIONS.md — журнал архітектурних рішень

Формат запису:
## YYYY-MM-DD — Назва рішення
Контекст → Рішення → Чому → Альтернативи, які відкинули

## 2026-06-13 — Відмова від pandas-ta; індикатори власною реалізацією
- **Контекст:** pandas-ta потрібна для Етапу 3 (ATR, EMA, об'ємні метрики).
  При установці venv на Python 3.12 виявилось: версія 0.3.14b0 (єдина сумісна
  з numpy<2.0) зникла з PyPI — залишились лише beta 0.4.67b0 і 0.4.71b0, які
  вимагають numpy>=2.2.6. Встановити з GitHub неможливо — репозиторій видалений.
- **Рішення:** pandas-ta повністю видалена з requirements.txt. На Етапі 3
  реалізуємо потрібні індикатори вручну на чистому pandas у
  `app/analysis/indicators.py`. Потрібних нам функцій мало і вони прості:
  EMA, ATR, об'ємні метрики — все в 20-30 рядків без зовнішніх залежностей.
- **numpy обмеження збережено:** `numpy>=1.26,<2.0` залишається, оскільки
  `smartmoneyconcepts` (вимагає numba>=0.58.1) потрібний для Етапу 3 і сумісний
  з numpy 1.x. Перевірено: smartmoneyconcepts 0.0.27 залежить від
  `pandas>=2.0.2, numpy>=1.24.3, numba>=0.58.1` — без pandas-ta.
- **Відкинуті альтернативи:** встановлення 0.4.x beta з `--no-deps` (ризик
  прихованих несумісностей); пошук з GitHub (репо видалено); заморозка Python
  на 3.11 (не вирішує PyPI-проблему).

## 2026-06-12 — Backend scaffold (Etap 1)
- `Base` та `engine` живуть у `db/session.py`; моделі — окремо в `db/models.py`.
  Альтернатива `db/base.py` відкинута — зайвий файл без реальної користі при такому ланцюгу імпортів.
- `StaticPool` автоматично вмикається коли `DATABASE_URL=sqlite+aiosqlite:///:memory:` —
  всі з'єднання тоді шарять одну in-memory базу (критично для тестів без patch-магії).
- `settings = get_settings()` на рівні модуля — зручний доступ поза FastAPI;
  `get_settings()` з `lru_cache` — для `Depends` і тест-оверрайду через `cache_clear()`.
- Enum-колонки з `native_enum=False` — переноситься між SQLite і PostgreSQL без міграції типів.
- `client_order_id` у `positions` — `sa.Uuid` (SQLAlchemy 2.0+): VARCHAR(36) на SQLite,
  native UUID на PostgreSQL, без додаткового коду.
- CI (`.github/workflows/ci.yml`): pytest + ruff + mypy, required secrets як env vars.

## 2026-06-11 — Стартові рішення
- Кросплатформність через Telegram Mini App (не нативні апи, не Flutter):
  один кодбейз, нуль сторів-модерації, досвід ContentFlow.
- Non-custodial: тільки API-ключі користувачів без права виводу.
- Стоп-лоси лише біржові (нативні TP/SL), ніколи програмні.
- Позиціювання: score + історичний win rate, без "ймовірностей" руху ціни.
- SQLite на старті, PostgreSQL при зростанні (через DATABASE_URL).

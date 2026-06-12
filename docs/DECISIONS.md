# DECISIONS.md — журнал архітектурних рішень

Формат запису:
## YYYY-MM-DD — Назва рішення
Контекст → Рішення → Чому → Альтернативи, які відкинули

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

"""News sentiment analysis via Google Gemini API (Etap 4.2).

Fetches NewsItem rows where sentiment IS NULL, scores them in batches via
Gemini generateContent, and writes sentiment (-10..+10) and importance (1..5)
back to the DB.

Disabled automatically when GEMINI_API_KEY is empty — the rest of the pipeline
continues without sentiment data.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import NewsItem
from app.db.session import AsyncSessionLocal

log = structlog.get_logger(__name__)

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent"
)
_MAX_RETRIES  = 5
_BACKOFF_BASE = 2.0
_HTTP_TIMEOUT = 30.0

# Matches ```json...``` or ```...``` blocks.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


# ── Pure helpers ───────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove ```json...``` wrappers; trim to outermost [ ... ] if present."""
    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def _parse_gemini_response(
    text: str,
    expected: int,
) -> list[dict[str, int]] | None:
    """Parse and validate a Gemini JSON response for *expected* items.

    Returns None (and logs a warning) on any structural or parse failure —
    callers keep sentiment=NULL for the affected batch.
    """
    try:
        data = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("sentiment_parse_error", error=str(exc), raw=text[:300])
        return None

    if not isinstance(data, list) or len(data) != expected:
        log.warning(
            "sentiment_wrong_length",
            expected=expected,
            got=len(data) if isinstance(data, list) else type(data).__name__,
        )
        return None

    validated: list[dict[str, int]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            log.warning("sentiment_item_not_dict", index=i)
            return None
        try:
            s   = int(item["sentiment"])
            imp = int(item["importance"])
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("sentiment_item_invalid_fields", index=i, error=str(exc))
            return None
        # Clamp to valid ranges — Gemini occasionally overshoots by 1.
        validated.append({
            "sentiment":  max(-10, min(10, s)),
            "importance": max(1,   min(5,  imp)),
        })

    return validated


def _build_prompt(titles: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    return (
        "You are a crypto market sentiment analyst.\n"
        f"Analyze the following {len(titles)} crypto news headlines.\n"
        "Return a JSON array with exactly one object per headline, in the same order.\n"
        'Each object must have exactly two integer keys:\n'
        '  "sentiment": -10 (very bearish) to +10 (very bullish), 0 = neutral\n'
        '  "importance": 1 (minor noise) to 5 (major market event)\n'
        "Return ONLY the JSON array. No explanations, no markdown, no extra text.\n\n"
        f"Headlines:\n{numbered}"
    )


# ── Gemini HTTP call with retry ────────────────────────────────────────────────

async def _call_gemini(
    client: httpx.AsyncClient,
    prompt: str,
    *,
    api_key: str,
    model: str,
) -> str | None:
    """POST to Gemini generateContent. Returns extracted text or None after all retries."""
    url     = _GEMINI_ENDPOINT.format(model=model)
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    for attempt in range(_MAX_RETRIES):
        try:
            resp = await client.post(
                url,
                json=payload,
                params={"key": api_key},
                timeout=_HTTP_TIMEOUT,
            )

            if resp.status_code == 429:
                delay = _BACKOFF_BASE ** attempt
                log.warning(
                    "sentiment_rate_limit",
                    attempt=attempt, next_delay_s=delay,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                continue

            if resp.status_code >= 500:
                delay = _BACKOFF_BASE ** attempt
                log.warning(
                    "sentiment_server_error",
                    status=resp.status_code, attempt=attempt, next_delay_s=delay,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                continue

            resp.raise_for_status()  # raises HTTPStatusError for remaining 4xx
            data = resp.json()
            return str(data["candidates"][0]["content"]["parts"][0]["text"])

        except httpx.HTTPStatusError as exc:
            # Non-retryable 4xx (e.g. 400 bad request, 401 invalid key).
            log.error("sentiment_http_error", status=exc.response.status_code, error=str(exc))
            return None
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            delay = _BACKOFF_BASE ** attempt
            log.warning(
                "sentiment_network_error",
                attempt=attempt, error=str(exc), next_delay_s=delay,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(delay)
        except (KeyError, IndexError, TypeError) as exc:
            log.error("sentiment_response_structure_error", error=str(exc))
            return None

    log.error("sentiment_call_failed", attempts=_MAX_RETRIES)
    return None


# ── Batch analysis ─────────────────────────────────────────────────────────────

async def analyze_batch(
    client: httpx.AsyncClient,
    items: list[NewsItem],
    *,
    api_key: str,
    model: str,
) -> list[tuple[int, int | None, int | None]]:
    """Analyze one batch. Returns [(news_item_id, sentiment, importance), ...].

    sentiment/importance are None when Gemini call or parsing fails — the
    caller leaves those DB rows with sentiment=NULL.
    """
    prompt = _build_prompt([it.title for it in items])
    raw    = await _call_gemini(client, prompt, api_key=api_key, model=model)

    if raw is None:
        return [(it.id, None, None) for it in items]

    parsed = _parse_gemini_response(raw, expected=len(items))
    if parsed is None:
        return [(it.id, None, None) for it in items]

    return [
        (it.id, entry["sentiment"], entry["importance"])
        for it, entry in zip(items, parsed, strict=True)
    ]


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def _fetch_unanalyzed(session: AsyncSession, limit: int) -> list[NewsItem]:
    """Return up to *limit* NewsItem rows where sentiment IS NULL."""
    result = await session.execute(
        select(NewsItem)
        .where(NewsItem.sentiment.is_(None))
        .order_by(NewsItem.published_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def _write_results(
    session: AsyncSession,
    results: list[tuple[int, int | None, int | None]],
) -> int:
    """Write sentiment/importance back to DB. Returns number of rows updated."""
    id_map = {r[0]: (r[1], r[2]) for r in results if r[1] is not None}
    if not id_map:
        return 0

    rows = (await session.execute(
        select(NewsItem).where(NewsItem.id.in_(list(id_map)))
    )).scalars().all()

    for row in rows:
        row.sentiment, row.importance = id_map[row.id]

    await session.commit()
    return len(rows)


# ── Scheduler job ──────────────────────────────────────────────────────────────

async def run_sentiment_analysis() -> None:
    """Fetch unanalyzed news, batch-score via Gemini, persist results."""
    if not settings.gemini_api_key:
        log.warning("sentiment_disabled_no_api_key")
        return

    batch_size  = settings.sentiment_batch_size
    fetch_limit = batch_size * 10  # up to 10 batches per run

    async with AsyncSessionLocal() as session:
        items = await _fetch_unanalyzed(session, limit=fetch_limit)

    if not items:
        log.debug("sentiment_no_pending_items")
        return

    log.info("sentiment_analysis_start", total=len(items), batch_size=batch_size)
    api_key = settings.gemini_api_key
    model   = settings.gemini_model
    all_results: list[tuple[int, int | None, int | None]] = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(items), batch_size):
            batch   = items[i : i + batch_size]
            results = await analyze_batch(client, batch, api_key=api_key, model=model)
            all_results.extend(results)

    async with AsyncSessionLocal() as session:
        updated = await _write_results(session, all_results)

    log.info("sentiment_analysis_done", analyzed=len(items), updated=updated)


def start_sentiment_scheduler(
    scheduler: AsyncIOScheduler | None = None,
) -> AsyncIOScheduler:
    """Register run_sentiment_analysis on *scheduler* and start if not running."""
    if scheduler is None:
        scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        run_sentiment_analysis,
        trigger="interval",
        minutes=settings.sentiment_analyze_interval_minutes,
        id="sentiment_analyze",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=120,
    )
    if not scheduler.running:
        scheduler.start()
    return scheduler

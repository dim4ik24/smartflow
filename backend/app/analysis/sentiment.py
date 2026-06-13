"""News sentiment analysis via Google Gemini API (Etap 4.2).

Uses the current google-genai SDK (``from google import genai``), NOT the
deprecated google-generativeai package.

Structured output (response_mime_type="application/json" + response_schema)
constrains Gemini to emit a valid JSON array — no markdown fences possible
at the protocol level.  A manual fallback parser (_strip_fences +
_parse_gemini_response) is kept anyway because structured output occasionally
still misbehaves in edge cases.

Disabled automatically when GEMINI_API_KEY is empty.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# aiohttp (used internally by google-genai SDK) on Windows tries to use
# aiodns/pycares for async DNS, but c-ares cannot read the Windows DNS config
# and raises "Could not contact DNS servers". Disabling aiodns before the
# import forces aiohttp to fall back to ThreadedResolver (getaddrinfo via
# thread pool), which works on all platforms. No effect on Linux production.
if sys.platform == "win32":
    sys.modules.setdefault("aiodns", None)  # type: ignore[arg-type]

from google import genai
from google.genai import types
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import NewsItem
from app.db.session import AsyncSessionLocal

log = structlog.get_logger(__name__)

_MAX_RETRIES  = 5
_BACKOFF_BASE = 2.0

# Fallback: strip ```json...``` / ```...``` wrappers the model might emit.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")


# ── Structured-output schema ───────────────────────────────────────────────────

class _SentimentScore(BaseModel):
    sentiment: int    # -10..+10
    importance: int   # 1..5


# ── Pure helpers (fallback parsing) ───────────────────────────────────────────

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
    """Parse and validate JSON text for *expected* items.

    Returns None on any failure — callers keep sentiment=NULL for the batch.
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
        "Each object must have exactly two integer keys:\n"
        '  "sentiment": -10 (very bearish) to +10 (very bullish), 0 = neutral\n'
        '  "importance": 1 (minor noise) to 5 (major market event)\n'
        "Return ONLY the JSON array. No explanations, no markdown, no extra text.\n\n"
        f"Headlines:\n{numbered}"
    )


# ── Gemini SDK call with retry ─────────────────────────────────────────────────

async def _call_gemini(
    client: genai.Client,
    prompt: str,
    *,
    model: str,
) -> str | None:
    """Call Gemini via async SDK. Returns response text or None after all retries.

    Retryable: 429 (rate limit), 5xx (server error), network/timeout exceptions.
    Non-retryable: 4xx (except 429) — wrong key, bad request, etc.
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[_SentimentScore],
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text

        except Exception as exc:
            # Extract HTTP status code if the SDK attaches one.
            code: int | None = getattr(exc, "code", None)

            # Non-retryable: definite API error that is NOT rate-limit or server fault.
            if code is not None and code != 429 and code < 500:
                log.error("sentiment_api_fatal", code=code, error=str(exc))
                return None

            delay = _BACKOFF_BASE ** attempt
            log.warning(
                "sentiment_retry",
                code=code, attempt=attempt, error=str(exc), next_delay_s=delay,
            )
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(delay)

    log.error("sentiment_call_failed", attempts=_MAX_RETRIES)
    return None


# ── Batch analysis ─────────────────────────────────────────────────────────────

async def analyze_batch(
    client: genai.Client,
    items: list[NewsItem],
    *,
    model: str,
) -> list[tuple[int, int | None, int | None]]:
    """Analyze one batch of NewsItem rows.

    Returns [(news_item_id, sentiment, importance), ...].
    sentiment/importance are None when the Gemini call or parsing fails.
    """
    prompt = _build_prompt([it.title for it in items])
    raw    = await _call_gemini(client, prompt, model=model)

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

    client = genai.Client(api_key=settings.gemini_api_key)
    model  = settings.gemini_model
    all_results: list[tuple[int, int | None, int | None]] = []

    for i in range(0, len(items), batch_size):
        batch   = items[i : i + batch_size]
        results = await analyze_batch(client, batch, model=model)
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

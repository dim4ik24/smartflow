"""Crypto news collector — RSS feeds + Fear & Greed Index.

Collects every ``settings.news_collect_interval_minutes`` minutes (APScheduler).
No API keys required for any source.
Sentiment analysis (Gemini) is handled separately in sentiment.py (Etap 4.2).
"""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import MarketSentiment, NewsItem
from app.db.session import AsyncSessionLocal

log = structlog.get_logger(__name__)

_HTTP_TIMEOUT  = 15.0   # seconds per request
_MAX_RETRIES   = 3
_BACKOFF_BASE  = 2.0    # delay = BACKOFF_BASE ** attempt  →  1 s, 2 s, 4 s
_FETCH_LIMIT   = 30     # max entries kept per RSS feed per cycle
_FG_SOURCE     = "fear_greed"


# ── Symbol extraction ──────────────────────────────────────────────────────────

def _compile_patterns(
    syns: dict[str, list[str]],
) -> dict[str, list[re.Pattern[str]]]:
    """Compile word-boundary regex patterns from a coin_synonyms dict."""
    return {
        base: [
            re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            for term in terms
        ]
        for base, terms in syns.items()
    }


# Module-level cache compiled from settings at import time.
_SETTINGS_PATTERNS: dict[str, list[re.Pattern[str]]] = _compile_patterns(
    settings.coin_synonyms
)


def extract_symbols(
    text: str,
    watched_symbols: list[str] | None = None,
    coin_synonyms: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return watched-symbol strings mentioned in *text*.

    Uses word-boundary regex so "ada" inside "Canada" does not match ADA/USDT.
    Returns ``[]`` for general market news (no coin detected).

    Parameters
    ----------
    text:
        Combined title + summary string to search.
    watched_symbols:
        Override list of symbols; defaults to ``settings.watched_symbols``.
    coin_synonyms:
        Override mapping (useful in tests); defaults to the module-level
        compiled ``settings.coin_synonyms`` patterns.
    """
    if not text:
        return []
    ws   = watched_symbols if watched_symbols is not None else settings.watched_symbols
    pats = _compile_patterns(coin_synonyms) if coin_synonyms is not None else _SETTINGS_PATTERNS

    result: list[str] = []
    for symbol in ws:
        base = symbol.split("/")[0]
        for pat in pats.get(base, []):
            if pat.search(text):
                result.append(symbol)
                break
    return result


# ── HTTP retry helper ──────────────────────────────────────────────────────────

async def _fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_retries: int = _MAX_RETRIES,
) -> httpx.Response | None:
    """GET *url* with exponential backoff. Returns ``None`` after all retries."""
    for attempt in range(max_retries):
        try:
            resp = await client.get(url, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            delay = _BACKOFF_BASE ** attempt
            log.warning(
                "news_fetch_retry",
                url=url, attempt=attempt, error=str(exc), next_delay_s=delay,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
    log.error("news_fetch_failed", url=url, attempts=max_retries)
    return None


# ── RSS ────────────────────────────────────────────────────────────────────────

def _entry_published(entry: Any) -> datetime:
    pt = entry.get("published_parsed") or entry.get("updated_parsed")
    if pt:
        return datetime(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], tzinfo=UTC)
    return datetime.now(UTC)


async def fetch_rss_feed(
    client: httpx.AsyncClient,
    feed_url: str,
) -> list[dict[str, Any]]:
    """Fetch and parse one RSS feed. Returns ``[]`` on any failure."""
    resp = await _fetch_with_retry(client, feed_url)
    if resp is None:
        return []

    # feedparser is synchronous CPU-bound work — run in thread pool.
    loop = asyncio.get_running_loop()
    feed: Any = await loop.run_in_executor(None, feedparser.parse, resp.text)

    items: list[dict[str, Any]] = []
    for entry in feed.entries[:_FETCH_LIMIT]:
        url = entry.get("link", "").strip()
        if not url:
            continue
        title   = entry.get("title", "").strip()
        summary = re.sub(
            r"<[^>]+>", " ",
            entry.get("summary", entry.get("description", "")),
        )
        items.append({
            "source":       feed_url,
            "title":        title,
            "url":          url,
            "symbols":      extract_symbols(f"{title} {summary}"),
            "published_at": _entry_published(entry),
        })

    log.info("news_rss_fetched", feed=feed_url, count=len(items))
    return items


# ── Fear & Greed ───────────────────────────────────────────────────────────────

async def fetch_fear_greed(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """Fetch the current Fear & Greed value. Returns 0 or 1 item dict."""
    resp = await _fetch_with_retry(client, settings.fear_greed_url)
    if resp is None:
        return []

    try:
        data  = resp.json()
        entry = data["data"][0]
        value = int(entry["value"])
        label = entry["value_classification"]
        ts    = datetime.fromtimestamp(int(entry["timestamp"]), tz=UTC)
        # URL is unique per hourly data point → serves as dedup key.
        url   = f"{settings.fear_greed_url}?ts={int(ts.timestamp())}"
        return [{
            "source":            _FG_SOURCE,
            "title":             f"Fear & Greed Index: {value} ({label})",
            "url":               url,
            "symbols":           [],
            "published_at":      ts,
            "fear_greed_value":  value,
            "classification":    label,
        }]
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        log.error("news_fear_greed_parse_error", error=str(exc))
        return []


# ── Database ───────────────────────────────────────────────────────────────────

async def upsert_news_items(
    session: AsyncSession,
    items: list[dict[str, Any]],
) -> int:
    """Insert new items, skip URLs already present in DB. Returns count inserted."""
    if not items:
        return 0

    urls = [it["url"] for it in items]
    result = await session.execute(
        select(NewsItem.url).where(NewsItem.url.in_(urls))
    )
    seen: set[str] = {row[0] for row in result}

    inserted = 0
    for it in items:
        if it["url"] in seen:
            continue
        session.add(NewsItem(
            source       = it["source"],
            title        = it["title"],
            url          = it["url"],
            symbols      = it["symbols"],
            sentiment    = None,
            importance   = None,
            published_at = it["published_at"],
        ))
        inserted += 1

    if inserted:
        await session.commit()
    return inserted


# ── Fear & Greed DB upsert ────────────────────────────────────────────────────

async def upsert_fear_greed(
    session: AsyncSession,
    items: list[dict[str, Any]],
) -> int:
    """Insert F&G snapshots not already present. Deduplicates by ts. Returns count inserted."""
    if not items:
        return 0

    timestamps = [it["published_at"] for it in items]
    result = await session.execute(
        select(MarketSentiment.ts).where(MarketSentiment.ts.in_(timestamps))
    )
    # SQLite returns naive datetimes; normalise to UTC-aware for comparison.
    seen: set[datetime] = {
        row[0] if row[0].tzinfo is not None else row[0].replace(tzinfo=UTC)
        for row in result
    }

    inserted = 0
    for it in items:
        if it["published_at"] in seen:
            continue
        session.add(MarketSentiment(
            ts=it["published_at"],
            fear_greed_value=it["fear_greed_value"],
            classification=it["classification"],
        ))
        inserted += 1

    if inserted:
        await session.commit()
    return inserted


async def get_latest_fear_greed(
    session: AsyncSession,
) -> MarketSentiment | None:
    """Return the most recent Fear & Greed snapshot, or None if table is empty."""
    result = await session.execute(
        select(MarketSentiment)
        .order_by(MarketSentiment.ts.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ── Collection job ─────────────────────────────────────────────────────────────

async def collect_news() -> None:
    """Fetch all sources and persist to DB. Registered as an APScheduler job."""
    log.info("news_collect_start")
    all_items: list[dict[str, Any]] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "SmartFlow-NewsBot/1.0"},
        follow_redirects=True,
    ) as client:
        # All RSS feeds concurrently — one feed failure does not stop others.
        rss_results = await asyncio.gather(
            *[fetch_rss_feed(client, url) for url in settings.news_rss_feeds],
            return_exceptions=True,
        )
        for res in rss_results:
            if isinstance(res, list):
                all_items.extend(res)
            else:
                log.error("news_rss_task_exception", error=str(res))

        fg_items = await fetch_fear_greed(client)

    async with AsyncSessionLocal() as session:
        news_inserted = await upsert_news_items(session, all_items)
        fg_inserted   = await upsert_fear_greed(session, fg_items)

    log.info(
        "news_collect_done",
        fetched=len(all_items),
        news_inserted=news_inserted,
        fg_inserted=fg_inserted,
    )


# ── Scheduler integration ──────────────────────────────────────────────────────

def start_news_scheduler(
    scheduler: AsyncIOScheduler | None = None,
) -> AsyncIOScheduler:
    """Register ``collect_news`` on *scheduler* and start it if not running."""
    if scheduler is None:
        scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        collect_news,
        trigger="interval",
        minutes=settings.news_collect_interval_minutes,
        id="news_collect",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,
    )
    if not scheduler.running:
        scheduler.start()
    return scheduler

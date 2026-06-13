"""Tests for app/collectors/news.py."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import sqlalchemy as sa

from app.collectors.news import (
    extract_symbols,
    fetch_fear_greed,
    fetch_rss_feed,
    get_latest_fear_greed,
    upsert_fear_greed,
    upsert_news_items,
)
from app.db.models import MarketSentiment, NewsItem

# ── Test fixtures / helpers ────────────────────────────────────────────────────

_SYMS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
_SYNS: dict[str, list[str]] = {
    "BTC": ["Bitcoin",   "BTC"],
    "ETH": ["Ethereum",  "ETH",  "Ether"],
    "SOL": ["Solana",    "SOL"],
    "ADA": ["Cardano",   "ADA"],
}

_RSS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Crypto News</title>
    <link>https://example.com</link>
    <item>
      <title>Bitcoin hits $100k as BTC bulls dominate market</title>
      <link>https://example.com/btc-100k</link>
      <description>Bitcoin price surged above $100,000 in weekend trading.</description>
      <pubDate>Fri, 01 Mar 2024 12:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Crypto market outlook: macro factors in focus</title>
      <link>https://example.com/market-outlook</link>
      <description>A general analysis of macroeconomic conditions.</description>
      <pubDate>Fri, 01 Mar 2024 11:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Ethereum upgrade improves Ether transfer speeds</title>
      <link>https://example.com/eth-upgrade</link>
      <description>The Ethereum network completed its scheduled upgrade.</description>
      <pubDate>Fri, 01 Mar 2024 10:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

_FEAR_GREED_JSON = (
    '{"name":"Fear and Greed Index",'
    '"data":[{"value":"42","value_classification":"Fear",'
    '"timestamp":"1704844800","time_until_update":"3600"}]}'
)


def _ok_response(text: str) -> MagicMock:
    """Generic 200 response; .json() only works when text is valid JSON."""
    resp = MagicMock()
    resp.status_code = 200
    resp.text = text
    resp.raise_for_status = MagicMock()
    try:
        parsed = __import__("json").loads(text)
        resp.json = MagicMock(return_value=parsed)
    except ValueError:
        resp.json = MagicMock(side_effect=ValueError("not JSON"))
    return resp


def _error_response(status: int = 503) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=MagicMock()
        )
    )
    return resp


def _client_returning(resp: MagicMock) -> AsyncMock:
    c = AsyncMock()
    c.get = AsyncMock(return_value=resp)
    return c


def _client_raising(exc: Exception) -> AsyncMock:
    c = AsyncMock()
    c.get = AsyncMock(side_effect=exc)
    return c


def _uid_url(prefix: str = "https://test.example.com/") -> str:
    """Unique URL per call — avoids cross-test DB collisions in shared in-memory DB."""
    return f"{prefix}{uuid.uuid4()}"


def _item(url: str | None = None, symbols: list[str] | None = None) -> dict:
    return {
        "source":       "test_feed",
        "title":        "Test news item",
        "url":          url or _uid_url(),
        "symbols":      symbols or [],
        "published_at": datetime(2024, 3, 1, 12, tzinfo=UTC),
    }


# ── extract_symbols — pure function ───────────────────────────────────────────

def test_extract_btc_from_title():
    result = extract_symbols("Bitcoin surges 10% — BTC eyes $100k", _SYMS, _SYNS)
    assert result == ["BTC/USDT"]


def test_extract_multiple_coins():
    result = extract_symbols("Bitcoin and Ethereum rally; Solana also gains", _SYMS, _SYNS)
    assert result == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_general_news_returns_empty():
    result = extract_symbols("Global markets rally on positive macro data", _SYMS, _SYNS)
    assert result == []


def test_word_boundary_no_false_positive_in_word():
    # "ada" inside "Canada" must NOT match ADA/USDT.
    syns = {"ADA": ["Cardano", "ADA"]}
    result = extract_symbols("Canada central bank cuts rates today", ["ADA/USDT"], syns)
    assert result == []


def test_case_insensitive_match():
    syns = {"BTC": ["Bitcoin", "BTC"]}
    result = extract_symbols("bitcoin is trading at its highest ever", ["BTC/USDT"], syns)
    assert result == ["BTC/USDT"]


def test_synonym_phrase_match():
    # "Ether" should match ETH/USDT.
    result = extract_symbols("Ether demand grows amid DeFi boom", _SYMS, _SYNS)
    assert result == ["ETH/USDT"]


def test_no_duplicate_symbol_when_multiple_terms_match():
    # Both "Bitcoin" and "BTC" appear — result must contain BTC/USDT only once.
    syns = {"BTC": ["Bitcoin", "BTC"]}
    result = extract_symbols("Bitcoin (BTC) breaks $90k barrier", ["BTC/USDT"], syns)
    assert result == ["BTC/USDT"]


def test_empty_text_returns_empty():
    assert extract_symbols("", _SYMS, _SYNS) == []


# ── fetch_rss_feed ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rss_parses_all_entries():
    client = _client_returning(_ok_response(_RSS_XML))
    items = await fetch_rss_feed(client, "https://example.com/rss")
    assert len(items) == 3


@pytest.mark.asyncio
async def test_rss_btc_entry_maps_to_btc_usdt():
    client = _client_returning(_ok_response(_RSS_XML))
    items = await fetch_rss_feed(client, "https://example.com/rss")
    btc = next(it for it in items if "btc-100k" in it["url"])
    assert "BTC/USDT" in btc["symbols"]
    assert isinstance(btc["published_at"], datetime)


@pytest.mark.asyncio
async def test_rss_general_entry_has_no_symbols():
    client = _client_returning(_ok_response(_RSS_XML))
    items = await fetch_rss_feed(client, "https://example.com/rss")
    general = next(it for it in items if "market-outlook" in it["url"])
    assert general["symbols"] == []


@pytest.mark.asyncio
async def test_rss_eth_entry_maps_via_synonym():
    client = _client_returning(_ok_response(_RSS_XML))
    items = await fetch_rss_feed(client, "https://example.com/rss")
    eth = next(it for it in items if "eth-upgrade" in it["url"])
    assert "ETH/USDT" in eth["symbols"]


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_rss_returns_empty_on_http_error(_mock_sleep: AsyncMock):
    client = _client_returning(_error_response(503))
    items = await fetch_rss_feed(client, "https://fail.example.com/rss")
    assert items == []


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_rss_returns_empty_on_timeout(_mock_sleep: AsyncMock):
    client = _client_raising(httpx.TimeoutException("timed out"))
    items = await fetch_rss_feed(client, "https://timeout.example.com/rss")
    assert items == []


# ── fetch_fear_greed ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fear_greed_parses_value_and_label():
    client = _client_returning(_ok_response(_FEAR_GREED_JSON))
    items = await fetch_fear_greed(client)
    assert len(items) == 1
    it = items[0]
    assert "42" in it["title"]
    assert "Fear" in it["title"]
    assert it["source"] == "fear_greed"
    assert it["symbols"] == []
    assert isinstance(it["published_at"], datetime)


@pytest.mark.asyncio
async def test_fear_greed_url_contains_timestamp():
    client = _client_returning(_ok_response(_FEAR_GREED_JSON))
    items = await fetch_fear_greed(client)
    assert "ts=1704844800" in items[0]["url"]


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_fear_greed_returns_empty_on_timeout(_mock_sleep: AsyncMock):
    client = _client_raising(httpx.TimeoutException("timed out"))
    items = await fetch_fear_greed(client)
    assert items == []


@pytest.mark.asyncio
async def test_fear_greed_returns_empty_on_malformed_json():
    client = _client_returning(_ok_response('{"unexpected": "structure"}'))
    items = await fetch_fear_greed(client)
    assert items == []


# ── upsert_news_items — DB tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_inserts_new_items(db_session):
    items = [_item(), _item()]
    n = await upsert_news_items(db_session, items)
    assert n == 2


@pytest.mark.asyncio
async def test_upsert_deduplicates_same_url(db_session):
    url = _uid_url()
    n1 = await upsert_news_items(db_session, [_item(url=url)])
    n2 = await upsert_news_items(db_session, [_item(url=url)])
    assert n1 == 1
    assert n2 == 0


@pytest.mark.asyncio
async def test_upsert_partial_dedup_inserts_only_new(db_session):
    url_a, url_b = _uid_url(), _uid_url()
    await upsert_news_items(db_session, [_item(url=url_a)])
    n = await upsert_news_items(db_session, [_item(url=url_a), _item(url=url_b)])
    assert n == 1  # url_b only


@pytest.mark.asyncio
async def test_upsert_empty_list_returns_zero(db_session):
    assert await upsert_news_items(db_session, []) == 0


@pytest.mark.asyncio
async def test_upsert_persists_symbols_json(db_session):
    from sqlalchemy import select as sa_select

    url = _uid_url()
    await upsert_news_items(db_session, [_item(url=url, symbols=["BTC/USDT", "ETH/USDT"])])
    row = (await db_session.execute(
        sa_select(NewsItem).where(NewsItem.url == url)
    )).scalar_one()
    assert row.symbols == ["BTC/USDT", "ETH/USDT"]


@pytest.mark.asyncio
async def test_upsert_sentiment_is_null(db_session):
    from sqlalchemy import select as sa_select

    url = _uid_url()
    await upsert_news_items(db_session, [_item(url=url)])
    row = (await db_session.execute(
        sa_select(NewsItem).where(NewsItem.url == url)
    )).scalar_one()
    assert row.sentiment is None
    assert row.importance is None


# ── upsert_fear_greed — MarketSentiment DB tests ──────────────────────────────

_FG_DEFAULT_TS = datetime(2024, 1, 10, 12, tzinfo=UTC)


def _fg_item(ts: datetime | None = None, value: int = 42, label: str = "Fear") -> dict:
    resolved_ts = ts or _FG_DEFAULT_TS
    return {
        "source":           "fear_greed",
        "title":            f"Fear & Greed Index: {value} ({label})",
        "url":              f"https://api.alternative.me/fng/?ts={int(resolved_ts.timestamp())}",
        "symbols":          [],
        "published_at":     resolved_ts,
        "fear_greed_value": value,
        "classification":   label,
    }


@pytest.mark.asyncio
async def test_fg_upsert_inserts_new_row(db_session):
    from sqlalchemy import select as sa_select

    ts = datetime(2024, 3, 1, 12, tzinfo=UTC)
    n = await upsert_fear_greed(db_session, [_fg_item(ts=ts, value=55, label="Greed")])
    assert n == 1
    row = (await db_session.execute(
        sa_select(MarketSentiment).where(MarketSentiment.ts == ts)
    )).scalar_one()
    assert row.fear_greed_value == 55
    assert row.classification == "Greed"


@pytest.mark.asyncio
async def test_fg_upsert_deduplicates_same_ts(db_session):
    ts = datetime(2024, 3, 2, 12, tzinfo=UTC)
    n1 = await upsert_fear_greed(db_session, [_fg_item(ts=ts)])
    n2 = await upsert_fear_greed(db_session, [_fg_item(ts=ts)])
    assert n1 == 1
    assert n2 == 0


@pytest.mark.asyncio
async def test_fg_upsert_empty_returns_zero(db_session):
    assert await upsert_fear_greed(db_session, []) == 0


@pytest.mark.asyncio
async def test_fg_not_written_to_news_items(db_session):
    from sqlalchemy import select as sa_select

    ts = datetime(2024, 3, 3, 12, tzinfo=UTC)
    await upsert_fear_greed(db_session, [_fg_item(ts=ts)])
    count = (await db_session.execute(
        sa_select(sa.func.count()).select_from(NewsItem).where(NewsItem.source == "fear_greed")
    )).scalar_one()
    assert count == 0


# ── get_latest_fear_greed ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_latest_fear_greed_returns_most_recent(db_session):
    # Use far-future timestamps to guarantee these rows are the newest in the shared DB.
    ts_old = datetime(2099, 1, 1, 10, tzinfo=UTC)
    ts_new = datetime(2099, 1, 1, 11, tzinfo=UTC)
    await upsert_fear_greed(db_session, [_fg_item(ts=ts_old, value=30, label="Fear")])
    await upsert_fear_greed(db_session, [_fg_item(ts=ts_new, value=75, label="Greed")])

    row = await get_latest_fear_greed(db_session)
    assert row is not None
    assert row.fear_greed_value == 75
    assert row.classification == "Greed"

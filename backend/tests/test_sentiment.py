"""Tests for app/analysis/sentiment.py.

Mocking strategy:
  - Pure helpers (_strip_fences, _parse_gemini_response): no mocking.
  - _call_gemini: mock client.aio.models.generate_content (AsyncMock).
    SDK exceptions are simulated with plain Exception subclasses that carry
    a .code attribute — _call_gemini checks getattr(exc, "code", None), so
    the real SDK exception type is not required for unit tests.
  - analyze_batch: mock _call_gemini via patch (simpler than mocking the client).
  - DB helpers: use the shared in-memory db_session fixture.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.analysis.sentiment import (
    _MAX_RETRIES,
    _call_gemini,
    _parse_gemini_response,
    _strip_fences,
    _write_results,
    analyze_batch,
    run_sentiment_analysis,
)
from app.db.models import NewsItem

# ── Helpers ────────────────────────────────────────────────────────────────────

def _news_item(*, id: int, title: str) -> MagicMock:
    item = MagicMock(spec=NewsItem)
    item.id    = id
    item.title = title
    return item


def _sdk_response(text: str) -> MagicMock:
    """Simulate a successful Gemini SDK response object."""
    r = MagicMock()
    r.text = text
    return r


def _sdk_client(*, side_effect=None, return_value=None) -> MagicMock:
    """Mock genai.Client whose aio.models.generate_content is an AsyncMock."""
    client = MagicMock()
    if side_effect is not None:
        client.aio.models.generate_content = AsyncMock(side_effect=side_effect)
    else:
        client.aio.models.generate_content = AsyncMock(return_value=return_value)
    return client


class _ApiError(Exception):
    """Minimal stand-in for google.genai.errors.APIError in unit tests."""
    def __init__(self, code: int) -> None:
        super().__init__(f"API error {code}")
        self.code = code


# ── _strip_fences — pure ───────────────────────────────────────────────────────

def test_strip_fences_removes_json_fence():
    assert _strip_fences('```json\n[{"a":1}]\n```') == '[{"a":1}]'


def test_strip_fences_removes_plain_fence():
    assert _strip_fences('```\n[{"a":1}]\n```') == '[{"a":1}]'


def test_strip_fences_passthrough_no_fence():
    assert _strip_fences('[{"a":1}]') == '[{"a":1}]'


def test_strip_fences_trims_preamble_and_suffix():
    assert _strip_fences('Sure, here: [{"a":1}] done.') == '[{"a":1}]'


# ── _parse_gemini_response — pure ──────────────────────────────────────────────

def test_parse_valid_response():
    raw = json.dumps([{"sentiment": 7, "importance": 3}])
    assert _parse_gemini_response(raw, expected=1) == [{"sentiment": 7, "importance": 3}]


def test_parse_two_items():
    raw = json.dumps([{"sentiment": -5, "importance": 4}, {"sentiment": 2, "importance": 1}])
    result = _parse_gemini_response(raw, expected=2)
    assert result is not None
    assert result[0]["sentiment"] == -5
    assert result[1]["sentiment"] == 2


def test_parse_clamps_sentiment_above_max():
    raw = json.dumps([{"sentiment": 15, "importance": 3}])
    result = _parse_gemini_response(raw, expected=1)
    assert result is not None
    assert result[0]["sentiment"] == 10


def test_parse_clamps_importance_below_min():
    raw = json.dumps([{"sentiment": 0, "importance": 0}])
    result = _parse_gemini_response(raw, expected=1)
    assert result is not None
    assert result[0]["importance"] == 1


def test_parse_wrong_length_returns_none():
    raw = json.dumps([{"sentiment": 5, "importance": 2}, {"sentiment": 1, "importance": 1}])
    assert _parse_gemini_response(raw, expected=1) is None


def test_parse_invalid_json_returns_none():
    assert _parse_gemini_response("not json", expected=1) is None


def test_parse_missing_importance_returns_none():
    raw = json.dumps([{"sentiment": 5}])
    assert _parse_gemini_response(raw, expected=1) is None


def test_parse_non_list_returns_none():
    raw = json.dumps({"sentiment": 5, "importance": 2})
    assert _parse_gemini_response(raw, expected=1) is None


def test_parse_strips_json_fence_before_parsing():
    raw = '```json\n[{"sentiment":3,"importance":2}]\n```'
    assert _parse_gemini_response(raw, expected=1) == [{"sentiment": 3, "importance": 2}]


# ── _call_gemini ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_gemini_success_returns_text():
    payload = '[{"sentiment":5,"importance":3}]'
    client  = _sdk_client(return_value=_sdk_response(payload))
    result  = await _call_gemini(client, "prompt", model="gemini-2.0-flash")
    assert result == payload


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_retries_on_429_then_succeeds(mock_sleep: AsyncMock):
    ok_text = '[{"sentiment":1,"importance":1}]'
    client  = _sdk_client(side_effect=[_ApiError(429), _sdk_response(ok_text)])
    result  = await _call_gemini(client, "prompt", model="gemini-2.0-flash")
    assert result == ok_text
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_retries_on_503(mock_sleep: AsyncMock):
    ok_text = '[{"sentiment":2,"importance":2}]'
    client  = _sdk_client(side_effect=[_ApiError(503), _sdk_response(ok_text)])
    result  = await _call_gemini(client, "prompt", model="gemini-2.0-flash")
    assert result == ok_text
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_retries_on_network_error(mock_sleep: AsyncMock):
    ok_text = '[{"sentiment":3,"importance":1}]'
    # Plain Exception (no .code) simulates a network/timeout error.
    client  = _sdk_client(side_effect=[Exception("connection reset"), _sdk_response(ok_text)])
    result  = await _call_gemini(client, "prompt", model="gemini-2.0-flash")
    assert result == ok_text
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_all_retries_exhausted_returns_none(mock_sleep: AsyncMock):
    client = _sdk_client(side_effect=Exception("timeout"))
    result = await _call_gemini(client, "prompt", model="gemini-2.0-flash")
    assert result is None
    assert mock_sleep.call_count == _MAX_RETRIES - 1


@pytest.mark.asyncio
async def test_call_gemini_non_retryable_4xx_returns_none_immediately():
    # 401 Unauthorized — no point retrying, return None after first attempt.
    client  = _sdk_client(side_effect=_ApiError(401))
    result  = await _call_gemini(client, "prompt", model="gemini-2.0-flash")
    assert result is None
    # Should NOT have been called more than once (no retry on 4xx).
    assert client.aio.models.generate_content.call_count == 1


# ── analyze_batch ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_batch_success():
    payload = json.dumps([{"sentiment": 7, "importance": 3}, {"sentiment": -5, "importance": 4}])
    client  = _sdk_client(return_value=_sdk_response(payload))
    items   = [_news_item(id=1, title="Bitcoin surges"), _news_item(id=2, title="Market crashes")]
    results = await analyze_batch(client, items, model="gemini-2.0-flash")
    assert results == [(1, 7, 3), (2, -5, 4)]


@pytest.mark.asyncio
async def test_analyze_batch_json_fence_stripped():
    """Structured output with a stray ```json``` wrapper is still handled."""
    inner  = json.dumps([{"sentiment": 5, "importance": 2}])
    fenced = f"```json\n{inner}\n```"
    client = _sdk_client(return_value=_sdk_response(fenced))
    items  = [_news_item(id=10, title="Ethereum upgrade")]
    results = await analyze_batch(client, items, model="gemini-2.0-flash")
    assert results == [(10, 5, 2)]


@pytest.mark.asyncio
async def test_analyze_batch_broken_response_preserves_null():
    """Unparseable response → sentinel (id, None, None), DB row stays NULL."""
    client  = _sdk_client(return_value=_sdk_response("Gemini went completely off script"))
    items   = [_news_item(id=3, title="Some news")]
    results = await analyze_batch(client, items, model="gemini-2.0-flash")
    assert results == [(3, None, None)]


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_analyze_batch_call_failure_preserves_null(mock_sleep: AsyncMock):
    """All retries exhausted → sentinel NULLs, no exception propagated."""
    client  = _sdk_client(side_effect=Exception("network failure"))
    items   = [_news_item(id=4, title="Other news")]
    results = await analyze_batch(client, items, model="gemini-2.0-flash")
    assert results == [(4, None, None)]


@pytest.mark.asyncio
async def test_analyze_batch_wrong_count_preserves_null():
    """Gemini returns 2 items for a 1-item batch → NULL."""
    payload = json.dumps([{"sentiment": 1, "importance": 1}, {"sentiment": 2, "importance": 2}])
    client  = _sdk_client(return_value=_sdk_response(payload))
    items   = [_news_item(id=5, title="Solo headline")]
    results = await analyze_batch(client, items, model="gemini-2.0-flash")
    assert results == [(5, None, None)]


# ── _write_results — DB ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_results_updates_sentiment_and_importance(db_session):
    from sqlalchemy import select as sa_select

    from app.collectors.news import upsert_news_items

    url = f"https://sentiment-test.example.com/{uuid.uuid4()}"
    await upsert_news_items(db_session, [{
        "source": "test", "title": "Bitcoin breaks $100k",
        "url": url, "symbols": ["BTC/USDT"],
        "published_at": datetime(2024, 5, 1, 12, tzinfo=UTC),
    }])
    item = (await db_session.execute(
        sa_select(NewsItem).where(NewsItem.url == url)
    )).scalar_one()
    assert item.sentiment is None

    updated = await _write_results(db_session, [(item.id, 8, 4)])
    assert updated == 1

    await db_session.refresh(item)
    assert item.sentiment == 8
    assert item.importance == 4


@pytest.mark.asyncio
async def test_write_results_skips_null_entries(db_session):
    updated = await _write_results(db_session, [(99999, None, None)])
    assert updated == 0


@pytest.mark.asyncio
async def test_write_results_empty_list_returns_zero(db_session):
    assert await _write_results(db_session, []) == 0


# ── run_sentiment_analysis ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_sentiment_analysis_skips_when_no_api_key():
    from app.config import settings
    with patch.object(settings, "gemini_api_key", ""):
        # Must complete silently — no SDK call, no DB interaction.
        await run_sentiment_analysis()

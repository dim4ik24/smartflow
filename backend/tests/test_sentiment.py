"""Tests for app/analysis/sentiment.py."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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


def _gemini_ok(text: str) -> MagicMock:
    """200 response with Gemini JSON structure."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={
        "candidates": [{"content": {"parts": [{"text": text}]}}]
    })
    return resp


def _rate_limit_resp() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 429
    resp.raise_for_status = MagicMock()
    return resp


def _server_error_resp() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 503
    resp.raise_for_status = MagicMock()
    return resp


def _post_client(*responses: MagicMock) -> AsyncMock:
    """AsyncMock client whose .post returns *responses in sequence."""
    client = AsyncMock()
    client.post = AsyncMock(side_effect=list(responses))
    return client


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
    result = _parse_gemini_response(raw, expected=1)
    assert result == [{"sentiment": 7, "importance": 3}]


def test_parse_two_items():
    raw = json.dumps([{"sentiment": -5, "importance": 4}, {"sentiment": 2, "importance": 1}])
    result = _parse_gemini_response(raw, expected=2)
    assert result is not None
    assert len(result) == 2
    assert result[0]["sentiment"] == -5


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


def test_parse_missing_importance_field_returns_none():
    raw = json.dumps([{"sentiment": 5}])
    assert _parse_gemini_response(raw, expected=1) is None


def test_parse_non_list_returns_none():
    raw = json.dumps({"sentiment": 5, "importance": 2})
    assert _parse_gemini_response(raw, expected=1) is None


def test_parse_strips_json_fence_before_parsing():
    raw = '```json\n[{"sentiment":3,"importance":2}]\n```'
    result = _parse_gemini_response(raw, expected=1)
    assert result == [{"sentiment": 3, "importance": 2}]


# ── _call_gemini ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_gemini_success_returns_text():
    payload = '[{"sentiment":5,"importance":3}]'
    client  = _post_client(_gemini_ok(payload))
    result  = await _call_gemini(client, "prompt", api_key="k", model="gemini-1.5-flash")
    assert result == payload


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_retries_on_429_then_succeeds(mock_sleep: AsyncMock):
    ok_text = '[{"sentiment":1,"importance":1}]'
    client  = _post_client(_rate_limit_resp(), _gemini_ok(ok_text))
    result  = await _call_gemini(client, "prompt", api_key="k", model="gemini-1.5-flash")
    assert result == ok_text
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_retries_on_500(mock_sleep: AsyncMock):
    ok_text = '[{"sentiment":2,"importance":2}]'
    client  = _post_client(_server_error_resp(), _gemini_ok(ok_text))
    result  = await _call_gemini(client, "prompt", api_key="k", model="gemini-1.5-flash")
    assert result == ok_text
    mock_sleep.assert_called_once()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_call_gemini_all_retries_exhausted_returns_none(mock_sleep: AsyncMock):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    result = await _call_gemini(client, "prompt", api_key="k", model="gemini-1.5-flash")
    assert result is None
    assert mock_sleep.call_count == _MAX_RETRIES - 1


@pytest.mark.asyncio
async def test_call_gemini_non_retryable_4xx_returns_none():
    resp = MagicMock()
    resp.status_code = 401
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(status_code=401)
        )
    )
    client = _post_client(resp)
    result = await _call_gemini(client, "prompt", api_key="bad-key", model="gemini-1.5-flash")
    assert result is None


# ── analyze_batch ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_batch_success():
    payload = json.dumps([{"sentiment": 7, "importance": 3}, {"sentiment": -5, "importance": 4}])
    client  = _post_client(_gemini_ok(payload))
    items   = [_news_item(id=1, title="Bitcoin surges"), _news_item(id=2, title="Market crashes")]
    results = await analyze_batch(client, items, api_key="key", model="gemini-1.5-flash")
    assert results == [(1, 7, 3), (2, -5, 4)]


@pytest.mark.asyncio
async def test_analyze_batch_json_fence_stripped():
    inner  = json.dumps([{"sentiment": 5, "importance": 2}])
    fenced = f"```json\n{inner}\n```"
    client = _post_client(_gemini_ok(fenced))
    items  = [_news_item(id=10, title="Ethereum upgrade")]
    results = await analyze_batch(client, items, api_key="key", model="gemini-1.5-flash")
    assert results == [(10, 5, 2)]


@pytest.mark.asyncio
async def test_analyze_batch_broken_response_preserves_null():
    client  = _post_client(_gemini_ok("This is not JSON at all — Gemini went off script"))
    items   = [_news_item(id=3, title="Some news")]
    results = await analyze_batch(client, items, api_key="key", model="gemini-1.5-flash")
    assert results == [(3, None, None)]


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_analyze_batch_call_failure_preserves_null(mock_sleep: AsyncMock):
    client = AsyncMock()
    client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    items   = [_news_item(id=4, title="Other news")]
    results = await analyze_batch(client, items, api_key="key", model="gemini-1.5-flash")
    assert results == [(4, None, None)]


@pytest.mark.asyncio
async def test_analyze_batch_wrong_count_preserves_null():
    # Gemini returns 2 items for a 1-item batch.
    payload = json.dumps([{"sentiment": 1, "importance": 1}, {"sentiment": 2, "importance": 2}])
    client  = _post_client(_gemini_ok(payload))
    items   = [_news_item(id=5, title="Solo headline")]
    results = await analyze_batch(client, items, api_key="key", model="gemini-1.5-flash")
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
async def test_write_results_skips_null_results(db_session):
    from app.collectors.news import upsert_news_items

    url = f"https://sentiment-test.example.com/{uuid.uuid4()}"
    await upsert_news_items(db_session, [{
        "source": "test", "title": "Unanalyzed news",
        "url": url, "symbols": [],
        "published_at": datetime(2024, 5, 1, 12, tzinfo=UTC),
    }])
    # Result with None sentiment — should not update anything.
    updated = await _write_results(db_session, [(9999, None, None)])
    assert updated == 0


@pytest.mark.asyncio
async def test_write_results_empty_list_returns_zero(db_session):
    assert await _write_results(db_session, []) == 0


# ── run_sentiment_analysis — integration ──────────────────────────────────────

@pytest.mark.asyncio
async def test_run_sentiment_analysis_skips_when_no_api_key():
    from app.config import settings
    with patch.object(settings, "gemini_api_key", ""):
        # Must return silently without calling Gemini or touching DB.
        await run_sentiment_analysis()

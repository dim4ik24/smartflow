"""Tests for FastAPI app setup and health endpoint."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_unknown_route_returns_404(client: AsyncClient) -> None:
    response = await client.get("/nonexistent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_docs_not_exposed_in_non_debug_mode(client: AsyncClient) -> None:
    # settings.debug is False in test env (DEBUG=false in conftest.py)
    response = await client.get("/docs")
    assert response.status_code == 404

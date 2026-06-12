"""Standalone entrypoint for the market WebSocket OHLCV collector.

Invoke as:  python -m app.collectors.run_collector
systemd:    ExecStart=/opt/smartflow/venv/bin/python -m app.collectors.run_collector
"""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog

from app.config import settings


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
            if settings.debug
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
        format="%(message)s",
    )


def main() -> None:
    _configure_logging()
    # Import after logging config so module-level structlog.get_logger() calls
    # in market_ws pick up the configured processors.
    from app.collectors.market_ws import run_collector

    asyncio.run(run_collector())


if __name__ == "__main__":
    main()

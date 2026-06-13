"""Live test: run_sentiment_analysis() on news already in DB."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func

from app.analysis.sentiment import run_sentiment_analysis
from app.db.models import NewsItem
from app.db.session import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as s:
        null_count = (await s.execute(
            select(func.count()).select_from(NewsItem).where(NewsItem.sentiment.is_(None))
        )).scalar_one()

    print(f"Items with sentiment=NULL: {null_count}")
    if null_count == 0:
        print("Nothing to analyze.")
        return

    print("Running run_sentiment_analysis()...\n")
    try:
        await run_sentiment_analysis()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        return

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(NewsItem)
            .where(NewsItem.sentiment.is_not(None))
            .order_by(NewsItem.published_at.desc())
            .limit(10)
        )).scalars().all()

    if not rows:
        async with AsyncSessionLocal() as s:
            still_null = (await s.execute(
                select(func.count()).select_from(NewsItem).where(NewsItem.sentiment.is_(None))
            )).scalar_one()
        print(f"No items scored. Still NULL: {still_null}")
        return

    print(f"\n{'Headline':<72} {'Sent':>5} {'Imp':>4}")
    print("-" * 84)
    for r in rows:
        title = r.title[:70] + ".." if len(r.title) > 70 else r.title
        symbols = f"[{','.join(r.symbols)}] " if r.symbols else ""
        label = f"{symbols}{title}"[:72]
        print(f"{label:<72} {r.sentiment:>+5}  {r.importance:>3}")


if __name__ == "__main__":
    asyncio.run(main())

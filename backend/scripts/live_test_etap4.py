"""Live test for Etap 4: collect_news + run_sentiment_analysis."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func

from app.collectors.news import collect_news
from app.analysis.sentiment import run_sentiment_analysis
from app.db.models import NewsItem
from app.db.session import AsyncSessionLocal, Base, engine


async def main() -> None:
    # Ensure tables exist (create if first run).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # ── Step 1: collect news ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: collect_news()")
    print("=" * 60)

    count_before = 0
    async with AsyncSessionLocal() as s:
        count_before = (await s.execute(
            select(func.count()).select_from(NewsItem)
        )).scalar_one()

    await collect_news()

    async with AsyncSessionLocal() as s:
        count_after = (await s.execute(
            select(func.count()).select_from(NewsItem)
        )).scalar_one()

    inserted = count_after - count_before
    print(f"\nTotal in DB: {count_after}  |  Inserted this run: {inserted}")

    # 5 headlines that have at least one symbol
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(NewsItem)
            .where(NewsItem.symbols != "[]")
            .order_by(NewsItem.published_at.desc())
            .limit(5)
        )).scalars().all()

    print("\n--- 5 latest headlines WITH symbols ---")
    for r in rows:
        symbols_str = ", ".join(r.symbols) if r.symbols else "—"
        print(f"  [{symbols_str}]  {r.title[:90]}")

    # ── Step 2: sentiment analysis ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: run_sentiment_analysis()")
    print("=" * 60)

    # Record which IDs are NULL before analysis.
    async with AsyncSessionLocal() as s:
        null_ids = [r[0] for r in (await s.execute(
            select(NewsItem.id).where(NewsItem.sentiment.is_(None)).limit(100)
        )).all()]

    print(f"\nNews items with sentiment=NULL before analysis: {len(null_ids)}")

    try:
        await run_sentiment_analysis()
    except Exception as exc:
        print(f"\n[ERROR] run_sentiment_analysis raised:\n{exc!r}")
        return

    # Show results for items that were NULL before (now hopefully scored).
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(NewsItem)
            .where(NewsItem.id.in_(null_ids[:30]))
            .where(NewsItem.sentiment.is_not(None))
            .order_by(NewsItem.published_at.desc())
            .limit(10)
        )).scalars().all()

    if not rows:
        print("\nNo items were scored (all may still be NULL — check errors above).")
        # Show raw NULL count to help debug.
        async with AsyncSessionLocal() as s:
            still_null = (await s.execute(
                select(func.count()).select_from(NewsItem).where(NewsItem.sentiment.is_(None))
            )).scalar_one()
        print(f"Items still with sentiment=NULL: {still_null}")
        return

    print(f"\n{'Headline':<70} {'Sent':>5} {'Imp':>4}")
    print("-" * 82)
    for r in rows:
        title = r.title[:68] + ".." if len(r.title) > 68 else r.title
        print(f"{title:<70} {r.sentiment:>+5}  {r.importance:>3}")


if __name__ == "__main__":
    asyncio.run(main())

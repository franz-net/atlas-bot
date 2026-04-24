"""
Backfill forward-return outcomes for shadow-logged near-miss rows.

For each unbackfilled row older than 1 day, fetches daily bars for the
symbol and computes the 1-day and 3-day forward return relative to the
price captured at rejection time. Groups rows by symbol so we only
hit the API once per symbol per run.

Run from scheduler after EOD, or manually:
    python -m src.jobs.backfill_near_miss_outcomes
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.api.tradestation import TradeStationClient
from src.ledger.ledger import TradingLedger
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'near_miss_backfill')


def _compute_forward_return(
    entry_price: float,
    rejection_date: datetime,
    bars: pd.DataFrame,
    trading_days_ahead: int,
) -> Optional[float]:
    """
    Return the forward return (decimal fraction) from entry_price to the
    close N trading days after rejection_date, or None if bars are missing.
    """
    if bars is None or bars.empty or entry_price is None or entry_price <= 0:
        return None

    # DataFrame index is expected to be timestamps; normalize to dates.
    try:
        bar_dates = pd.to_datetime(bars.index).date
    except Exception:
        return None

    reject_date = rejection_date.date()
    future = [(i, d) for i, d in enumerate(bar_dates) if d > reject_date]
    if len(future) < trading_days_ahead:
        return None

    target_idx = future[trading_days_ahead - 1][0]
    close = bars.iloc[target_idx].get('Close')
    if close is None or pd.isna(close):
        return None

    return float((close - entry_price) / entry_price)


async def backfill_once(
    client: TradeStationClient,
    ledger: TradingLedger,
    max_rows: int = 500,
) -> Dict:
    """
    Run one pass of the backfill.

    Args:
        client: Authenticated TradeStation client
        ledger: TradingLedger
        max_rows: Cap rows processed per run

    Returns:
        Dict with counts: pending, updated, skipped
    """
    pending = ledger.get_near_misses_pending_backfill(max_rows=max_rows)
    if not pending:
        logger.info("No near-misses pending backfill")
        return {'pending': 0, 'updated': 0, 'skipped': 0}

    # Group rows by symbol to minimize API calls
    by_symbol: Dict[str, List[Dict]] = defaultdict(list)
    for row in pending:
        by_symbol[row['symbol']].append(row)

    logger.info(
        f"Backfilling {len(pending)} near-misses across {len(by_symbol)} symbols"
    )

    updated = 0
    skipped = 0

    for symbol, rows in by_symbol.items():
        try:
            bars = await client.get_historical_bars(
                symbol, interval='daily', bars_back=10
            )
        except Exception as e:
            logger.warning(f"Backfill: failed to fetch bars for {symbol}: {e}")
            skipped += len(rows)
            continue

        for row in rows:
            try:
                rej_ts = datetime.fromisoformat(row['timestamp'])
            except (ValueError, TypeError):
                skipped += 1
                continue

            entry = row.get('entry_price')
            outcome_1d = _compute_forward_return(entry, rej_ts, bars, 1)
            outcome_3d = _compute_forward_return(entry, rej_ts, bars, 3)

            # Only mark as backfilled once the 1d return is available.
            # 3d may still be None if not enough trading days have passed;
            # that's fine — we'll write what we have and move on.
            if outcome_1d is None:
                skipped += 1
                continue

            if ledger.update_near_miss_outcome(
                row['id'], outcome_1d, outcome_3d
            ):
                updated += 1
            else:
                skipped += 1

    result = {'pending': len(pending), 'updated': updated, 'skipped': skipped}
    logger.info(
        f"Backfill complete: {updated} updated, {skipped} skipped "
        f"(of {len(pending)} pending)"
    )
    return result


async def _main() -> None:
    """CLI entry point."""
    ledger = TradingLedger()
    client = TradeStationClient()
    async with client:
        await backfill_once(client, ledger)
    ledger.close()


if __name__ == '__main__':
    asyncio.run(_main())

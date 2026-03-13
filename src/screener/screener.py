"""
Stock screener for Project Atlas.

Filters the watchlist universe to 5-10 ranked candidates per cycle.
All screening logic runs in Python — Claude only receives the final packages.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.api.tradestation import TradeStationClient
from src.config.constants import (
    BRIEF_ASYNC_SLEEP_SECONDS,
    SCREENER_MAX_SPREAD_PCT,
    SCREENER_MAX_VWAP_DISTANCE_PCT,
    SCREENER_MIN_ATR_PCT,
    SCREENER_MIN_RELATIVE_VOLUME,
    SCREENER_PRICE_MAX,
    SCREENER_PRICE_MIN,
    SCREENER_QUOTE_BATCH_SIZE,
    SCREENER_RSI_MAX,
    SCREENER_RSI_MIN,
    SCREENER_TOP_CANDIDATES,
)
from strategy.thresholds import (
    RANKING_WEIGHT_VOLUME,
    RANKING_WEIGHT_MOMENTUM,
    RANKING_WEIGHT_ATR,
)
from src.screener.indicators import (
    calculate_atr_pct,
    calculate_macd_histogram,
    calculate_momentum_5d,
    calculate_relative_volume,
    calculate_rsi,
    calculate_spread_pct,
    calculate_vwap,
    calculate_vwap_distance_pct,
    determine_market_regime,
)
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'screener')


class StockScreener:
    """
    Filters the watchlist universe to top candidates for Claude.

    Pipeline:
    1. Load watchlist (~150-200 symbols)
    2. Batch-fetch quotes, apply price/spread filter
    3. Fetch daily bars for survivors, apply technical filters
    4. Fetch intraday bars for finalists, compute VWAP
    5. Rank and return top 5-10 candidates
    """

    COLD_THRESHOLD = 12  # Consecutive failures before demotion (~1 hour at 5-min cycles)
    COLD_CHECK_INTERVAL_SECONDS = 3600  # Re-check cold symbols every hour

    def __init__(self, client: TradeStationClient):
        """
        Initialize the screener.

        Args:
            client: Authenticated TradeStationClient instance
        """
        self._client = client
        self._watchlist_path = Path('data/watchlist.json')
        self._cold_stats_path = Path('data/cold_stats.json')
        self._consecutive_failures: Dict[str, int] = {}
        self._cold_demoted_at: Dict[str, datetime] = {}
        self._load_cold_stats()

    def load_watchlist(self) -> List[str]:
        """
        Load stock symbols from the watchlist file.

        Returns:
            List of ticker symbols
        """
        try:
            with open(self._watchlist_path, 'r') as f:
                data = json.load(f)

            symbols = [entry['symbol'] for entry in data.get('symbols', [])]
            logger.info(f"Loaded {len(symbols)} symbols from watchlist")
            return symbols
        except FileNotFoundError:
            logger.error(f"Watchlist not found: {self._watchlist_path}")
            return []
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return []

    def _is_symbol_cold(self, symbol: str) -> bool:
        """
        Check if a symbol is demoted to hourly checks.

        A symbol is cold if it has failed COLD_THRESHOLD consecutive cycles
        and less than COLD_CHECK_INTERVAL_SECONDS have elapsed since demotion.
        When the interval expires, the symbol gets one re-check cycle.

        Args:
            symbol: Ticker symbol

        Returns:
            True if the symbol should be skipped this cycle
        """
        failures = self._consecutive_failures.get(symbol, 0)
        if failures < self.COLD_THRESHOLD:
            return False

        demoted_at = self._cold_demoted_at.get(symbol)
        if demoted_at is None:
            return False

        elapsed = (datetime.now() - demoted_at).total_seconds()
        if elapsed >= self.COLD_CHECK_INTERVAL_SECONDS:
            # Time for hourly re-check — reset demotion clock
            self._cold_demoted_at[symbol] = datetime.now()
            return False

        return True

    def _record_symbol_pass(self, symbol: str) -> None:
        """
        Record that a symbol passed at least the first screening gate.

        Resets the consecutive failure counter and removes cold demotion.

        Args:
            symbol: Ticker symbol
        """
        was_cold = self._consecutive_failures.get(symbol, 0) >= self.COLD_THRESHOLD
        self._consecutive_failures[symbol] = 0
        self._cold_demoted_at.pop(symbol, None)
        if was_cold:
            logger.info(f"Cold list: {symbol} woke up (passed screening)")

    def _record_symbol_failure(self, symbol: str) -> None:
        """
        Record that a symbol failed the first screening gate (price/spread).

        After COLD_THRESHOLD consecutive failures, the symbol is demoted
        to hourly re-checks.

        Args:
            symbol: Ticker symbol
        """
        self._consecutive_failures[symbol] = self._consecutive_failures.get(symbol, 0) + 1
        if self._consecutive_failures[symbol] == self.COLD_THRESHOLD:
            self._cold_demoted_at[symbol] = datetime.now()
            logger.info(
                f"Cold list: {symbol} demoted after {self.COLD_THRESHOLD} "
                f"consecutive failures — hourly re-checks only"
            )

    def get_cold_stats(self) -> Dict[str, int]:
        """
        Return consecutive failure counts for all tracked symbols.

        Used by the watchlist rotation feature to identify persistently
        failing symbols.

        Returns:
            Dict mapping symbol to consecutive failure count
        """
        return dict(self._consecutive_failures)

    def _load_cold_stats(self) -> None:
        """
        Load cold list state from disk on startup.

        Restores consecutive failure counts so the cold list survives restarts.
        Demotion timestamps are re-set to now for any cold symbols (conservative).
        """
        try:
            if not self._cold_stats_path.exists():
                return
            with open(self._cold_stats_path, 'r') as f:
                data = json.load(f)
            self._consecutive_failures = {
                sym: count for sym, count in data.items()
                if isinstance(count, int)
            }
            # Re-set demotion timestamps for symbols already at threshold
            now = datetime.now()
            for sym, count in self._consecutive_failures.items():
                if count >= self.COLD_THRESHOLD:
                    self._cold_demoted_at[sym] = now
            cold_count = sum(
                1 for c in self._consecutive_failures.values()
                if c >= self.COLD_THRESHOLD
            )
            logger.info(
                f"Cold stats loaded: {len(self._consecutive_failures)} symbols tracked, "
                f"{cold_count} cold"
            )
        except Exception as e:
            logger.warning(f"Failed to load cold stats: {e}")

    def _save_cold_stats(self) -> None:
        """
        Persist cold list state to disk after each screening cycle.

        Only saves the failure counts — demotion timestamps are derived on load.
        """
        try:
            tmp_path = self._cold_stats_path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(self._consecutive_failures, f)
            tmp_path.rename(self._cold_stats_path)
        except Exception as e:
            logger.warning(f"Failed to save cold stats: {e}")

    async def fetch_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch quotes for all symbols in batches via the TS API.

        Uses comma-separated symbol queries to minimize API calls.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to quote data
        """
        quotes = {}
        batches = [
            symbols[i:i + SCREENER_QUOTE_BATCH_SIZE]
            for i in range(0, len(symbols), SCREENER_QUOTE_BATCH_SIZE)
        ]

        for batch in batches:
            # TS v3 quotes endpoint accepts comma-separated symbols
            symbols_str = ','.join(batch)
            try:
                result = await self._client._make_request(
                    'GET', f'marketdata/quotes/{symbols_str}'
                )
                if result and isinstance(result, dict) and 'Quotes' in result:
                    for q in result['Quotes']:
                        sym = q.get('Symbol', '')
                        if sym:
                            quotes[sym] = q
                elif result and isinstance(result, list):
                    for q in result:
                        sym = q.get('Symbol', '')
                        if sym:
                            quotes[sym] = q
            except Exception as e:
                logger.warning(f"Quote batch failed: {e}")

            await asyncio.sleep(BRIEF_ASYNC_SLEEP_SECONDS)

        logger.info(f"Fetched {len(quotes)} quotes from {len(symbols)} symbols")
        return quotes

    def apply_price_spread_filter(
        self, quotes: Dict[str, Dict]
    ) -> Tuple[List[str], Dict[str, Dict]]:
        """
        First-pass filter on price range and bid-ask spread.

        Args:
            quotes: Dict mapping symbol to quote data

        Returns:
            Tuple of (surviving symbols, filtered quotes dict)
        """
        survivors = []
        filtered_quotes = {}

        for symbol, quote in quotes.items():
            price = float(quote.get('Last', 0))
            bid = float(quote.get('Bid', 0))
            ask = float(quote.get('Ask', 0))

            if price < SCREENER_PRICE_MIN or price > SCREENER_PRICE_MAX:
                continue

            spread = calculate_spread_pct(bid, ask)
            if spread is None or spread > SCREENER_MAX_SPREAD_PCT:
                continue

            survivors.append(symbol)
            filtered_quotes[symbol] = quote

        logger.info(
            f"Price/spread filter: {len(quotes)} -> {len(survivors)} symbols"
        )
        return survivors, filtered_quotes

    async def fetch_daily_bars(
        self, symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily bars for a list of symbols.

        Args:
            symbols: Symbols to fetch bars for

        Returns:
            Dict mapping symbol to daily OHLCV DataFrame
        """
        bars = {}
        for i, symbol in enumerate(symbols):
            try:
                df = await self._client.get_historical_bars(
                    symbol, interval='daily', bars_back=30
                )
                if df is not None and not df.empty:
                    bars[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to fetch daily bars for {symbol}: {e}")

            # Rate limit: sleep every batch
            if (i + 1) % SCREENER_QUOTE_BATCH_SIZE == 0:
                await asyncio.sleep(BRIEF_ASYNC_SLEEP_SECONDS)

        logger.info(f"Fetched daily bars for {len(bars)}/{len(symbols)} symbols")
        return bars

    def apply_technical_filters(
        self,
        symbols: List[str],
        quotes: Dict[str, Dict],
        daily_bars: Dict[str, pd.DataFrame],
        minutes_since_open: int = 0,
    ) -> List[Dict]:
        """
        Apply technical indicator filters and return candidates with metrics.

        Filters applied in order: relative volume, ATR %, RSI.

        Args:
            symbols: Symbols that passed price/spread filter
            quotes: Quote data by symbol
            daily_bars: Daily bar DataFrames by symbol
            minutes_since_open: Minutes since 9:30 EST for volume normalization

        Returns:
            List of candidate dicts with symbol and computed metrics
        """
        candidates = []

        for symbol in symbols:
            quote = quotes.get(symbol, {})
            bars = daily_bars.get(symbol)
            if bars is None or bars.empty:
                continue

            price = float(quote.get('Last', 0))
            volume = float(quote.get('Volume', 0))

            # Calculate indicators
            rel_vol = calculate_relative_volume(
                volume, bars, minutes_since_open=minutes_since_open
            )
            atr_pct = calculate_atr_pct(bars, price)
            rsi = calculate_rsi(bars)
            momentum = calculate_momentum_5d(bars)
            bid = float(quote.get('Bid', 0))
            ask = float(quote.get('Ask', 0))
            spread = calculate_spread_pct(bid, ask)

            # Apply filters — skip if indicator couldn't be calculated
            if rel_vol is None or rel_vol < SCREENER_MIN_RELATIVE_VOLUME:
                continue
            if atr_pct is None or atr_pct < SCREENER_MIN_ATR_PCT:
                continue
            if rsi is None or rsi < SCREENER_RSI_MIN or rsi > SCREENER_RSI_MAX:
                continue

            candidates.append({
                'symbol': symbol,
                'relative_volume': rel_vol,
                'atr_pct': atr_pct,
                'rsi': rsi,
                'momentum_5d': momentum or 0,
                'spread_pct': spread or 0,
            })

        logger.info(
            f"Technical filters: {len(symbols)} -> {len(candidates)} candidates"
        )
        return candidates

    async def fetch_intraday_bars(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch 1-min and 15-min bars for finalist symbols.

        Args:
            symbols: Symbols to fetch intraday bars for

        Returns:
            Dict mapping symbol to {'1min': df, '15min': df}
        """
        bars = {}
        for symbol in symbols:
            bars[symbol] = {}
            try:
                df_1m = await self._client.get_historical_bars(
                    symbol, interval='1min', bars_back=20
                )
                bars[symbol]['1min'] = df_1m if df_1m is not None else pd.DataFrame()

                await asyncio.sleep(BRIEF_ASYNC_SLEEP_SECONDS)

                df_15m = await self._client.get_historical_bars(
                    symbol, interval='15min', bars_back=20
                )
                bars[symbol]['15min'] = df_15m if df_15m is not None else pd.DataFrame()

            except Exception as e:
                logger.warning(f"Failed to fetch intraday bars for {symbol}: {e}")
                bars[symbol].setdefault('1min', pd.DataFrame())
                bars[symbol].setdefault('15min', pd.DataFrame())

            await asyncio.sleep(BRIEF_ASYNC_SLEEP_SECONDS)

        logger.info(f"Fetched intraday bars for {len(bars)} symbols")
        return bars

    def apply_vwap_filter(
        self,
        candidates: List[Dict],
        quotes: Dict[str, Dict],
        intraday_bars: Dict[str, Dict[str, pd.DataFrame]],
    ) -> List[Dict]:
        """
        Apply VWAP distance filter and market regime classification.

        Computes VWAP distance and classifies each candidate into a market
        regime archetype: TREND_LONG, TREND_SHORT, REVERSAL_LONG, or AVOID.
        Candidates tagged AVOID are filtered out.

        Args:
            candidates: Candidates that passed technical filters
            quotes: Quote data by symbol
            intraday_bars: Intraday bar data by symbol

        Returns:
            Filtered candidates with VWAP metrics, regime, and allowed_direction
        """
        regime_to_direction = {
            'TREND_LONG': 'LONG_ONLY',
            'REVERSAL_LONG': 'LONG_ONLY',
            'TREND_SHORT': 'SHORT_ONLY',
        }

        filtered = []
        for candidate in candidates:
            symbol = candidate['symbol']
            quote = quotes.get(symbol, {})
            bars_1min = intraday_bars.get(symbol, {}).get('1min', pd.DataFrame())

            price = float(quote.get('Last', 0))
            vwap = calculate_vwap(bars_1min)
            vwap_dist = calculate_vwap_distance_pct(vwap, price)

            # Skip if VWAP distance too large
            if vwap_dist is not None and vwap_dist > SCREENER_MAX_VWAP_DISTANCE_PCT:
                continue

            # Classify market regime (passes RSI for reversal gate check)
            rsi = candidate.get('rsi')
            regime = determine_market_regime(bars_1min, price, vwap, rsi=rsi)
            if regime == 'AVOID':
                logger.info(
                    f"Regime filter: {symbol} rejected — AVOID (no clear setup)"
                )
                continue

            # Compute MACD histogram for AI (pre-computed, no LLM math)
            macd_hist, macd_hist_prev = calculate_macd_histogram(bars_1min)

            candidate['vwap'] = vwap or 0
            candidate['vwap_distance_pct'] = vwap_dist or 0
            candidate['macd_histogram'] = macd_hist or 0
            candidate['macd_histogram_prev'] = macd_hist_prev or 0
            candidate['regime'] = regime
            candidate['allowed_direction'] = regime_to_direction[regime]
            filtered.append(candidate)

            logger.info(
                f"Regime filter: {symbol} tagged {regime} "
                f"(allowed: {candidate['allowed_direction']})"
            )

        logger.info(
            f"VWAP filter: {len(candidates)} -> {len(filtered)} candidates"
        )
        return filtered

    def rank_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Rank candidates by composite score and return top N.

        Scoring weights relative volume, momentum, and ATR range.

        Args:
            candidates: Filtered candidates with all metrics

        Returns:
            Top-ranked candidates sorted by score
        """
        if not candidates:
            return []

        for c in candidates:
            c['score'] = (
                c.get('relative_volume', 0) * RANKING_WEIGHT_VOLUME
                + c.get('momentum_5d', 0) * 100 * RANKING_WEIGHT_MOMENTUM
                + c.get('atr_pct', 0) * RANKING_WEIGHT_ATR
            )

        ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
        top = ranked[:SCREENER_TOP_CANDIDATES]

        logger.info(
            f"Ranked {len(candidates)} candidates, returning top {len(top)}"
        )
        return top

    def _get_minutes_since_open(self) -> int:
        """
        Calculate minutes elapsed since market open (9:30 EST).

        Returns:
            Minutes since open (0-390), or 0 if before open
        """
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo('US/Eastern'))
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

        if now < market_open:
            return 0

        delta = now - market_open
        minutes = int(delta.total_seconds() / 60)
        return min(minutes, 390)

    async def screen(self) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Run the full screening pipeline.

        Returns:
            Tuple of:
                - Ranked candidate list with metrics
                - Quotes dict by symbol
                - Bars dict by symbol (daily + intraday)
        """
        start = time.time()
        logger.info("Starting screening cycle")

        # 1. Load watchlist
        symbols = self.load_watchlist()
        if not symbols:
            logger.warning("Empty watchlist — no candidates to screen")
            return [], {}, {}

        # 1b. Filter out cold symbols (demoted to hourly checks)
        active_symbols = [s for s in symbols if not self._is_symbol_cold(s)]
        cold_count = len(symbols) - len(active_symbols)
        if cold_count > 0:
            logger.info(f"Cold list: {cold_count} symbols demoted to hourly checks")

        # 2. Fetch quotes and apply price/spread filter
        quotes = await self.fetch_quotes_batch(active_symbols)
        survivors, filtered_quotes = self.apply_price_spread_filter(quotes)

        # Track pass/fail for cold list
        passed_set = set(survivors)
        for sym in active_symbols:
            if sym in passed_set:
                self._record_symbol_pass(sym)
            else:
                self._record_symbol_failure(sym)
        if not survivors:
            logger.warning("No symbols passed price/spread filter")
            return [], {}, {}

        # 3. Fetch daily bars and apply technical filters
        daily_bars = await self.fetch_daily_bars(survivors)
        minutes = self._get_minutes_since_open()
        candidates = self.apply_technical_filters(
            survivors, filtered_quotes, daily_bars, minutes
        )
        if not candidates:
            logger.warning("No symbols passed technical filters")
            return [], {}, {}

        # 4. Fetch intraday bars for survivors
        finalist_symbols = [c['symbol'] for c in candidates]
        intraday_bars = await self.fetch_intraday_bars(finalist_symbols)

        # 5. Apply VWAP filter
        candidates = self.apply_vwap_filter(candidates, filtered_quotes, intraday_bars)

        # 6. Rank and return top candidates
        ranked = self.rank_candidates(candidates)

        # Merge bar data for the builder
        all_bars = {}
        for symbol in [c['symbol'] for c in ranked]:
            all_bars[symbol] = {
                'daily': daily_bars.get(symbol, pd.DataFrame()),
                '1min': intraday_bars.get(symbol, {}).get('1min', pd.DataFrame()),
                '15min': intraday_bars.get(symbol, {}).get('15min', pd.DataFrame()),
            }

        elapsed = time.time() - start
        logger.info(
            f"Screening complete: {len(ranked)} candidates in {elapsed:.1f}s"
        )

        # Persist cold stats for resume across restarts
        self._save_cold_stats()

        return ranked, filtered_quotes, all_bars

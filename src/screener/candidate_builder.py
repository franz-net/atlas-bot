"""
Candidate package builder for the Atlas decision engine.

Assembles the structured data package (SPEC Section 3.4) that Claude
receives for each screened candidate. Pure data transformation — no
API calls.
"""

import logging
from datetime import datetime, time, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'candidate_builder')


class CandidateBuilder:
    """
    Builds structured candidate packages matching SPEC Section 3.4 format.

    Each package contains all data Claude needs to make a trading decision:
    price, indicators, bar data, and news (empty until Sprint 3).
    """

    def build_candidate_package(
        self,
        symbol: str,
        quote: Dict,
        metrics: Dict,
        bars: Dict[str, pd.DataFrame],
        news: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Assemble a single candidate package for Claude.

        Args:
            symbol: Ticker symbol
            quote: Quote dict from TradeStation (Last, Bid, Ask, Volume)
            metrics: Computed indicators from screener (atr_pct, rsi, etc.)
            bars: Dict of DataFrames keyed by timeframe ('daily', '15min', '1min')
            news: List of news items from NewsFetcher (empty array if none)

        Returns:
            Complete candidate package matching SPEC 3.4 format
        """
        daily = bars.get('daily', pd.DataFrame())
        recent_high = float(daily['High'].tail(20).max()) if not daily.empty else 0.0
        recent_low = float(daily['Low'].tail(20).min()) if not daily.empty else 0.0

        return {
            'symbol': symbol,
            'price': float(quote.get('Last', 0)),
            'bid': float(quote.get('Bid', 0)),
            'ask': float(quote.get('Ask', 0)),
            'spread_pct': metrics.get('spread_pct', 0),
            'relative_volume': metrics.get('relative_volume', 0),
            'atr_pct': metrics.get('atr_pct', 0),
            'rsi_14': metrics.get('rsi', 0),
            'vwap': metrics.get('vwap', 0),
            'vwap_distance_pct': metrics.get('vwap_distance_pct', 0),
            'momentum_5d': metrics.get('momentum_5d', 0),
            'macd_histogram': metrics.get('macd_histogram', 0),
            'macd_histogram_prev': metrics.get('macd_histogram_prev', 0),
            'regime': metrics.get('regime', 'TREND_LONG'),
            'allowed_direction': metrics.get('allowed_direction', 'LONG_ONLY'),
            'recent_high': recent_high,
            'recent_low': recent_low,
            'news': (news if news is not None else [])[:3],
        }

    def build_all_packages(
        self,
        candidates: List[Dict],
        quotes: Dict[str, Dict],
        bars_data: Dict[str, Dict[str, pd.DataFrame]],
        news_data: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Dict]:
        """
        Build packages for all ranked candidates.

        Args:
            candidates: Ranked candidate list from screener (each has 'symbol' + metrics)
            quotes: Dict of quotes keyed by symbol
            bars_data: Dict of bar DataFrames keyed by symbol then timeframe
            news_data: Dict mapping symbol to list of news items

        Returns:
            List of complete candidate packages for Claude
        """
        if news_data is None:
            news_data = {}

        packages = []
        for candidate in candidates:
            symbol = candidate['symbol']
            try:
                quote = quotes.get(symbol, {})
                bars = bars_data.get(symbol, {})
                news = news_data.get(symbol, [])
                package = self.build_candidate_package(
                    symbol, quote, candidate, bars, news,
                )
                packages.append(package)
            except Exception as e:
                logger.warning(f"Failed to build package for {symbol}: {e}")
                continue

        logger.info(f"Built {len(packages)} candidate packages")
        return packages

    @staticmethod
    def get_session_phase() -> str:
        """
        Determine the current trading session phase.

        Returns:
            'MORNING_VOLATILITY' (9:30-10:00), 'NORMAL' (10:00-15:30),
            or 'EOD_CLOSE' (15:30+)
        """
        est = ZoneInfo('US/Eastern')
        now = datetime.now(est).time()

        if now < time(10, 0):
            return 'MORNING_VOLATILITY'
        elif now >= time(15, 30):
            return 'EOD_CLOSE'
        return 'NORMAL'

    def _serialize_bars(self, df: pd.DataFrame, count: int = 20) -> List[Dict]:
        """
        Convert DataFrame tail to JSON-serializable list of dicts.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            count: Number of recent bars to include

        Returns:
            List of bar dicts with ISO timestamp strings
        """
        if df is None or df.empty:
            return []

        try:
            tail = df.tail(count).copy()
            records = []
            for idx, row in tail.iterrows():
                record = {
                    'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    'open': float(row.get('Open', 0)),
                    'high': float(row.get('High', 0)),
                    'low': float(row.get('Low', 0)),
                    'close': float(row.get('Close', 0)),
                    'volume': int(row.get('Volume', 0)),
                }
                records.append(record)
            return records
        except Exception as e:
            logger.warning(f"Bar serialization failed: {e}")
            return []

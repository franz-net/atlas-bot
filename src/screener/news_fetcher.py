"""
News fetcher for Project Atlas.

Pulls company news headlines from Finnhub API. Headlines from the last
4 hours are included in candidate packages for the AI decision engine.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import aiohttp

from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'news_fetcher')

FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'

# Simple keyword lists for basic sentiment tagging
_POSITIVE_KEYWORDS = [
    'upgrade', 'upgrades', 'upgraded', 'buy', 'outperform', 'overweight',
    'beat', 'beats', 'exceeds', 'raises', 'raised', 'record', 'surge',
    'surges', 'soars', 'jumps', 'gains', 'bullish', 'strong', 'growth',
    'profit', 'dividend', 'approval', 'approved', 'partnership', 'deal',
    'acquisition', 'breakout', 'positive', 'optimistic', 'boost',
]

_NEGATIVE_KEYWORDS = [
    'downgrade', 'downgrades', 'downgraded', 'sell', 'underperform',
    'underweight', 'miss', 'misses', 'missed', 'lowers', 'lowered',
    'cut', 'cuts', 'decline', 'declines', 'drops', 'falls', 'plunges',
    'bearish', 'weak', 'loss', 'lawsuit', 'investigation', 'recall',
    'warning', 'negative', 'concern', 'risk', 'layoffs', 'restructuring',
]


class NewsFetcher:
    """
    Fetches company news headlines from Finnhub API.

    Headlines are tagged with basic sentiment (positive/negative/neutral)
    using keyword matching. Rate limited to stay within Finnhub's 30 req/sec.
    """

    def __init__(self, client=None):
        """
        Initialize the news fetcher.

        Args:
            client: Unused — kept for backward compatibility with scheduler init.
        """
        self._api_key = os.getenv('FINNHUB_API_KEY', '')
        if not self._api_key:
            logger.warning("FINNHUB_API_KEY not set — news fetch will be disabled")

    async def fetch_news(
        self, symbol: str, hours_back: int = 4
    ) -> List[Dict]:
        """
        Fetch recent news headlines for a symbol from Finnhub.

        Args:
            symbol: Ticker symbol
            hours_back: How many hours of headlines to include

        Returns:
            List of news dicts with headline, source, timestamp, sentiment
        """
        if not self._api_key:
            return []

        try:
            now = datetime.now(timezone.utc)
            date_to = now.strftime('%Y-%m-%d')
            date_from = (now - timedelta(hours=hours_back)).strftime('%Y-%m-%d')

            url = (
                f"{FINNHUB_BASE_URL}/company-news"
                f"?symbol={symbol}&from={date_from}&to={date_to}"
            )
            headers = {'X-Finnhub-Token': self._api_key}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 429:
                        logger.warning(f"Finnhub rate limit hit for {symbol}")
                        return []
                    if resp.status != 200:
                        logger.warning(f"Finnhub {symbol}: status {resp.status}")
                        return []

                    items = await resp.json()

            if not isinstance(items, list):
                return []

            cutoff = now - timedelta(hours=hours_back)
            headlines = []
            for item in items:
                parsed = self._parse_news_item(item, cutoff)
                if parsed:
                    headlines.append(parsed)

            logger.info(f"{symbol}: {len(headlines)} headlines in last {hours_back}h")
            return headlines

        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return []

    async def fetch_news_batch(
        self, symbols: List[str], hours_back: int = 4
    ) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple symbols with rate limiting.

        Finnhub allows 30 req/sec — we add a small delay to stay safe.

        Args:
            symbols: List of ticker symbols
            hours_back: How many hours of headlines to include

        Returns:
            Dict mapping symbol to list of news items
        """
        news = {}
        for symbol in symbols:
            news[symbol] = await self.fetch_news(symbol, hours_back)
            await asyncio.sleep(0.05)  # ~20 req/sec max, well within 30/sec limit

        total = sum(len(items) for items in news.values())
        logger.info(f"Fetched {total} total headlines for {len(symbols)} symbols")
        return news

    def _parse_news_item(
        self, item: Dict, cutoff: datetime
    ) -> Optional[Dict]:
        """
        Parse a Finnhub news item into the normalized format.

        Args:
            item: Raw news item from Finnhub
            cutoff: Oldest timestamp to include

        Returns:
            Normalized news dict, or None if too old or unparseable
        """
        try:
            headline = item.get('headline', '')
            if not headline:
                return None

            # Finnhub returns UNIX timestamp
            unix_ts = item.get('datetime', 0)
            if unix_ts:
                ts = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
                if ts < cutoff:
                    return None
                timestamp_iso = ts.isoformat()
            else:
                timestamp_iso = datetime.now(timezone.utc).isoformat()

            sentiment = self._tag_sentiment(headline)

            return {
                'headline': headline,
                'source': item.get('source', 'unknown'),
                'timestamp': timestamp_iso,
                'sentiment': sentiment,
                'summary': item.get('summary', ''),
            }

        except Exception as e:
            logger.warning(f"Failed to parse news item: {e}")
            return None

    def _tag_sentiment(self, headline: str) -> str:
        """
        Tag headline sentiment using keyword matching.

        Args:
            headline: News headline text

        Returns:
            'positive', 'negative', or 'neutral'
        """
        lower = headline.lower()

        pos_count = sum(1 for kw in _POSITIVE_KEYWORDS if kw in lower)
        neg_count = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in lower)

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'

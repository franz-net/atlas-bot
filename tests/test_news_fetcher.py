"""
Sprint 3 Tests — News Fetcher (Finnhub)

Tests use mocked API responses — no live calls.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.screener.news_fetcher import NewsFetcher


@pytest.fixture
def fetcher():
    with patch.dict('os.environ', {'FINNHUB_API_KEY': 'test-key'}):
        return NewsFetcher()


class TestSentimentTagging:
    """Test keyword-based sentiment tagging."""

    def test_positive_headline(self, fetcher):
        assert fetcher._tag_sentiment('Company upgrades guidance after record earnings') == 'positive'

    def test_negative_headline(self, fetcher):
        assert fetcher._tag_sentiment('Stock plunges after CEO announces layoffs') == 'negative'

    def test_neutral_headline(self, fetcher):
        assert fetcher._tag_sentiment('Company releases quarterly report') == 'neutral'

    def test_mixed_defaults_to_count(self, fetcher):
        result = fetcher._tag_sentiment('Stock surges and gains on upgrade despite some concern')
        assert result == 'positive'


class TestNewsItemParsing:
    """Test parsing of Finnhub news items."""

    def test_parse_valid_item(self, fetcher):
        now = datetime.now(timezone.utc)
        item = {
            'headline': 'NVDA beats earnings estimates',
            'source': 'Reuters',
            'datetime': int(now.timestamp()),
            'summary': 'Full article summary here.',
        }
        cutoff = now - timedelta(hours=4)
        result = fetcher._parse_news_item(item, cutoff)

        assert result is not None
        assert result['headline'] == 'NVDA beats earnings estimates'
        assert result['source'] == 'Reuters'
        assert result['sentiment'] == 'positive'
        assert result['summary'] == 'Full article summary here.'

    def test_parse_old_item_filtered(self, fetcher):
        old_time = datetime.now(timezone.utc) - timedelta(hours=5)
        item = {
            'headline': 'Old news',
            'datetime': int(old_time.timestamp()),
        }
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        result = fetcher._parse_news_item(item, cutoff)
        assert result is None

    def test_parse_missing_headline(self, fetcher):
        item = {'source': 'Reuters'}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        assert fetcher._parse_news_item(item, cutoff) is None

    def test_parse_no_timestamp(self, fetcher):
        item = {'headline': 'Breaking news'}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
        result = fetcher._parse_news_item(item, cutoff)
        assert result is not None
        assert result['headline'] == 'Breaking news'


class TestFetchNews:
    """Test full fetch flow with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_fetch_returns_headlines(self, fetcher):
        now = datetime.now(timezone.utc)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {
                'headline': 'AAPL beats Q4 estimates',
                'source': 'Reuters',
                'datetime': int(now.timestamp()),
                'summary': 'Apple reported strong earnings.',
            },
            {
                'headline': 'AAPL announces new product',
                'source': 'CNBC',
                'datetime': int(now.timestamp()),
                'summary': 'New product launch.',
            },
        ])

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch('src.screener.news_fetcher.aiohttp.ClientSession') as mock_cs:
            mock_cs.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_session),
                __aexit__=AsyncMock(return_value=False),
            )
            headlines = await fetcher.fetch_news('AAPL')

        assert len(headlines) == 2
        assert headlines[0]['headline'] == 'AAPL beats Q4 estimates'

    @pytest.mark.asyncio
    async def test_fetch_no_api_key(self):
        with patch.dict('os.environ', {'FINNHUB_API_KEY': ''}):
            fetcher = NewsFetcher()
            headlines = await fetcher.fetch_news('AAPL')
            assert headlines == []

    @pytest.mark.asyncio
    async def test_fetch_rate_limited(self, fetcher):
        mock_response = AsyncMock()
        mock_response.status = 429

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch('src.screener.news_fetcher.aiohttp.ClientSession') as mock_cs:
            mock_cs.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_session),
                __aexit__=AsyncMock(return_value=False),
            )
            headlines = await fetcher.fetch_news('AAPL')

        assert headlines == []

    @pytest.mark.asyncio
    async def test_fetch_api_error(self, fetcher):
        with patch('src.screener.news_fetcher.aiohttp.ClientSession', side_effect=Exception('Network error')):
            headlines = await fetcher.fetch_news('AAPL')
            assert headlines == []

    @pytest.mark.asyncio
    async def test_fetch_batch(self, fetcher):
        """Batch fetch calls fetch_news for each symbol."""
        now = datetime.now(timezone.utc)

        async def mock_fetch(symbol, hours_back=4):
            return [{'headline': f'{symbol} news', 'source': 'test',
                     'timestamp': now.isoformat(), 'sentiment': 'neutral'}]

        fetcher.fetch_news = mock_fetch
        result = await fetcher.fetch_news_batch(['AAPL', 'NVDA'])
        assert 'AAPL' in result
        assert 'NVDA' in result
        assert len(result['AAPL']) == 1


class TestNewsInCandidatePackage:
    """Test that news integrates correctly into candidate packages."""

    def test_package_with_news(self):
        import pandas as pd
        from src.screener.candidate_builder import CandidateBuilder

        builder = CandidateBuilder()
        news = [
            {
                'headline': 'Stock surges on earnings beat',
                'source': 'Reuters',
                'timestamp': '2026-03-10T10:00:00+00:00',
                'sentiment': 'positive',
            }
        ]

        quote = {'Last': '100', 'Bid': '99.99', 'Ask': '100.01'}
        metrics = {
            'spread_pct': 0.01, 'relative_volume': 2.0,
            'atr_pct': 2.0, 'rsi': 55, 'vwap': 100,
            'vwap_distance_pct': 0.5, 'momentum_5d': 0.03,
        }
        bars = {'daily': pd.DataFrame(), '1min': pd.DataFrame(), '15min': pd.DataFrame()}

        package = builder.build_candidate_package('TEST', quote, metrics, bars, news)
        assert len(package['news']) == 1
        assert package['news'][0]['sentiment'] == 'positive'

    def test_package_without_news(self):
        import pandas as pd
        from src.screener.candidate_builder import CandidateBuilder

        builder = CandidateBuilder()
        quote = {'Last': '100', 'Bid': '99.99', 'Ask': '100.01'}
        metrics = {
            'spread_pct': 0.01, 'relative_volume': 2.0,
            'atr_pct': 2.0, 'rsi': 55, 'vwap': 100,
            'vwap_distance_pct': 0.5, 'momentum_5d': 0.03,
        }
        bars = {'daily': pd.DataFrame(), '1min': pd.DataFrame(), '15min': pd.DataFrame()}

        package = builder.build_candidate_package('TEST', quote, metrics, bars)
        assert package['news'] == []

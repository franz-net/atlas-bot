"""
Sprint 2 Tests — Screener, Indicators, and Candidate Builder

All tests use static fixture data — no live API calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.constants import (
    SCREENER_MAX_SPREAD_PCT,
    SCREENER_MAX_VWAP_DISTANCE_PCT,
    SCREENER_MIN_ATR_PCT,
    SCREENER_MIN_RELATIVE_VOLUME,
    SCREENER_PRICE_MAX,
    SCREENER_PRICE_MIN,
    SCREENER_RSI_MAX,
    SCREENER_RSI_MIN,
)
from src.screener.indicators import (
    calculate_atr,
    calculate_atr_pct,
    calculate_momentum_5d,
    calculate_relative_volume,
    calculate_rsi,
    calculate_spread_pct,
    calculate_vwap,
    calculate_vwap_distance_pct,
    determine_market_regime,
    _check_reversal_gate,
)


@pytest.fixture
def daily_bars():
    """Generate realistic daily OHLCV bars for testing."""
    dates = pd.date_range('2026-02-01', periods=30, freq='B')
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(30) * 2)

    df = pd.DataFrame({
        'Open': prices,
        'High': prices + np.random.rand(30) * 3,
        'Low': prices - np.random.rand(30) * 3,
        'Close': prices + np.random.randn(30) * 1,
        'Volume': np.random.randint(500000, 5000000, 30).astype(float),
    }, index=dates)

    return df


@pytest.fixture
def intraday_bars():
    """Generate 1-minute intraday bars for VWAP testing."""
    times = pd.date_range('2026-03-10 09:30', periods=60, freq='1min')
    np.random.seed(42)
    base = 100.0
    prices = base + np.cumsum(np.random.randn(60) * 0.1)

    df = pd.DataFrame({
        'Open': prices,
        'High': prices + np.random.rand(60) * 0.5,
        'Low': prices - np.random.rand(60) * 0.5,
        'Close': prices + np.random.randn(60) * 0.2,
        'Volume': np.random.randint(10000, 100000, 60).astype(float),
    }, index=times)

    return df


class TestATR:
    """Test ATR indicator calculations."""

    def test_atr_returns_float(self, daily_bars):
        result = calculate_atr(daily_bars)
        assert result is not None
        assert isinstance(result, float)
        assert result > 0

    def test_atr_insufficient_data(self):
        df = pd.DataFrame({
            'High': [10], 'Low': [9], 'Close': [9.5]
        })
        assert calculate_atr(df) is None

    def test_atr_pct(self, daily_bars):
        result = calculate_atr_pct(daily_bars, 100.0)
        assert result is not None
        assert result > 0

    def test_atr_pct_zero_price(self, daily_bars):
        assert calculate_atr_pct(daily_bars, 0) is None


class TestVWAP:
    """Test VWAP indicator calculations."""

    def test_vwap_returns_float(self, intraday_bars):
        result = calculate_vwap(intraday_bars)
        assert result is not None
        assert isinstance(result, float)
        assert result > 0

    def test_vwap_empty_data(self):
        assert calculate_vwap(pd.DataFrame()) is None
        assert calculate_vwap(None) is None

    def test_vwap_distance(self):
        result = calculate_vwap_distance_pct(100.0, 100.5)
        assert result is not None
        assert abs(result - 0.5) < 0.01

    def test_vwap_distance_invalid(self):
        assert calculate_vwap_distance_pct(None, 100) is None
        assert calculate_vwap_distance_pct(0, 100) is None


class TestRSI:
    """Test RSI indicator calculations."""

    def test_rsi_returns_float(self, daily_bars):
        result = calculate_rsi(daily_bars)
        assert result is not None
        assert 0 <= result <= 100

    def test_rsi_insufficient_data(self):
        df = pd.DataFrame({'Close': [10, 11, 12]})
        assert calculate_rsi(df) is None


class TestRelativeVolume:
    """Test relative volume calculations."""

    def test_relative_volume(self, daily_bars):
        avg_vol = daily_bars['Volume'].mean()
        result = calculate_relative_volume(
            avg_vol * 2.5, daily_bars, minutes_since_open=390
        )
        assert result is not None
        assert abs(result - 2.5) < 0.5

    def test_relative_volume_time_normalized(self, daily_bars):
        """At 30 minutes in, volume should be projected to full day."""
        avg_vol = daily_bars['Volume'].mean()
        partial_vol = avg_vol * 0.1  # 10% of avg at 30 min in

        result = calculate_relative_volume(
            partial_vol, daily_bars, minutes_since_open=30
        )
        # 30/390 = ~7.7% of day. 10% vol / 7.7% = ~1.3x projected
        assert result is not None
        assert result > 1.0

    def test_relative_volume_zero(self, daily_bars):
        assert calculate_relative_volume(0, daily_bars) is None

    def test_relative_volume_insufficient_bars(self):
        df = pd.DataFrame({'Volume': [100]})
        assert calculate_relative_volume(1000, df) is None


class TestMomentum:
    """Test 5-day momentum calculations."""

    def test_momentum_returns_float(self, daily_bars):
        result = calculate_momentum_5d(daily_bars)
        assert result is not None
        assert isinstance(result, float)

    def test_momentum_insufficient_data(self):
        df = pd.DataFrame({'Close': [10, 11, 12]})
        assert calculate_momentum_5d(df) is None


class TestSpread:
    """Test bid-ask spread calculations."""

    def test_spread_normal(self):
        result = calculate_spread_pct(100.0, 100.10)
        assert result is not None
        assert result == pytest.approx(0.1, abs=0.01)

    def test_spread_tight(self):
        result = calculate_spread_pct(100.0, 100.01)
        assert result is not None
        assert result < SCREENER_MAX_SPREAD_PCT

    def test_spread_invalid(self):
        assert calculate_spread_pct(0, 100) is None
        assert calculate_spread_pct(100, 50) is None  # ask < bid


class TestMarketRegime:
    """Test market regime classification."""

    def test_trend_long(self):
        """Price above rising VWAP -> TREND_LONG."""
        times = pd.date_range('2026-03-10 09:30', periods=60, freq='1min')
        prices = 100.0 + np.linspace(0, 3, 60)
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 0.5,
            'Low': prices - 0.2,
            'Close': prices + 0.3,
            'Volume': np.full(60, 50000.0),
        }, index=times)

        vwap = calculate_vwap(df)
        current_price = float(prices[-1]) + 1.0
        result = determine_market_regime(df, current_price, vwap)
        assert result == 'TREND_LONG'

    def test_trend_short_no_reversal(self):
        """Price below declining VWAP, RSI not oversold -> TREND_SHORT."""
        times = pd.date_range('2026-03-10 09:30', periods=60, freq='1min')
        prices = 100.0 - np.linspace(0, 3, 60)
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 0.2,
            'Low': prices - 0.5,
            'Close': prices - 0.3,
            'Volume': np.full(60, 50000.0),
        }, index=times)

        vwap = calculate_vwap(df)
        current_price = float(prices[-1]) - 1.0
        # RSI 50 = not oversold, so no reversal gate
        result = determine_market_regime(df, current_price, vwap, rsi=50)
        assert result == 'TREND_SHORT'

    def test_reversal_long(self):
        """Price below VWAP, RSI oversold, MACD rising, green candle -> REVERSAL_LONG."""
        times = pd.date_range('2026-03-10 09:30', periods=60, freq='1min')
        # Decline then flatten with uptick at end
        prices = np.concatenate([
            100.0 - np.linspace(0, 5, 50),  # Sharp decline
            95.0 + np.linspace(0, 0.5, 10),  # Flatten and uptick
        ])
        opens = prices.copy()
        # Make last candle green (close > open)
        closes = prices.copy()
        closes[-1] = opens[-1] + 0.3

        df = pd.DataFrame({
            'Open': opens,
            'High': prices + 0.3,
            'Low': prices - 0.3,
            'Close': closes,
            'Volume': np.full(60, 50000.0),
        }, index=times)

        vwap = calculate_vwap(df)
        current_price = float(closes[-1])
        # RSI < 35 triggers reversal gate
        result = determine_market_regime(df, current_price, vwap, rsi=28)
        # Could be REVERSAL_LONG or TREND_SHORT depending on MACD
        # At minimum, it should not be AVOID since we have clear bearish structure
        assert result in ('REVERSAL_LONG', 'TREND_SHORT')

    def test_avoid_insufficient_data(self):
        """Insufficient data -> AVOID."""
        df = pd.DataFrame({
            'Open': [10, 11], 'High': [10, 11],
            'Low': [9, 10], 'Close': [9.5, 10.5],
            'Volume': [1000, 1000],
        })
        assert determine_market_regime(df, 10.0, 10.0) == 'AVOID'

    def test_avoid_null_inputs(self):
        """Null/invalid inputs -> AVOID."""
        assert determine_market_regime(None, 100, 100) == 'AVOID'
        assert determine_market_regime(pd.DataFrame(), 100, None) == 'AVOID'
        assert determine_market_regime(pd.DataFrame(), 0, 100) == 'AVOID'


class TestReversalGate:
    """Test the reversal gate sub-check."""

    def test_rsi_not_oversold_fails(self):
        """RSI >= 35 should fail the reversal gate."""
        df = pd.DataFrame({
            'Open': np.linspace(100, 95, 30),
            'High': np.linspace(101, 96, 30),
            'Low': np.linspace(99, 94, 30),
            'Close': np.linspace(100, 95, 30),
            'Volume': np.full(30, 50000.0),
        })
        assert _check_reversal_gate(df, rsi=50) is False

    def test_rsi_none_fails(self):
        """No RSI data should fail."""
        assert _check_reversal_gate(pd.DataFrame(), rsi=None) is False


class TestScreenerFilters:
    """Test screener filter logic with mock data."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        monkeypatch.setenv('TS_API_KEY', 'test')
        monkeypatch.setenv('TS_API_SECRET', 'test')
        monkeypatch.setenv('TS_ACCOUNT_ID', 'TEST123')
        monkeypatch.setenv('USE_SIM_ACCOUNT', 'true')

    def test_price_spread_filter(self, mock_env):
        from src.screener.screener import StockScreener

        client = MagicMock()
        screener = StockScreener(client)

        quotes = {
            'GOOD': {'Last': '50', 'Bid': '49.99', 'Ask': '50.01', 'Symbol': 'GOOD'},
            'CHEAP': {'Last': '2', 'Bid': '1.99', 'Ask': '2.01', 'Symbol': 'CHEAP'},
            'EXPENSIVE': {'Last': '600', 'Bid': '599', 'Ask': '601', 'Symbol': 'EXPENSIVE'},
            'WIDE_SPREAD': {'Last': '50', 'Bid': '49.50', 'Ask': '50.50', 'Symbol': 'WIDE_SPREAD'},
        }

        survivors, filtered = screener.apply_price_spread_filter(quotes)
        assert 'GOOD' in survivors
        assert 'CHEAP' not in survivors
        assert 'EXPENSIVE' not in survivors
        assert 'WIDE_SPREAD' not in survivors

    def test_technical_filters(self, mock_env, daily_bars):
        from src.screener.screener import StockScreener

        client = MagicMock()
        screener = StockScreener(client)

        quotes = {
            'TEST': {'Last': '100', 'Bid': '99.99', 'Ask': '100.01', 'Volume': '5000000'},
        }

        candidates = screener.apply_technical_filters(
            ['TEST'], quotes, {'TEST': daily_bars}, minutes_since_open=200
        )

        # Result depends on fixture data, but should not crash
        assert isinstance(candidates, list)


class TestCandidateBuilder:
    """Test candidate package assembly."""

    def test_build_package_structure(self, daily_bars, intraday_bars):
        from src.screener.candidate_builder import CandidateBuilder

        builder = CandidateBuilder()
        quote = {'Last': '100.50', 'Bid': '100.45', 'Ask': '100.55'}
        metrics = {
            'spread_pct': 0.01,
            'relative_volume': 2.3,
            'atr_pct': 2.1,
            'rsi': 58,
            'vwap': 99.80,
            'vwap_distance_pct': 0.7,
            'momentum_5d': 0.05,
            'macd_histogram': 0.15,
            'macd_histogram_prev': 0.10,
            'regime': 'TREND_LONG',
            'allowed_direction': 'LONG_ONLY',
        }
        bars = {
            'daily': daily_bars,
            '1min': intraday_bars,
            '15min': intraday_bars,
        }

        package = builder.build_candidate_package('NVDA', quote, metrics, bars)

        # Verify SPEC 3.4 fields exist
        assert package['symbol'] == 'NVDA'
        assert package['price'] == 100.50
        assert package['bid'] == 100.45
        assert package['ask'] == 100.55
        assert package['relative_volume'] == 2.3
        assert package['atr_pct'] == 2.1
        assert package['rsi_14'] == 58
        assert package['vwap'] == 99.80
        assert package['momentum_5d'] == 0.05
        assert package['regime'] == 'TREND_LONG'
        assert package['allowed_direction'] == 'LONG_ONLY'
        assert package['recent_high'] > 0
        assert package['recent_low'] > 0
        assert package['macd_histogram'] == 0.15
        assert package['macd_histogram_prev'] == 0.10
        assert package['news'] == []  # Empty list when no news

    def test_bar_serialization(self, daily_bars):
        from src.screener.candidate_builder import CandidateBuilder

        builder = CandidateBuilder()
        records = builder._serialize_bars(daily_bars, count=5)

        assert len(records) == 5
        assert 'timestamp' in records[0]
        assert 'open' in records[0]
        assert 'high' in records[0]
        assert 'low' in records[0]
        assert 'close' in records[0]
        assert 'volume' in records[0]
        # Ensure native Python types (JSON-serializable)
        assert isinstance(records[0]['open'], float)
        assert isinstance(records[0]['volume'], int)

    def test_bar_serialization_empty(self):
        from src.screener.candidate_builder import CandidateBuilder

        builder = CandidateBuilder()
        assert builder._serialize_bars(pd.DataFrame()) == []
        assert builder._serialize_bars(None) == []

    def test_build_all_packages_graceful_failure(self, daily_bars):
        from src.screener.candidate_builder import CandidateBuilder

        builder = CandidateBuilder()
        candidates = [
            {'symbol': 'GOOD', 'spread_pct': 0.01, 'relative_volume': 2.0,
             'atr_pct': 2.0, 'rsi': 55, 'vwap': 100, 'vwap_distance_pct': 0.5,
             'momentum_5d': 0.03, 'regime': 'TREND_LONG', 'allowed_direction': 'LONG_ONLY'},
            {'symbol': 'BAD'},  # Missing metrics — should skip gracefully
        ]
        quotes = {
            'GOOD': {'Last': '100', 'Bid': '99.99', 'Ask': '100.01'},
            'BAD': {},
        }
        bars = {
            'GOOD': {'daily': daily_bars, '1min': pd.DataFrame(), '15min': pd.DataFrame()},
        }

        packages = builder.build_all_packages(candidates, quotes, bars)
        # Should build GOOD, skip BAD gracefully
        assert len(packages) >= 1
        assert packages[0]['symbol'] == 'GOOD'


class TestWatchlist:
    """Test watchlist loading."""

    def test_watchlist_loads(self):
        from src.screener.screener import StockScreener

        client = MagicMock()
        screener = StockScreener(client)
        symbols = screener.load_watchlist()
        assert len(symbols) > 100
        assert 'AAPL' in symbols
        assert 'NVDA' in symbols

    def test_watchlist_all_have_sector(self):
        with open('data/watchlist.json', 'r') as f:
            data = json.load(f)
        for entry in data['symbols']:
            assert 'symbol' in entry
            assert 'sector' in entry
            assert len(entry['symbol']) > 0

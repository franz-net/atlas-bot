"""
Tests for the cold list pre-filter in StockScreener.

Verifies that symbols which consistently fail screening get demoted
to hourly re-checks, and wake up when they pass.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.screener.screener import StockScreener


@pytest.fixture
def screener(tmp_path):
    """Create a screener with a mock client and clean cold stats."""
    client = MagicMock()
    s = StockScreener(client)
    s._cold_stats_path = tmp_path / 'cold_stats.json'
    s._consecutive_failures = {}
    s._cold_demoted_at = {}
    return s


class TestColdTracking:
    """Tests for cold list state management."""

    def test_new_symbol_not_cold(self, screener):
        assert not screener._is_symbol_cold('AAPL')

    def test_below_threshold_not_cold(self, screener):
        screener._consecutive_failures['AAPL'] = 11
        assert not screener._is_symbol_cold('AAPL')

    def test_at_threshold_is_cold(self, screener):
        screener._consecutive_failures['AAPL'] = 12
        screener._cold_demoted_at['AAPL'] = datetime.now()
        assert screener._is_symbol_cold('AAPL')

    def test_cold_symbol_wakes_up_after_interval(self, screener):
        screener._consecutive_failures['AAPL'] = 12
        # Demoted over an hour ago
        screener._cold_demoted_at['AAPL'] = datetime.now() - timedelta(seconds=3601)
        assert not screener._is_symbol_cold('AAPL')

    def test_record_failure_increments(self, screener):
        screener._record_symbol_failure('AAPL')
        assert screener._consecutive_failures['AAPL'] == 1
        screener._record_symbol_failure('AAPL')
        assert screener._consecutive_failures['AAPL'] == 2

    def test_record_failure_demotes_at_threshold(self, screener):
        for _ in range(12):
            screener._record_symbol_failure('AAPL')
        assert screener._consecutive_failures['AAPL'] == 12
        assert 'AAPL' in screener._cold_demoted_at

    def test_record_pass_resets_counter(self, screener):
        screener._consecutive_failures['AAPL'] = 12
        screener._cold_demoted_at['AAPL'] = datetime.now()
        screener._record_symbol_pass('AAPL')
        assert screener._consecutive_failures['AAPL'] == 0
        assert 'AAPL' not in screener._cold_demoted_at

    def test_get_cold_stats(self, screener):
        screener._consecutive_failures['AAPL'] = 5
        screener._consecutive_failures['MSFT'] = 12
        stats = screener.get_cold_stats()
        assert stats == {'AAPL': 5, 'MSFT': 12}

    def test_get_cold_stats_returns_copy(self, screener):
        screener._consecutive_failures['AAPL'] = 5
        stats = screener.get_cold_stats()
        stats['AAPL'] = 999
        assert screener._consecutive_failures['AAPL'] == 5


class TestColdPersistence:
    """Tests for cold list persistence across restarts."""

    def test_save_and_load(self, screener, tmp_path):
        stats_path = tmp_path / 'cold_stats.json'
        screener._cold_stats_path = stats_path

        # Build up failures
        screener._consecutive_failures = {'KO': 15, 'PG': 3, 'AAPL': 12}
        screener._save_cold_stats()

        assert stats_path.exists()

        # Simulate restart — new screener loading saved state
        new_screener = StockScreener(MagicMock())
        new_screener._cold_stats_path = stats_path
        new_screener._load_cold_stats()

        assert new_screener._consecutive_failures['KO'] == 15
        assert new_screener._consecutive_failures['PG'] == 3
        assert new_screener._consecutive_failures['AAPL'] == 12
        # Cold symbols should have demotion timestamps set
        assert 'KO' in new_screener._cold_demoted_at
        assert 'AAPL' in new_screener._cold_demoted_at
        assert 'PG' not in new_screener._cold_demoted_at  # Below threshold

    def test_load_missing_file(self, screener, tmp_path):
        screener._cold_stats_path = tmp_path / 'nonexistent.json'
        screener._load_cold_stats()  # Should not raise
        assert screener._consecutive_failures == {}

    def test_load_corrupted_file(self, screener, tmp_path):
        stats_path = tmp_path / 'cold_stats.json'
        stats_path.write_text('not valid json{{{')
        screener._cold_stats_path = stats_path
        screener._load_cold_stats()  # Should not raise
        assert screener._consecutive_failures == {}


class TestColdThreshold:
    """Tests for the COLD_THRESHOLD constant."""

    def test_threshold_is_twelve(self, screener):
        assert screener.COLD_THRESHOLD == 12

    def test_interval_is_one_hour(self, screener):
        assert screener.COLD_CHECK_INTERVAL_SECONDS == 3600

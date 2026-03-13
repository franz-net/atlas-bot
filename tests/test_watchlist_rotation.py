"""
Tests for the weekly watchlist rotation feature.

All tests use mocked Opus responses and in-memory SQLite.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ledger.ledger import TradingLedger
from src.screener.watchlist_rotation import WatchlistRotation


@pytest.fixture
def ledger():
    """Create an in-memory ledger."""
    return TradingLedger(db_path=':memory:')


@pytest.fixture
def mock_screener():
    """Create a mock screener with cold stats."""
    screener = MagicMock()
    screener.get_cold_stats.return_value = {
        'KO': 15,
        'PG': 18,
        'LMT': 12,
        'AAPL': 2,  # Not high failure
    }
    return screener


@pytest.fixture
def sample_watchlist(tmp_path):
    """Create a temp watchlist file."""
    data = {
        'version': '1.0',
        'updated': '2026-03-10',
        'notes': 'Test watchlist',
        'symbols': [
            {'symbol': 'AAPL', 'sector': 'Technology'},
            {'symbol': 'NVDA', 'sector': 'Technology'},
            {'symbol': 'KO', 'sector': 'Consumer Staples'},
            {'symbol': 'PG', 'sector': 'Consumer Staples'},
            {'symbol': 'LMT', 'sector': 'Industrials'},
            {'symbol': 'JPM', 'sector': 'Financials'},
        ],
    }
    wl_path = tmp_path / 'watchlist.json'
    with open(wl_path, 'w') as f:
        json.dump(data, f)
    return wl_path


@pytest.fixture
def rotation(ledger, mock_screener, sample_watchlist):
    """Create a WatchlistRotation with mocked dependencies."""
    with patch.dict('os.environ', {
        'ANTHROPIC_API_KEY': 'test-key',
        'CLAUDE_REVIEW_MODEL': 'claude-opus-4-20250514',
    }):
        with patch('src.screener.watchlist_rotation.ClaudeProvider') as mock_cls:
            mock_provider = MagicMock()
            mock_provider.review = AsyncMock()
            mock_cls.return_value = mock_provider

            rot = WatchlistRotation(ledger, mock_screener)
            rot._watchlist_path = sample_watchlist
            rot._provider = mock_provider
            return rot


class TestValidateDiff:
    """Tests for diff validation and sanitization."""

    def test_valid_diff_passes(self, rotation):
        diff = {
            'remove': ['KO', 'PG'],
            'add': [{'symbol': 'MSTR', 'sector': 'Technology'}],
            'reasoning': {'KO': 'low volume', 'MSTR': 'high ATR'},
            'notes': 'test',
        }
        current = ['AAPL', 'NVDA', 'KO', 'PG', 'LMT', 'JPM']
        result = rotation._validate_diff(diff, current, open_symbols=[])
        assert result['remove'] == ['KO', 'PG']
        assert len(result['add']) == 1
        assert result['add'][0]['symbol'] == 'MSTR'

    def test_cannot_remove_open_position(self, rotation):
        diff = {
            'remove': ['AAPL', 'KO'],
            'add': [],
        }
        current = ['AAPL', 'KO']
        result = rotation._validate_diff(diff, current, open_symbols=['AAPL'])
        assert result['remove'] == ['KO']  # AAPL filtered out

    def test_cannot_remove_nonexistent_symbol(self, rotation):
        diff = {
            'remove': ['FAKE'],
            'add': [],
        }
        result = rotation._validate_diff(diff, ['AAPL'], open_symbols=[])
        assert result['remove'] == []

    def test_cannot_add_existing_symbol(self, rotation):
        diff = {
            'remove': [],
            'add': [{'symbol': 'AAPL', 'sector': 'Technology'}],
        }
        result = rotation._validate_diff(diff, ['AAPL'], open_symbols=[])
        assert result['add'] == []

    def test_caps_at_max_removals(self, rotation):
        diff = {
            'remove': [f'SYM{i}' for i in range(20)],
            'add': [],
        }
        current = [f'SYM{i}' for i in range(20)]
        result = rotation._validate_diff(diff, current, open_symbols=[])
        assert len(result['remove']) == 10

    def test_caps_at_max_additions(self, rotation):
        diff = {
            'remove': [],
            'add': [{'symbol': f'NEW{i}', 'sector': 'Tech'} for i in range(20)],
        }
        result = rotation._validate_diff(diff, ['AAPL'], open_symbols=[])
        assert len(result['add']) == 10

    def test_string_additions_handled(self, rotation):
        """Opus might return strings instead of dicts for add."""
        diff = {
            'remove': [],
            'add': ['MSTR', 'IONQ'],
        }
        result = rotation._validate_diff(diff, ['AAPL'], open_symbols=[])
        assert len(result['add']) == 2
        assert result['add'][0]['symbol'] == 'MSTR'
        assert result['add'][0]['sector'] == 'Unknown'

    def test_empty_diff(self, rotation):
        diff = {'remove': [], 'add': []}
        result = rotation._validate_diff(diff, ['AAPL'], open_symbols=[])
        assert result['remove'] == []
        assert result['add'] == []


class TestApplyDiff:
    """Tests for applying the diff to the watchlist file."""

    def test_apply_removals(self, rotation):
        diff = {'remove': ['KO', 'PG'], 'add': []}
        assert rotation._apply_diff(diff)

        with open(rotation._watchlist_path) as f:
            data = json.load(f)
        symbols = [s['symbol'] for s in data['symbols']]
        assert 'KO' not in symbols
        assert 'PG' not in symbols
        assert 'AAPL' in symbols

    def test_apply_additions(self, rotation):
        diff = {
            'remove': [],
            'add': [{'symbol': 'MSTR', 'sector': 'Technology'}],
        }
        assert rotation._apply_diff(diff)

        with open(rotation._watchlist_path) as f:
            data = json.load(f)
        symbols = [s['symbol'] for s in data['symbols']]
        assert 'MSTR' in symbols
        assert len(symbols) == 7  # 6 original + 1

    def test_apply_both(self, rotation):
        diff = {
            'remove': ['KO'],
            'add': [{'symbol': 'MSTR', 'sector': 'Technology'}],
        }
        assert rotation._apply_diff(diff)

        with open(rotation._watchlist_path) as f:
            data = json.load(f)
        symbols = [s['symbol'] for s in data['symbols']]
        assert 'KO' not in symbols
        assert 'MSTR' in symbols
        assert len(symbols) == 6  # 6 - 1 + 1


class TestBuildRotationData:
    """Tests for building rotation data from screener and ledger."""

    def test_data_includes_watchlist(self, rotation):
        data = rotation._build_rotation_data('2026-03-14')
        assert data['watchlist_size'] == 6
        assert len(data['current_watchlist']) == 6

    def test_data_includes_cold_stats(self, rotation):
        data = rotation._build_rotation_data('2026-03-14')
        assert 'KO' in data['high_failure_symbols']
        assert 'PG' in data['high_failure_symbols']
        assert 'AAPL' not in data['high_failure_symbols']  # Only 2 failures

    def test_data_includes_sector_distribution(self, rotation):
        data = rotation._build_rotation_data('2026-03-14')
        assert 'Technology' in data['sector_distribution']
        assert data['sector_distribution']['Technology'] == 2


class TestRunRotation:
    """Tests for the full rotation pipeline."""

    @pytest.mark.asyncio
    async def test_skip_when_no_data(self, rotation):
        rotation._screener.get_cold_stats.return_value = {}
        result = await rotation.run_rotation('2026-03-14')
        assert result['skipped'] is True

    @pytest.mark.asyncio
    async def test_successful_rotation(self, rotation):
        rotation._provider.review.return_value = {
            'success': True,
            'content': json.dumps({
                'remove': ['KO'],
                'add': [{'symbol': 'MSTR', 'sector': 'Technology'}],
                'reasoning': {'KO': 'low volume', 'MSTR': 'high ATR'},
                'notes': 'Consumer staples underperforming',
            }),
            'cost_estimate': 0.03,
            'prompt_tokens': 2000,
            'completion_tokens': 200,
        }

        result = await rotation.run_rotation('2026-03-14')
        assert result['success'] is True
        assert 'KO' in result['removed']
        assert 'MSTR' in result['added']

        # Verify watchlist updated
        with open(rotation._watchlist_path) as f:
            data = json.load(f)
        symbols = [s['symbol'] for s in data['symbols']]
        assert 'KO' not in symbols
        assert 'MSTR' in symbols

    @pytest.mark.asyncio
    async def test_provider_failure(self, rotation):
        rotation._provider.review.return_value = {
            'success': False,
            'error': 'API error',
            'cost_estimate': 0,
        }
        result = await rotation.run_rotation('2026-03-14')
        assert result['success'] is False
        assert 'API error' in result['reason']

    @pytest.mark.asyncio
    async def test_malformed_json(self, rotation):
        rotation._provider.review.return_value = {
            'success': True,
            'content': 'not valid json {{{',
            'cost_estimate': 0.01,
            'prompt_tokens': 100,
            'completion_tokens': 50,
        }
        result = await rotation.run_rotation('2026-03-14')
        assert result['success'] is False
        assert 'parse' in result['reason'].lower()

    @pytest.mark.asyncio
    async def test_empty_diff_no_file_change(self, rotation):
        rotation._provider.review.return_value = {
            'success': True,
            'content': json.dumps({
                'remove': [],
                'add': [],
                'reasoning': {},
                'notes': 'No changes needed',
            }),
            'cost_estimate': 0.02,
            'prompt_tokens': 1500,
            'completion_tokens': 100,
        }
        result = await rotation.run_rotation('2026-03-14')
        assert result['success'] is True
        assert result['added'] == []
        assert result['removed'] == []


class TestLedgerRecording:
    """Tests for watchlist change persistence in the ledger."""

    def test_record_watchlist_change(self, ledger):
        success = ledger.record_watchlist_change(
            week_ending='2026-03-14',
            symbols_added='["MSTR"]',
            symbols_removed='["KO"]',
            reasoning='KO: low volume; MSTR: high ATR',
            full_response='{}',
            prompt_tokens=2000,
            completion_tokens=200,
            cost_estimate=0.03,
        )
        assert success

        changes = ledger.get_watchlist_changes()
        assert len(changes) == 1
        assert changes[0]['week_ending'] == '2026-03-14'
        assert 'MSTR' in changes[0]['symbols_added']
        assert 'KO' in changes[0]['symbols_removed']

    def test_get_watchlist_changes_ordering(self, ledger):
        ledger.record_watchlist_change(
            week_ending='2026-03-07',
            symbols_added='[]', symbols_removed='[]',
            reasoning='', full_response='{}',
        )
        ledger.record_watchlist_change(
            week_ending='2026-03-14',
            symbols_added='[]', symbols_removed='[]',
            reasoning='', full_response='{}',
        )
        changes = ledger.get_watchlist_changes()
        assert changes[0]['week_ending'] == '2026-03-14'  # Most recent first


class TestParseResponse:
    """Tests for JSON response parsing."""

    def test_parse_valid_json(self, rotation):
        content = '{"remove": ["KO"], "add": []}'
        result = rotation._parse_response(content)
        assert result['remove'] == ['KO']

    def test_parse_with_code_fences(self, rotation):
        content = '```json\n{"remove": ["KO"], "add": []}\n```'
        result = rotation._parse_response(content)
        assert result['remove'] == ['KO']

    def test_parse_empty(self, rotation):
        assert rotation._parse_response('') is None
        assert rotation._parse_response(None) is None

    def test_parse_malformed(self, rotation):
        assert rotation._parse_response('not json') is None

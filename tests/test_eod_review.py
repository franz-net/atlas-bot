"""
Sprint 7 Tests — EOD Review

Tests use mocked Claude provider and in-memory ledger — no live API calls.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.eod_review import EODReview
from src.ledger.ledger import TradingLedger


@pytest.fixture
def ledger():
    """Create an in-memory ledger."""
    l = TradingLedger(db_path=':memory:')
    yield l
    l.close()


@pytest.fixture
def sample_review_response():
    return json.dumps({
        'date': '2026-03-11',
        'summary': 'Solid day with one winning trade on NVDA.',
        'trades_reviewed': [
            {
                'symbol': 'NVDA',
                'result': 'WIN',
                'pnl': 49.20,
                'entry_quality': 'GOOD',
                'exit_quality': 'GOOD',
                'notes': 'Clean entry on volume breakout with news catalyst.',
            }
        ],
        'patterns_identified': [
            'Strong entries on high relative volume days.',
        ],
        'risk_assessment': {
            'rules_followed': True,
            'violations': [],
            'position_sizing_quality': 'GOOD',
        },
        'recommendations': [
            'Consider tighter stops on morning entries.',
        ],
        'watchlist_suggestions': {
            'add': ['AMD'],
            'remove': [],
            'reasoning': 'AMD showing similar volume patterns to NVDA.',
        },
        'overall_grade': 'A',
    })


def _populate_ledger(ledger):
    """Add sample trades to the ledger for today."""
    today = datetime.now().strftime('%Y-%m-%d')

    # Record a cycle
    ledger.record_cycle({
        'cycle_id': 'eod-test-001',
        'timestamp': f'{today}T10:00:00',
        'candidates_evaluated': 5,
        'action': 'ENTER',
        'model': 'claude-sonnet-4-20250514',
        'provider': 'claude',
        'prompt_tokens': 2000,
        'completion_tokens': 200,
        'cache_creation_input_tokens': 0,
        'cache_read_input_tokens': 0,
        'cost_estimate': 0.009,
        'raw_response': '{}',
    })

    # Record a trade entry and exit
    trade_id = ledger.record_trade_entry(
        cycle_id='eod-test-001',
        symbol='NVDA',
        direction='LONG',
        shares=2,
        entry_price=875.40,
        stop_loss_price=851.00,
        take_profit_price=910.00,
        entry_reasoning='Volume breakout with upgrade catalyst.',
        news_catalyst='NVDA upgrade by Goldman Sachs',
        entry_order_id='ENT-001',
        stop_order_id='STP-001',
        tp_order_id='LMT-001',
        phase='GROWTH',
        active_capital=1000.0,
    )

    ledger.record_trade_exit(
        trade_id=trade_id,
        exit_price=900.00,
        pnl_dollars=49.20,
        pnl_pct=2.81,
        exit_reasoning='Take profit hit.',
        active_capital=1049.20,
    )

    return trade_id


# ==================== Review Data Building Tests ====================

class TestReviewDataBuilding:
    """Test building the review data payload."""

    def test_build_review_data_empty(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        data = review._build_review_data('2026-03-11')
        assert data['week_ending'] == '2026-03-11'
        assert data['closed_trades_this_week'] == []
        assert data['entries_this_week'] == []
        assert data['cycles_this_week'] == 0

    def test_build_review_data_with_trades(self, ledger):
        _populate_ledger(ledger)
        review = EODReview.__new__(EODReview)
        review._ledger = ledger

        today = datetime.now().strftime('%Y-%m-%d')
        data = review._build_review_data(today)

        assert len(data['closed_trades_this_week']) == 1
        assert data['closed_trades_this_week'][0]['symbol'] == 'NVDA'
        assert data['closed_trades_this_week'][0]['pnl_dollars'] == 49.20
        assert data['cycles_this_week'] == 1

    def test_build_review_data_defaults_to_today(self, ledger):
        from zoneinfo import ZoneInfo
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        data = review._build_review_data()
        today_est = datetime.now(ZoneInfo('US/Eastern')).strftime('%Y-%m-%d')
        assert data['week_ending'] == today_est


# ==================== Review Parsing Tests ====================

class TestReviewParsing:
    """Test parsing Opus review responses."""

    def test_parse_valid_json(self, ledger, sample_review_response):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        parsed = review._parse_review(sample_review_response)
        assert parsed is not None
        assert parsed['overall_grade'] == 'A'
        assert len(parsed['trades_reviewed']) == 1

    def test_parse_json_with_code_fences(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        content = '```json\n{"overall_grade": "B", "summary": "OK day."}\n```'
        parsed = review._parse_review(content)
        assert parsed is not None
        assert parsed['overall_grade'] == 'B'

    def test_parse_empty_content(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        assert review._parse_review('') is None
        assert review._parse_review(None) is None

    def test_parse_malformed_json(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        assert review._parse_review('not json') is None


# ==================== Review Execution Tests ====================

class TestReviewExecution:
    """Test running the review with mocked provider."""

    @pytest.mark.asyncio
    async def test_skip_when_no_activity(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        review._provider = MagicMock()

        result = await review.run_review('2026-01-01')  # No trades on this date
        assert result['success'] is True
        assert result['skipped'] is True
        assert result['cost_estimate'] == 0

    @pytest.mark.asyncio
    async def test_successful_review(self, ledger, sample_review_response):
        _populate_ledger(ledger)
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        review._provider = MagicMock()
        review._provider.review = AsyncMock(return_value={
            'success': True,
            'content': sample_review_response,
            'prompt_tokens': 3000,
            'completion_tokens': 500,
            'cost_estimate': 0.12,
            'provider': 'claude',
            'model': 'claude-opus-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        })

        today = datetime.now().strftime('%Y-%m-%d')
        result = await review.run_review(today)

        assert result['success'] is True
        assert result['skipped'] is False
        assert result['review'] is not None
        assert result['review']['overall_grade'] == 'A'

    @pytest.mark.asyncio
    async def test_provider_failure(self, ledger):
        _populate_ledger(ledger)
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        review._provider = MagicMock()
        review._provider.review = AsyncMock(return_value={
            'success': False,
            'content': '',
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cost_estimate': 0,
            'provider': 'claude',
            'model': 'claude-opus-4-20250514',
            'error': 'API rate limit',
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        })

        today = datetime.now().strftime('%Y-%m-%d')
        result = await review.run_review(today)

        assert result['success'] is False
        assert 'rate limit' in result['reason'].lower()

    @pytest.mark.asyncio
    async def test_malformed_response(self, ledger):
        _populate_ledger(ledger)
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        review._provider = MagicMock()
        review._provider.review = AsyncMock(return_value={
            'success': True,
            'content': 'Here is my review of today...',  # Not JSON
            'prompt_tokens': 3000,
            'completion_tokens': 500,
            'cost_estimate': 0.12,
            'provider': 'claude',
            'model': 'claude-opus-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        })

        today = datetime.now().strftime('%Y-%m-%d')
        result = await review.run_review(today)

        assert result['review'] is None
        assert 'parse' in result['reason'].lower()


# ==================== Format Tests ====================

class TestReviewFormatting:
    """Test human-readable formatting."""

    def test_format_review(self, ledger, sample_review_response):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        parsed = json.loads(sample_review_response)

        output = review.format_review(parsed)
        assert 'Grade: A' in output
        assert 'NVDA' in output
        assert 'WIN' in output
        assert 'AMD' in output
        assert 'tighter stops' in output

    def test_format_none(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        assert 'No review data' in review.format_review(None)

    def test_format_empty_review(self, ledger):
        review = EODReview.__new__(EODReview)
        review._ledger = ledger
        output = review.format_review({})
        assert 'No review data' in output


# ==================== DB Persistence Tests ====================

class TestReviewPersistence:
    """Test that reviews are stored in the database."""

    def test_record_review(self, ledger):
        success = ledger.record_review(
            week_ending='2026-03-14',
            overall_grade='B',
            summary='Decent week with 2 wins and 1 loss.',
            review_json='{"overall_grade": "B"}',
            prompt_tokens=3000,
            completion_tokens=500,
            cost_estimate=0.019,
        )
        assert success is True

        reviews = ledger.get_reviews()
        assert len(reviews) == 1
        assert reviews[0]['overall_grade'] == 'B'
        assert reviews[0]['week_ending'] == '2026-03-14'
        assert reviews[0]['cost_estimate'] == 0.019

    def test_get_reviews_ordering(self, ledger):
        ledger.record_review(
            week_ending='2026-03-07', overall_grade='C',
            summary='First week.', review_json='{}',
        )
        ledger.record_review(
            week_ending='2026-03-14', overall_grade='B',
            summary='Second week.', review_json='{}',
        )

        reviews = ledger.get_reviews()
        assert len(reviews) == 2
        # Most recent first
        assert reviews[0]['week_ending'] == '2026-03-14'
        assert reviews[1]['week_ending'] == '2026-03-07'

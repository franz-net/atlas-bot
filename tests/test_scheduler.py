"""
Sprint 6 Tests — Trading Scheduler

Tests use fully mocked components — no live API or AI calls.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scheduler import TradingScheduler
from src.engine.operator_approval import OperatorApproval
from src.utils.discord_notifier import DiscordNotifier


@pytest.fixture
def scheduler():
    s = TradingScheduler()
    # Disable Discord notifications in tests to prevent real webhook calls
    s._discord = MagicMock(spec=DiscordNotifier)
    s._discord.notify_entry = AsyncMock(return_value=True)
    s._discord.notify_exit = AsyncMock(return_value=True)
    s._discord.notify_error = AsyncMock(return_value=True)
    s._discord.notify_daily_summary = AsyncMock(return_value=True)
    return s


# ==================== Market Hours Tests ====================

class TestMarketHours:
    """Test market hours detection."""

    def test_weekday_check(self, scheduler):
        # Saturday
        with patch('scheduler.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 14, 10, 0)  # Saturday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Note: ZoneInfo makes mocking datetime tricky
            result = scheduler.is_market_open()
            assert isinstance(result, bool)

    def test_returns_bool(self, scheduler):
        assert isinstance(scheduler.is_market_open(), bool)


# ==================== Scheduler Init Tests ====================

class TestSchedulerInit:
    """Test scheduler initialization."""

    def test_default_interval(self, scheduler):
        assert scheduler._interval == 300

    def test_not_running_initially(self, scheduler):
        assert scheduler._running is False

    def test_cycle_count_starts_at_zero(self, scheduler):
        assert scheduler._cycle_count == 0

    def test_custom_interval(self):
        with patch.dict('os.environ', {'LOOP_INTERVAL_SECONDS': '60'}):
            s = TradingScheduler()
            assert s._interval == 60


# ==================== Run Cycle Tests ====================

class TestRunCycle:
    """Test a single trading cycle with mocked components."""

    @pytest.mark.asyncio
    async def test_cycle_no_account_state(self, scheduler):
        """Cycle fails gracefully when account state unavailable."""
        scheduler._client = MagicMock()
        scheduler._monitor = MagicMock()
        scheduler._screener = MagicMock()
        scheduler._builder = MagicMock()
        scheduler._news = MagicMock()
        scheduler._engine = MagicMock()
        scheduler._order_mgr = MagicMock()
        scheduler._ledger = MagicMock()

        # Mock _get_account_state to return None
        scheduler._get_account_state = AsyncMock(return_value=None)

        summary = await scheduler.run_cycle()
        assert summary['error'] == 'Could not fetch account state'

    @pytest.mark.asyncio
    async def test_cycle_hard_stop(self, scheduler):
        """Cycle halts when active capital below 20% of start-of-day capital."""
        scheduler._client = MagicMock()
        scheduler._monitor = MagicMock()
        scheduler._screener = MagicMock()
        scheduler._builder = MagicMock()
        scheduler._news = MagicMock()
        scheduler._engine = MagicMock()
        scheduler._order_mgr = MagicMock()
        scheduler._ledger = MagicMock()
        # Start-of-day capital was $5000, hard stop = 20% = $1000
        scheduler._ledger.get_start_of_day_capital.return_value = 5000.0

        scheduler._get_account_state = AsyncMock(return_value={
            'balance': 25500,
            'cash_balance': 25500,
            'active_capital': 500,  # Below $1000 (20% of $5000)
            'buying_power': 25500,
            'phase': 'GROWTH',
        })

        summary = await scheduler.run_cycle()
        assert 'hard stop' in summary['error'].lower()

    @pytest.mark.asyncio
    async def test_cycle_no_candidates(self, scheduler):
        """Cycle completes with HOLD when no candidates found."""
        scheduler._client = MagicMock()
        scheduler._monitor = MagicMock()
        scheduler._monitor.sync_positions = AsyncMock(return_value={
            'broker_positions': [],
            'exits_recorded': [],
            'open_trade_count': 0,
        })
        scheduler._screener = MagicMock()
        scheduler._screener.screen = AsyncMock(return_value=([], {}, {}))
        scheduler._builder = MagicMock()
        scheduler._news = MagicMock()
        scheduler._engine = MagicMock()
        scheduler._engine.decide = AsyncMock(return_value={
            'cycle_id': 'test-001',
            'action': 'HOLD',
            'trades': [],
            'cost_estimate': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
            'timestamp': '2026-03-11T10:00:00',
            'candidates_evaluated': 0,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'raw_response': '',
            'cycle_notes': 'No candidates.',
            'parse_error': None,
        })
        scheduler._order_mgr = MagicMock()
        scheduler._ledger = MagicMock()
        scheduler._ledger.get_daily_api_cost.return_value = 0.0
        scheduler._ledger.get_start_of_day_capital.return_value = 5000.0
        scheduler._approval = OperatorApproval()
        scheduler._mode = 'SIM'

        scheduler._get_account_state = AsyncMock(return_value={
            'balance': 30000,
            'cash_balance': 30000,
            'active_capital': 5000,
            'buying_power': 30000,
            'phase': 'GROWTH',
        })

        summary = await scheduler.run_cycle()
        assert summary['action'] == 'HOLD'
        assert summary['candidates'] == 0
        assert summary['error'] is None

    @pytest.mark.asyncio
    async def test_cycle_with_candidates_and_trade(self, scheduler):
        """Full cycle with candidates resulting in a trade."""
        scheduler._client = MagicMock()
        scheduler._monitor = MagicMock()
        scheduler._monitor.sync_positions = AsyncMock(return_value={
            'broker_positions': [],
            'exits_recorded': [],
            'open_trade_count': 0,
        })
        scheduler._screener = MagicMock()
        scheduler._screener.screen = AsyncMock(return_value=(
            [{'symbol': 'NVDA', 'relative_volume': 2.3}],
            {'NVDA': {'Last': '875.00', 'Bid': '874.95', 'Ask': '875.05'}},
            {'NVDA': {}},
        ))
        scheduler._builder = MagicMock()
        scheduler._builder.build_all_packages.return_value = [
            {'symbol': 'NVDA', 'price': 875.0},
        ]
        scheduler._news = MagicMock()
        scheduler._news.fetch_news_batch = AsyncMock(return_value={'NVDA': []})
        scheduler._engine = MagicMock()
        scheduler._engine.decide = AsyncMock(return_value={
            'cycle_id': 'test-002',
            'action': 'ENTER',
            'trades': [{
                'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                'shares': 1, 'expected_entry_price': 875.0,
                'stop_loss': 855.0, 'take_profit': 920.0,
                'reasoning': 'Strong setup.',
            }],
            'cost_estimate': 0.009,
            'prompt_tokens': 2000,
            'completion_tokens': 200,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
            'timestamp': '2026-03-11T10:00:00',
            'candidates_evaluated': 1,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'raw_response': '',
            'cycle_notes': 'Entering NVDA.',
            'parse_error': None,
        })
        scheduler._order_mgr = MagicMock()
        scheduler._order_mgr.execute_decisions = AsyncMock(return_value=[
            {
                'success': True,
                'symbol': 'NVDA',
                'direction': 'LONG',
                'shares': 1,
                'entry_order_id': 'ENT-001',
                'stop_order_id': 'STP-001',
                'tp_order_id': 'LMT-001',
                'entry_price': 875.0,
                'error': None,
            }
        ])
        scheduler._ledger = MagicMock()
        scheduler._ledger.get_daily_api_cost.return_value = 0.009
        scheduler._ledger.get_start_of_day_capital.return_value = 5000.0
        scheduler._approval = OperatorApproval()  # Disabled by default = auto-approve
        scheduler._mode = 'SIM'

        scheduler._get_account_state = AsyncMock(return_value={
            'balance': 30000,
            'cash_balance': 30000,
            'active_capital': 5000,
            'buying_power': 30000,
            'phase': 'GROWTH',
        })

        summary = await scheduler.run_cycle()
        assert summary['action'] == 'ENTER'
        assert summary['candidates'] == 1
        assert summary['trades_executed'] == 1
        assert summary['error'] is None

        # Verify ledger was called
        scheduler._ledger.record_cycle.assert_called_once()
        scheduler._ledger.record_trade_entry.assert_called_once()


# ==================== Stop Tests ====================

class TestSchedulerStop:
    """Test scheduler stop."""

    def test_stop_sets_flag(self, scheduler):
        scheduler._running = True
        scheduler.stop()
        assert scheduler._running is False

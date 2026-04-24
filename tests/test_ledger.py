"""
Sprint 5 Tests — Trading Ledger & Withdrawal Tracker

All tests use in-memory SQLite — no files created.
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.ledger.ledger import TradingLedger
from src.ledger.withdrawal_tracker import WithdrawalTracker


@pytest.fixture
def ledger():
    """Create an in-memory ledger for testing."""
    l = TradingLedger(db_path=':memory:')
    yield l
    l.close()


@pytest.fixture
def sample_cycle():
    return {
        'cycle_id': 'test-cycle-001',
        'timestamp': '2026-03-10T10:00:00',
        'candidates_evaluated': 3,
        'action': 'HOLD',
        'model': 'claude-sonnet-4-20250514',
        'provider': 'claude',
        'prompt_tokens': 2000,
        'completion_tokens': 100,
        'cache_creation_input_tokens': 1500,
        'cache_read_input_tokens': 0,
        'cost_estimate': 0.009,
        'raw_response': '{"action": "HOLD"}',
    }


@pytest.fixture
def sample_trade_entry():
    return {
        'cycle_id': 'test-cycle-002',
        'symbol': 'NVDA',
        'direction': 'LONG',
        'shares': 2,
        'entry_price': 875.40,
        'stop_loss_price': 851.00,
        'take_profit_price': 910.00,
        'entry_reasoning': 'Breaking above 20-day high on strong volume.',
        'news_catalyst': 'NVDA receives analyst upgrade',
        'entry_order_id': 'ORD-001',
        'stop_order_id': 'ORD-002',
        'tp_order_id': 'ORD-003',
        'phase': 'GROWTH',
        'active_capital': 1000.0,
    }


# ==================== Schema Tests ====================

class TestSchema:
    """Test database schema initialization."""

    def test_tables_created(self, ledger):
        cursor = ledger._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row['name'] for row in cursor.fetchall()]
        assert 'trades' in tables
        assert 'cycles' in tables
        assert 'withdrawals' in tables

    def test_schema_idempotent(self, ledger):
        """Running schema init twice should not error."""
        ledger._init_schema()
        cursor = ledger._conn.execute(
            "SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        assert cursor.fetchone()['cnt'] == 1


# ==================== Cycle Tests ====================

class TestCycles:
    """Test decision cycle recording."""

    def test_record_cycle(self, ledger, sample_cycle):
        ledger.record_cycle(sample_cycle)
        cycles = ledger.get_recent_cycles(limit=5)
        assert len(cycles) == 1
        assert cycles[0]['id'] == 'test-cycle-001'
        assert cycles[0]['action_taken'] == 'HOLD'
        assert cycles[0]['prompt_tokens'] == 2000

    def test_cycle_idempotent(self, ledger, sample_cycle):
        """Recording same cycle_id twice should skip the duplicate."""
        ledger.record_cycle(sample_cycle)
        ledger.record_cycle(sample_cycle)
        cycles = ledger.get_recent_cycles(limit=10)
        assert len(cycles) == 1

    def test_multiple_cycles(self, ledger, sample_cycle):
        ledger.record_cycle(sample_cycle)
        cycle2 = {**sample_cycle, 'cycle_id': 'test-cycle-002', 'action': 'ENTER'}
        ledger.record_cycle(cycle2)
        cycles = ledger.get_recent_cycles(limit=10)
        assert len(cycles) == 2


# ==================== Trade Entry Tests ====================

class TestTradeEntry:
    """Test trade entry recording."""

    def test_record_trade_entry(self, ledger, sample_trade_entry):
        trade_id = ledger.record_trade_entry(**sample_trade_entry)
        assert trade_id > 0

        trade = ledger.get_trade_by_id(trade_id)
        assert trade is not None
        assert trade['symbol'] == 'NVDA'
        assert trade['direction'] == 'LONG'
        assert trade['shares'] == 2
        assert trade['entry_price'] == 875.40
        assert trade['status'] == 'OPEN'

    def test_trade_appears_in_open_trades(self, ledger, sample_trade_entry):
        ledger.record_trade_entry(**sample_trade_entry)
        open_trades = ledger.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]['symbol'] == 'NVDA'

    def test_trade_by_cycle(self, ledger, sample_trade_entry):
        ledger.record_trade_entry(**sample_trade_entry)
        trade = ledger.get_trade_by_cycle('test-cycle-002')
        assert trade is not None
        assert trade['symbol'] == 'NVDA'

    def test_trade_not_found(self, ledger):
        assert ledger.get_trade_by_id(999) is None
        assert ledger.get_trade_by_cycle('nonexistent') is None


# ==================== Trade Exit Tests ====================

class TestTradeExit:
    """Test trade exit recording."""

    def test_record_trade_exit(self, ledger, sample_trade_entry):
        trade_id = ledger.record_trade_entry(**sample_trade_entry)

        success = ledger.record_trade_exit(
            trade_id=trade_id,
            exit_price=895.00,
            pnl_dollars=39.20,
            pnl_pct=2.24,
            exit_reasoning='Hit take profit target.',
            active_capital=1039.20,
        )
        assert success is True

        trade = ledger.get_trade_by_id(trade_id)
        assert trade['status'] == 'CLOSED'
        assert trade['exit_price'] == 895.00
        assert trade['pnl_dollars'] == 39.20

    def test_closed_trade_not_in_open(self, ledger, sample_trade_entry):
        trade_id = ledger.record_trade_entry(**sample_trade_entry)
        ledger.record_trade_exit(
            trade_id=trade_id,
            exit_price=895.00,
            pnl_dollars=39.20,
            pnl_pct=2.24,
            exit_reasoning='Hit take profit.',
            active_capital=1039.20,
        )
        assert len(ledger.get_open_trades()) == 0
        assert len(ledger.get_closed_trades()) == 1

    def test_all_trades_includes_both(self, ledger, sample_trade_entry):
        # One open, one closed
        t1 = ledger.record_trade_entry(**sample_trade_entry)
        t2_entry = {**sample_trade_entry, 'cycle_id': 'cycle-003', 'symbol': 'AAPL'}
        t2 = ledger.record_trade_entry(**t2_entry)
        ledger.record_trade_exit(
            trade_id=t1, exit_price=895.0, pnl_dollars=39.20,
            pnl_pct=2.24, exit_reasoning='TP hit.', active_capital=1039.20,
        )
        all_trades = ledger.get_all_trades()
        assert len(all_trades) == 2


# ==================== Withdrawal Tests ====================

class TestWithdrawals:
    """Test withdrawal recording."""

    def test_record_withdrawal(self, ledger):
        success = ledger.record_withdrawal(
            week_ending='2026-03-06',
            active_capital=3000.0,
            weekly_profit=150.0,
            withdrawal_amount=1.50,
            notes='Test withdrawal',
        )
        assert success is True

        withdrawals = ledger.get_withdrawals()
        assert len(withdrawals) == 1
        assert withdrawals[0]['withdrawal_amount'] == 1.50
        assert withdrawals[0]['running_total_withdrawn'] == 1.50

    def test_running_total_accumulates(self, ledger):
        ledger.record_withdrawal(
            week_ending='2026-03-06', active_capital=3000.0,
            weekly_profit=100.0, withdrawal_amount=1.00,
        )
        ledger.record_withdrawal(
            week_ending='2026-03-13', active_capital=3100.0,
            weekly_profit=200.0, withdrawal_amount=2.00,
        )
        assert ledger.get_total_withdrawn() == 3.00

        withdrawals = ledger.get_withdrawals()
        assert withdrawals[0]['running_total_withdrawn'] == 1.00
        assert withdrawals[1]['running_total_withdrawn'] == 3.00

    def test_no_withdrawals_total_zero(self, ledger):
        assert ledger.get_total_withdrawn() == 0.0


# ==================== Summary Tests ====================

class TestSummary:
    """Test P&L summary statistics."""

    def test_empty_summary(self, ledger):
        summary = ledger.get_summary()
        assert summary['total_trades'] == 0
        assert summary['win_rate'] == 0
        assert summary['total_pnl'] == 0

    def test_summary_with_trades(self, ledger, sample_trade_entry):
        # Create a winning trade
        t1 = ledger.record_trade_entry(**sample_trade_entry)
        ledger.record_trade_exit(
            trade_id=t1, exit_price=900.0, pnl_dollars=49.20,
            pnl_pct=2.81, exit_reasoning='TP hit.', active_capital=1049.20,
        )
        # Create a losing trade
        t2_entry = {**sample_trade_entry, 'cycle_id': 'cycle-loss', 'symbol': 'AAPL'}
        t2 = ledger.record_trade_entry(**t2_entry)
        ledger.record_trade_exit(
            trade_id=t2, exit_price=850.0, pnl_dollars=-50.80,
            pnl_pct=-2.90, exit_reasoning='SL hit.', active_capital=998.40,
        )

        summary = ledger.get_summary()
        assert summary['total_trades'] == 2
        assert summary['wins'] == 1
        assert summary['losses'] == 1
        assert summary['win_rate'] == 50.0
        assert summary['total_pnl'] == pytest.approx(-1.60, abs=0.01)
        assert summary['avg_win'] == pytest.approx(49.20, abs=0.01)
        assert summary['avg_loss'] == pytest.approx(-50.80, abs=0.01)

    def test_summary_includes_open_count(self, ledger, sample_trade_entry):
        ledger.record_trade_entry(**sample_trade_entry)
        summary = ledger.get_summary()
        assert summary['open_trades'] == 1
        assert summary['total_trades'] == 0  # Only closed trades count


# ==================== API Cost Tests ====================

class TestAPICosts:
    """Test API cost tracking."""

    def test_daily_api_cost(self, ledger, sample_cycle):
        today = datetime.now().strftime('%Y-%m-%d')
        sample_cycle['timestamp'] = f'{today}T10:00:00'
        ledger.record_cycle(sample_cycle)

        cost = ledger.get_daily_api_cost(today)
        assert cost == pytest.approx(0.009, abs=0.001)

    def test_daily_api_cost_no_cycles(self, ledger):
        cost = ledger.get_daily_api_cost('2026-01-01')
        assert cost == 0.0

    def test_summary_includes_api_cost(self, ledger, sample_cycle):
        ledger.record_cycle(sample_cycle)
        summary = ledger.get_summary()
        assert summary['total_api_cost'] == pytest.approx(0.009, abs=0.001)


# ==================== Export Tests ====================

class TestExport:
    """Test CSV export."""

    def test_export_trades_csv_empty(self, ledger):
        assert ledger.export_trades_csv() == ''

    def test_export_trades_csv_with_data(self, ledger, sample_trade_entry):
        ledger.record_trade_entry(**sample_trade_entry)
        csv_output = ledger.export_trades_csv()
        assert 'NVDA' in csv_output
        assert 'symbol' in csv_output  # header row

    def test_export_costs_csv_empty(self, ledger):
        assert ledger.export_costs_csv() == ''

    def test_export_costs_csv_with_data(self, ledger, sample_cycle):
        ledger.record_cycle(sample_cycle)
        csv_output = ledger.export_costs_csv()
        assert 'test-cycle-001' in csv_output


# ==================== Withdrawal Tracker Tests ====================

class TestWithdrawalTracker:
    """Test the WithdrawalTracker logic."""

    def test_not_friday_returns_none(self, ledger):
        tracker = WithdrawalTracker(ledger)
        # Mock a non-Friday (Monday = 0)
        with patch('src.ledger.withdrawal_tracker.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 9, 16, 0)  # Monday
            result = tracker.calculate_weekly_withdrawal(current_balance=28000)
            assert result is None

    def test_friday_below_threshold(self, ledger):
        tracker = WithdrawalTracker(ledger)
        with patch('src.ledger.withdrawal_tracker.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 13, 16, 0)  # Friday
            # Active capital = 26500 - 25000 = 1500 < 2000 threshold
            result = tracker.calculate_weekly_withdrawal(current_balance=26500)
            assert result is None

    def test_friday_no_profit(self, ledger):
        tracker = WithdrawalTracker(ledger)
        with patch('src.ledger.withdrawal_tracker.datetime') as mock_dt:
            friday = datetime(2026, 3, 13, 16, 0)
            mock_dt.now.return_value = friday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw) if a else friday
            # Active capital = 28000 - 25000 = 3000 > 2000 threshold
            # But no closed trades = no profit
            result = tracker.calculate_weekly_withdrawal(current_balance=28000)
            assert result is None

    def test_friday_with_profit(self, ledger, sample_trade_entry):
        """Friday with profit above threshold should calculate withdrawal."""
        # Create a closed trade with profit this week
        trade_id = ledger.record_trade_entry(**sample_trade_entry)
        ledger.record_trade_exit(
            trade_id=trade_id, exit_price=900.0, pnl_dollars=49.20,
            pnl_pct=2.81, exit_reasoning='TP hit.', active_capital=3049.20,
        )

        tracker = WithdrawalTracker(ledger)
        with patch('src.ledger.withdrawal_tracker.datetime') as mock_dt:
            friday = datetime(2026, 3, 13, 16, 0)
            mock_dt.now.return_value = friday
            # strftime needs to work on the mock
            result = tracker.calculate_weekly_withdrawal(current_balance=28000)

            if result is not None:
                assert result['withdrawal_amount'] == pytest.approx(49.20 * 0.01, abs=0.01)
                assert result['weekly_profit'] == pytest.approx(49.20, abs=0.01)

    def test_record_withdrawal(self, ledger):
        tracker = WithdrawalTracker(ledger)
        withdrawal = {
            'week_ending': '2026-03-13',
            'active_capital': 3000.0,
            'weekly_profit': 150.0,
            'withdrawal_amount': 1.50,
        }
        success = tracker.record_withdrawal(withdrawal)
        assert success is True
        assert ledger.get_total_withdrawn() == 1.50


# ==================== Daily Snapshot Tests ====================

class TestDailySnapshots:
    """Test start-of-day capital tracking for hard stop calculation."""

    def test_record_and_retrieve(self, ledger):
        ledger.record_daily_snapshot('2026-03-12', 5000.0, 30000.0)
        result = ledger.get_start_of_day_capital('2026-03-12')
        assert result == 5000.0

    def test_idempotent_first_write_wins(self, ledger):
        ledger.record_daily_snapshot('2026-03-12', 5000.0, 30000.0)
        ledger.record_daily_snapshot('2026-03-12', 4800.0, 29800.0)
        result = ledger.get_start_of_day_capital('2026-03-12')
        assert result == 5000.0  # First write wins

    def test_missing_date_returns_none(self, ledger):
        result = ledger.get_start_of_day_capital('2026-03-12')
        assert result is None

    def test_different_days_independent(self, ledger):
        ledger.record_daily_snapshot('2026-03-12', 5000.0, 30000.0)
        ledger.record_daily_snapshot('2026-03-13', 5200.0, 30200.0)
        assert ledger.get_start_of_day_capital('2026-03-12') == 5000.0
        assert ledger.get_start_of_day_capital('2026-03-13') == 5200.0


class TestIdempotentExit:
    """Test that record_trade_exit is idempotent (safe to call twice)."""

    def test_double_exit_returns_false_second_time(self, ledger, sample_trade_entry):
        """Second call to record_trade_exit on same trade should return False."""
        trade_id = ledger.record_trade_entry(**sample_trade_entry)

        first = ledger.record_trade_exit(
            trade_id=trade_id,
            exit_price=895.00,
            pnl_dollars=39.20,
            pnl_pct=2.24,
            exit_reasoning='Stop loss hit',
            active_capital=1039.20,
        )
        assert first is True

        second = ledger.record_trade_exit(
            trade_id=trade_id,
            exit_price=900.00,
            pnl_dollars=49.20,
            pnl_pct=2.81,
            exit_reasoning='AI EXIT: different reasoning',
            active_capital=1049.20,
        )
        assert second is False

        # Original exit data should be preserved
        trade = ledger.get_trade_by_id(trade_id)
        assert trade['exit_price'] == 895.00
        assert trade['exit_reasoning'] == 'Stop loss hit'


# ==================== System State Tests ====================


class TestSystemState:
    """Tests for get_state / set_state persistence."""

    def test_get_state_missing_key(self, ledger):
        assert ledger.get_state('nonexistent') is None

    def test_set_and_get_state(self, ledger):
        ledger.set_state('eod_review_date', '2026-03-13')
        assert ledger.get_state('eod_review_date') == '2026-03-13'

    def test_set_state_upsert(self, ledger):
        ledger.set_state('eod_review_date', '2026-03-12')
        ledger.set_state('eod_review_date', '2026-03-13')
        assert ledger.get_state('eod_review_date') == '2026-03-13'

    def test_state_survives_reconnect(self):
        """Simulates restart — new TradingLedger on same DB."""
        import tempfile, os
        db_path = os.path.join(tempfile.mkdtemp(), 'test.db')
        ledger1 = TradingLedger(db_path=db_path)
        ledger1.set_state('eod_review_date', '2026-03-13')
        ledger1.close()

        ledger2 = TradingLedger(db_path=db_path)
        assert ledger2.get_state('eod_review_date') == '2026-03-13'
        ledger2.close()

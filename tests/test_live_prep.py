"""
Sprint 8 Tests — Live Prep (Preflight, Operator Approval, Mode Tracking)

Tests use in-memory ledger and mocked inputs — no live API calls.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.engine.operator_approval import OperatorApproval
from src.engine.preflight import PreflightCheck
from src.ledger.ledger import TradingLedger


@pytest.fixture
def ledger():
    """Create an in-memory ledger."""
    l = TradingLedger(db_path=':memory:')
    yield l
    l.close()


@pytest.fixture
def sample_trade():
    return {
        'action': 'ENTER',
        'symbol': 'NVDA',
        'direction': 'LONG',
        'shares': 2,
        'stop_loss': 851.00,
        'take_profit': 910.00,
        'reasoning': 'Strong volume breakout with catalyst.',
    }


# ==================== Mode Column Tests ====================

class TestModeColumn:
    """Test SIM/LIVE mode tracking in ledger."""

    def test_trade_defaults_to_sim(self, ledger):
        trade_id = ledger.record_trade_entry(
            cycle_id='mode-test-001', symbol='NVDA', direction='LONG',
            shares=1, entry_price=100.0, stop_loss_price=95.0,
            take_profit_price=110.0, entry_reasoning='Test.',
            news_catalyst=None, entry_order_id='O1', stop_order_id='O2',
            tp_order_id='O3', phase='GROWTH', active_capital=1000.0,
        )
        trade = ledger.get_trade_by_id(trade_id)
        assert trade['mode'] == 'SIM'

    def test_trade_with_live_mode(self, ledger):
        trade_id = ledger.record_trade_entry(
            cycle_id='mode-test-002', symbol='AAPL', direction='LONG',
            shares=1, entry_price=150.0, stop_loss_price=145.0,
            take_profit_price=160.0, entry_reasoning='Test.',
            news_catalyst=None, entry_order_id='O1', stop_order_id='O2',
            tp_order_id='O3', phase='GROWTH', active_capital=1000.0,
            mode='LIVE',
        )
        trade = ledger.get_trade_by_id(trade_id)
        assert trade['mode'] == 'LIVE'

    def test_cycle_defaults_to_sim(self, ledger):
        ledger.record_cycle({
            'cycle_id': 'mode-cycle-001',
            'timestamp': '2026-03-11T10:00:00',
            'candidates_evaluated': 1,
            'action': 'HOLD',
            'model': 'test', 'provider': 'test',
            'prompt_tokens': 0, 'completion_tokens': 0,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
            'cost_estimate': 0, 'raw_response': '',
        })
        cycles = ledger.get_recent_cycles(limit=1)
        assert cycles[0]['mode'] == 'SIM'

    def test_cycle_with_live_mode(self, ledger):
        ledger.record_cycle({
            'cycle_id': 'mode-cycle-002',
            'timestamp': '2026-03-11T10:00:00',
            'candidates_evaluated': 1,
            'action': 'HOLD',
            'model': 'test', 'provider': 'test',
            'prompt_tokens': 0, 'completion_tokens': 0,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
            'cost_estimate': 0, 'raw_response': '',
            'mode': 'LIVE',
        })
        cycles = ledger.get_recent_cycles(limit=1)
        assert cycles[0]['mode'] == 'LIVE'

    def test_get_trades_by_mode(self, ledger):
        # Create SIM and LIVE trades
        t1 = ledger.record_trade_entry(
            cycle_id='sim-t', symbol='NVDA', direction='LONG',
            shares=1, entry_price=100.0, stop_loss_price=95.0,
            take_profit_price=110.0, entry_reasoning='Test.',
            news_catalyst=None, entry_order_id='O1', stop_order_id='O2',
            tp_order_id='O3', phase='GROWTH', active_capital=1000.0,
            mode='SIM',
        )
        t2 = ledger.record_trade_entry(
            cycle_id='live-t', symbol='AAPL', direction='LONG',
            shares=1, entry_price=150.0, stop_loss_price=145.0,
            take_profit_price=160.0, entry_reasoning='Test.',
            news_catalyst=None, entry_order_id='O4', stop_order_id='O5',
            tp_order_id='O6', phase='GROWTH', active_capital=1000.0,
            mode='LIVE',
        )
        # Close both
        ledger.record_trade_exit(t1, 105.0, 5.0, 5.0, 'TP.', 1005.0)
        ledger.record_trade_exit(t2, 155.0, 5.0, 3.3, 'TP.', 1005.0)

        sim_trades = ledger.get_trades_by_mode('SIM')
        live_trades = ledger.get_trades_by_mode('LIVE')
        assert len(sim_trades) == 1
        assert sim_trades[0]['symbol'] == 'NVDA'
        assert len(live_trades) == 1
        assert live_trades[0]['symbol'] == 'AAPL'

    def test_summary_filtered_by_mode(self, ledger):
        t1 = ledger.record_trade_entry(
            cycle_id='sum-sim', symbol='NVDA', direction='LONG',
            shares=1, entry_price=100.0, stop_loss_price=95.0,
            take_profit_price=110.0, entry_reasoning='Test.',
            news_catalyst=None, entry_order_id='O1', stop_order_id='O2',
            tp_order_id='O3', phase='GROWTH', active_capital=1000.0,
            mode='SIM',
        )
        ledger.record_trade_exit(t1, 105.0, 5.0, 5.0, 'TP.', 1005.0)

        sim_summary = ledger.get_summary(mode='SIM')
        live_summary = ledger.get_summary(mode='LIVE')
        assert sim_summary['total_trades'] == 1
        assert live_summary['total_trades'] == 0


# ==================== Preflight Tests ====================

class TestPreflight:
    """Test pre-flight startup checks."""

    def test_startup_checks_pass(self, ledger, monkeypatch):
        monkeypatch.setenv('TS_API_KEY', 'test')
        monkeypatch.setenv('TS_API_SECRET', 'test')
        monkeypatch.setenv('TS_ACCOUNT_ID', 'test')
        monkeypatch.setenv('DECISION_PROVIDER', 'gemini')
        monkeypatch.setenv('GEMINI_API_KEY', 'test')
        monkeypatch.setenv('GEMINI_DECISION_MODEL', 'gemini-2.5-flash')
        monkeypatch.setenv('USE_SIM_ACCOUNT', 'true')

        preflight = PreflightCheck(ledger)
        result = preflight.run_startup_checks()
        assert result['passed'] is True
        assert result['sim_mode'] is True

    def test_startup_checks_missing_key(self, ledger, monkeypatch):
        monkeypatch.delenv('TS_API_KEY', raising=False)
        monkeypatch.setenv('TS_API_SECRET', 'test')
        monkeypatch.setenv('TS_ACCOUNT_ID', 'test')
        monkeypatch.setenv('DECISION_PROVIDER', 'claude')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
        monkeypatch.setenv('CLAUDE_DECISION_MODEL', 'test')

        preflight = PreflightCheck(ledger)
        result = preflight.run_startup_checks()
        assert result['passed'] is False

    def test_live_mode_warning(self, ledger, monkeypatch):
        monkeypatch.setenv('TS_API_KEY', 'test')
        monkeypatch.setenv('TS_API_SECRET', 'test')
        monkeypatch.setenv('TS_ACCOUNT_ID', 'test')
        monkeypatch.setenv('DECISION_PROVIDER', 'gemini')
        monkeypatch.setenv('GEMINI_API_KEY', 'test')
        monkeypatch.setenv('GEMINI_DECISION_MODEL', 'test')
        monkeypatch.setenv('USE_SIM_ACCOUNT', 'false')

        preflight = PreflightCheck(ledger)
        result = preflight.run_startup_checks()
        assert len(result['warnings']) > 0
        assert 'LIVE' in result['warnings'][0]


# ==================== Sim Validation Tests ====================

class TestSimValidation:
    """Test sim validation criteria."""

    def test_empty_ledger_not_ready(self, ledger):
        preflight = PreflightCheck(ledger)
        report = preflight.run_sim_validation()
        assert report['ready_for_live'] is False

    def test_validation_with_trades(self, ledger):
        # Create enough trades to test validation logic
        for i in range(5):
            tid = ledger.record_trade_entry(
                cycle_id=f'val-{i}', symbol='NVDA', direction='LONG',
                shares=1, entry_price=100.0, stop_loss_price=95.0,
                take_profit_price=110.0, entry_reasoning='Test reasoning here.',
                news_catalyst=None, entry_order_id=f'O{i}', stop_order_id=f'S{i}',
                tp_order_id=f'T{i}', phase='GROWTH', active_capital=1000.0,
            )
            ledger.record_trade_exit(
                tid, 105.0, 5.0, 5.0, 'Take profit hit.', 1005.0,
            )

        preflight = PreflightCheck(ledger)
        report = preflight.run_sim_validation()
        # Still not ready — need 30 trading days
        assert report['ready_for_live'] is False
        # But P&L and win rate should pass
        pnl_criterion = next(c for c in report['criteria'] if 'P&L' in c['name'])
        assert pnl_criterion['passed'] is True

    def test_drawdown_calculation(self, ledger):
        preflight = PreflightCheck(ledger)
        trades = [
            {
                'exit_timestamp': '2026-03-10T15:00:00',
                'pnl_dollars': -100.0,
                'active_capital_at_entry': 1000.0,
            },
        ]
        drawdown = preflight._calculate_max_daily_drawdown(trades)
        assert drawdown == pytest.approx(10.0, abs=0.1)

    def test_incomplete_trades_count(self, ledger):
        preflight = PreflightCheck(ledger)
        trades = [
            {'entry_reasoning': 'Good.', 'exit_reasoning': 'TP.', 'pnl_dollars': 5.0},
            {'entry_reasoning': '', 'exit_reasoning': 'TP.', 'pnl_dollars': 5.0},  # Missing
            {'entry_reasoning': 'Good.', 'exit_reasoning': None, 'pnl_dollars': 5.0},  # Missing
        ]
        assert preflight._count_incomplete_trades(trades) == 2


# ==================== Operator Approval Tests ====================

class TestOperatorApproval:
    """Test operator approval layer."""

    def test_disabled_by_default(self):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'false'}):
            approval = OperatorApproval()
            assert approval.is_enabled is False

    def test_auto_approves_when_disabled(self, sample_trade):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'false'}):
            approval = OperatorApproval()
            assert approval.request_approval(sample_trade, 1000.0) is True

    def test_enabled_requires_input(self, sample_trade):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'true'}):
            approval = OperatorApproval()
            assert approval.is_enabled is True
            # Simulate user typing 'y'
            with patch('builtins.input', return_value='y'):
                assert approval.request_approval(sample_trade, 1000.0) is True
            assert approval.stats['approved'] == 1

    def test_rejection(self, sample_trade):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'true'}):
            approval = OperatorApproval()
            with patch('builtins.input', return_value='n'):
                assert approval.request_approval(sample_trade, 1000.0) is False
            assert approval.stats['rejected'] == 1

    def test_eof_defaults_to_reject(self, sample_trade):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'true'}):
            approval = OperatorApproval()
            with patch('builtins.input', side_effect=EOFError):
                assert approval.request_approval(sample_trade, 1000.0) is False
            assert approval.stats['rejected'] == 1

    def test_auto_approve_after_n(self, sample_trade):
        with patch.dict('os.environ', {
            'OPERATOR_APPROVAL': 'true',
            'AUTO_APPROVE_AFTER': '2',
        }):
            approval = OperatorApproval()
            # Manually approve 2
            with patch('builtins.input', return_value='y'):
                approval.request_approval(sample_trade, 1000.0)
                approval.request_approval(sample_trade, 1000.0)
            # Third should auto-approve (no input needed)
            assert approval.request_approval(sample_trade, 1000.0) is True
            assert approval.stats['approved'] == 3

    def test_format_summary_disabled(self):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'false'}):
            approval = OperatorApproval()
            assert 'DISABLED' in approval.format_summary()

    def test_format_summary_enabled(self):
        with patch.dict('os.environ', {'OPERATOR_APPROVAL': 'true'}):
            approval = OperatorApproval()
            assert 'ENABLED' in approval.format_summary()


# ==================== Schema Migration Tests ====================

class TestSchemaMigration:
    """Test that mode column is added to existing databases."""

    def test_mode_column_exists_in_trades(self, ledger):
        columns = [
            row[1] for row in
            ledger._conn.execute('PRAGMA table_info(trades)').fetchall()
        ]
        assert 'mode' in columns

    def test_mode_column_exists_in_cycles(self, ledger):
        columns = [
            row[1] for row in
            ledger._conn.execute('PRAGMA table_info(cycles)').fetchall()
        ]
        assert 'mode' in columns

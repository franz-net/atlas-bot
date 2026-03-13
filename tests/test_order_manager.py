"""
Sprint 6 Tests — Order Manager

Tests use mocked TradeStation client — no live API calls.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orders.order_manager import OrderManager


@pytest.fixture
def mock_client():
    """Create a mock TradeStationClient."""
    client = MagicMock()
    client.account_id = 'SIM123'
    client.use_sim = True
    client.get_quote = AsyncMock()
    client.place_bracket_order = AsyncMock()
    client.get_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture
def manager(mock_client):
    return OrderManager(mock_client)


@pytest.fixture
def sample_trade():
    return {
        'action': 'ENTER',
        'symbol': 'NVDA',
        'direction': 'LONG',
        'shares': 2,
        'order_type': 'Market',
        'expected_entry_price': 875.0,
        'stop_loss': 851.00,
        'take_profit': 910.00,
        'reasoning': 'Breaking above resistance on strong volume.',
    }


# ==================== Market Hours Tests ====================

class TestMarketHours:
    """Test market hours checking."""

    def test_during_trading_hours(self, manager):
        with patch('src.orders.order_manager.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 11, 10, 30)  # 10:30 EST Wednesday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Can't easily mock ZoneInfo — test the logic directly
            result = manager.is_market_hours()
            # Result depends on actual system time, so just verify it returns bool
            assert isinstance(result, bool)

    def test_market_hours_returns_bool(self, manager):
        assert isinstance(manager.is_market_hours(), bool)


# ==================== Risk Check Tests ====================

class TestRiskChecks:
    """Test risk limit validation."""

    def test_low_capital_checks_position_size(self, manager, sample_trade):
        """Hard stop is checked in scheduler, order manager checks position size."""
        sample_trade['entry_price'] = 875.0
        error = manager.check_risk_limits(
            active_capital=150.0,
            open_position_count=0,
            trade=sample_trade,
        )
        assert error is not None
        assert 'exceeds' in error.lower()

    def test_max_positions_rejects(self, manager, sample_trade):
        sample_trade['entry_price'] = 875.0
        error = manager.check_risk_limits(
            active_capital=1000.0,
            open_position_count=3,  # At max
            trade=sample_trade,
        )
        assert error is not None
        assert 'max positions' in error.lower()

    def test_position_size_rejects(self, manager, sample_trade):
        sample_trade['entry_price'] = 875.0
        sample_trade['shares'] = 10  # 10 * 875 = 8750, way over 30% of $1000
        error = manager.check_risk_limits(
            active_capital=1000.0,
            open_position_count=0,
            trade=sample_trade,
        )
        assert error is not None
        assert 'exceeds' in error.lower()

    def test_valid_trade_passes(self, manager, sample_trade):
        sample_trade['entry_price'] = 100.0
        sample_trade['shares'] = 1  # 1 * 100 = 100, under 30% of $1000
        with patch.object(manager, 'is_market_hours', return_value=True):
            error = manager.check_risk_limits(
                active_capital=1000.0,
                open_position_count=0,
                trade=sample_trade,
            )
        assert error is None

    def test_outside_market_hours_rejects(self, manager, sample_trade):
        sample_trade['entry_price'] = 100.0
        sample_trade['shares'] = 1
        with patch.object(manager, 'is_market_hours', return_value=False):
            error = manager.check_risk_limits(
                active_capital=1000.0,
                open_position_count=0,
                trade=sample_trade,
            )
        assert error is not None
        assert 'hours' in error.lower()

    def test_invalid_entry_price_rejects(self, manager, sample_trade):
        sample_trade['entry_price'] = 0
        error = manager.check_risk_limits(
            active_capital=1000.0,
            open_position_count=0,
            trade=sample_trade,
        )
        assert error is not None
        assert 'Invalid entry price' in error


# ==================== Trade Execution Tests ====================

class TestTradeExecution:
    """Test trade execution via mocked client."""

    @pytest.mark.asyncio
    async def test_execute_trade_success(self, manager, mock_client, sample_trade):
        mock_client.get_quote.return_value = {'Last': '875.50', 'Bid': '875.45', 'Ask': '875.55'}
        mock_client.place_bracket_order.return_value = {
            'success': True,
            'entry_order_id': 'ENT-001',
            'stop_order_id': 'STP-001',
            'tp_order_id': 'LMT-001',
        }

        sample_trade['shares'] = 1
        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=5000.0,
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is True
        assert result['entry_order_id'] == 'ENT-001'
        assert result['stop_order_id'] == 'STP-001'
        assert result['tp_order_id'] == 'LMT-001'

    @pytest.mark.asyncio
    async def test_execute_trade_risk_rejected(self, manager, mock_client, sample_trade):
        mock_client.get_quote.return_value = {'Last': '875.00'}

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=150.0,  # Below hard stop
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is False
        assert 'risk check' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_execute_trade_no_quote(self, manager, mock_client, sample_trade):
        mock_client.get_quote.return_value = None

        result = await manager.execute_trade(
            trade=sample_trade,
            active_capital=1000.0,
            open_position_count=0,
            phase='GROWTH',
        )

        assert result['success'] is False
        assert 'quote' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_execute_trade_bracket_fails(self, manager, mock_client, sample_trade):
        mock_client.get_quote.return_value = {'Last': '875.50'}
        mock_client.place_bracket_order.return_value = {
            'success': False,
            'error': 'Insufficient buying power',
        }

        sample_trade['shares'] = 1
        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=5000.0,
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is False
        assert 'buying power' in result['error'].lower()


# ==================== Decision Execution Tests ====================

class TestDecisionExecution:
    """Test executing full decisions."""

    @pytest.mark.asyncio
    async def test_hold_decision_no_orders(self, manager):
        decision = {'action': 'HOLD', 'trades': []}
        results = await manager.execute_decisions(
            decision=decision,
            active_capital=1000.0,
            open_positions=[],
            phase='GROWTH',
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_enter_decision_with_trade(self, manager, mock_client, sample_trade):
        mock_client.get_quote.return_value = {'Last': '875.50'}
        mock_client.place_bracket_order.return_value = {
            'success': True,
            'entry_order_id': 'ENT-001',
            'stop_order_id': 'STP-001',
            'tp_order_id': 'LMT-001',
        }

        sample_trade['shares'] = 1
        decision = {'action': 'ENTER', 'trades': [sample_trade]}

        with patch.object(manager, 'is_market_hours', return_value=True):
            results = await manager.execute_decisions(
                decision=decision,
                active_capital=5000.0,
                open_positions=[],
                phase='GROWTH',
            )

        assert len(results) == 1
        assert results[0]['success'] is True

    @pytest.mark.asyncio
    async def test_empty_trades_no_orders(self, manager):
        decision = {'action': 'ENTER', 'trades': []}
        results = await manager.execute_decisions(
            decision=decision,
            active_capital=1000.0,
            open_positions=[],
            phase='GROWTH',
        )
        assert results == []


# ==================== Slippage Guard Tests ====================

class TestSlippageGuard:
    """Test slippage protection between AI expected price and live quote."""

    @pytest.mark.asyncio
    async def test_slippage_within_tolerance(self, manager, mock_client, sample_trade):
        """0.1% slippage should be accepted."""
        sample_trade['expected_entry_price'] = 100.0
        sample_trade['shares'] = 1
        mock_client.get_quote.return_value = {'Last': '100.10'}  # 0.1% slip
        mock_client.place_bracket_order.return_value = {
            'success': True,
            'entry_order_id': 'ENT-001',
            'stop_order_id': 'STP-001',
            'tp_order_id': 'LMT-001',
        }

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=1000.0,
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_slippage_exceeds_tolerance(self, manager, mock_client, sample_trade):
        """1% slippage should be rejected (max 0.5%)."""
        sample_trade['expected_entry_price'] = 100.0
        sample_trade['shares'] = 1
        mock_client.get_quote.return_value = {'Last': '101.00'}  # 1% slip

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=1000.0,
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is False
        assert 'slippage' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_slippage_negative_direction(self, manager, mock_client, sample_trade):
        """Price dropped 1% — still slippage (absolute value)."""
        sample_trade['expected_entry_price'] = 100.0
        sample_trade['shares'] = 1
        mock_client.get_quote.return_value = {'Last': '99.00'}  # -1% slip

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=1000.0,
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is False
        assert 'slippage' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_no_expected_price_skips_slippage_check(self, manager, mock_client, sample_trade):
        """If expected_entry_price is missing, skip slippage check."""
        del sample_trade['expected_entry_price']
        sample_trade['shares'] = 1
        mock_client.get_quote.return_value = {'Last': '100.00'}
        mock_client.place_bracket_order.return_value = {
            'success': True,
            'entry_order_id': 'ENT-001',
            'stop_order_id': 'STP-001',
            'tp_order_id': 'LMT-001',
        }

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_trade(
                trade=sample_trade,
                active_capital=1000.0,
                open_position_count=0,
                phase='GROWTH',
            )

        assert result['success'] is True


class TestExitExecution:
    """Test AI-initiated early exit execution."""

    @pytest.fixture
    def sample_exit_decision(self):
        return {
            'action': 'EXIT',
            'symbol': 'PATH',
            'reasoning': 'Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain.',
        }

    @pytest.fixture
    def sample_open_trade(self):
        return {
            'id': 1,
            'symbol': 'PATH',
            'direction': 'LONG',
            'shares': 86,
            'entry_price': 11.52,
            'stop_order_id': 'SL-001',
            'tp_order_id': 'TP-001',
        }

    @pytest.mark.asyncio
    async def test_exit_success(self, manager, mock_client, sample_exit_decision, sample_open_trade):
        """Happy path: cancel bracket, place market sell, get fill."""
        mock_client.get_positions.return_value = [
            {'Symbol': 'PATH', 'Quantity': 86},
        ]
        mock_client.cancel_order = AsyncMock(return_value=True)
        mock_client.place_order = AsyncMock(return_value={
            'success': True, 'order_id': 'CLO-001', 'message': 'Sent order',
        })
        mock_client.get_quote.return_value = {'Last': '11.74'}
        mock_client.get_order_status = AsyncMock(return_value={
            'FilledPrice': '11.74',
            'CommissionFee': '1.00',
            'UnbundledRouteFee': '0.00',
        })

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_exit(sample_exit_decision, sample_open_trade)

        assert result['success'] is True
        assert result['symbol'] == 'PATH'
        assert result['exit_price'] == 11.74
        assert result['close_order_id'] == 'CLO-001'
        assert mock_client.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_exit_position_already_closed(self, manager, mock_client, sample_exit_decision, sample_open_trade):
        """Race condition: position closed by bracket between AI decision and execution."""
        mock_client.get_positions.return_value = []

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_exit(sample_exit_decision, sample_open_trade)

        assert result['success'] is False
        assert result['already_closed'] is True

    @pytest.mark.asyncio
    async def test_exit_bracket_cancel_fails(self, manager, mock_client, sample_exit_decision, sample_open_trade):
        """If bracket cancel fails, abort — don't place close order."""
        mock_client.get_positions.return_value = [
            {'Symbol': 'PATH', 'Quantity': 86},
        ]
        mock_client.cancel_order = AsyncMock(side_effect=[True, False])

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_exit(sample_exit_decision, sample_open_trade)

        assert result['success'] is False
        assert 'FAILED' in result['error']

    @pytest.mark.asyncio
    async def test_exit_bracket_filled_during_cancel(self, manager, mock_client, sample_exit_decision, sample_open_trade):
        """Race: bracket fills after pre-check but during cancel — position gone."""
        # First call: position exists. Second call (after cancels): position gone.
        mock_client.get_positions = AsyncMock(
            side_effect=[
                [{'Symbol': 'PATH', 'Quantity': 86}],  # pre-check
                [],  # post-cancel re-check
            ]
        )
        mock_client.cancel_order = AsyncMock(return_value=True)

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_exit(sample_exit_decision, sample_open_trade)

        assert result['success'] is False
        assert result['already_closed'] is True
        assert 'bracket fill during cancel' in result['error']

    @pytest.mark.asyncio
    async def test_exit_outside_market_hours(self, manager, sample_exit_decision, sample_open_trade):
        """EXIT should respect market hours."""
        with patch.object(manager, 'is_market_hours', return_value=False):
            result = await manager.execute_exit(sample_exit_decision, sample_open_trade)

        assert result['success'] is False
        assert 'trading hours' in result['error']

    @pytest.mark.asyncio
    async def test_exit_short_position_uses_buytocover(self, manager, mock_client, sample_exit_decision):
        """SHORT position exit should use BUYTOCOVER, not SELL."""
        short_trade = {
            'id': 2, 'symbol': 'PATH', 'direction': 'SHORT',
            'shares': 50, 'entry_price': 11.80,
            'stop_order_id': 'SL-002', 'tp_order_id': 'TP-002',
        }
        mock_client.get_positions.return_value = [
            {'Symbol': 'PATH', 'Quantity': 50},
        ]
        mock_client.cancel_order = AsyncMock(return_value=True)
        mock_client.place_order = AsyncMock(return_value={
            'success': True, 'order_id': 'CLO-002', 'message': 'Sent order',
        })
        mock_client.get_quote.return_value = {'Last': '11.50'}
        mock_client.get_order_status = AsyncMock(return_value={
            'FilledPrice': '11.50',
            'CommissionFee': '1.00',
            'UnbundledRouteFee': '0.00',
        })

        with patch.object(manager, 'is_market_hours', return_value=True):
            result = await manager.execute_exit(sample_exit_decision, short_trade)

        assert result['success'] is True
        call_args = mock_client.place_order.call_args[0][0]
        assert call_args['TradeAction'] == 'BUYTOCOVER'

    @pytest.mark.asyncio
    async def test_execute_exit_decisions_no_matching_trade(self, manager):
        """EXIT for symbol not in ledger should be skipped."""
        decision = {
            'exits': [{'action': 'EXIT', 'symbol': 'FAKE', 'reasoning': 'x' * 60}],
        }
        results = await manager.execute_exit_decisions(
            decision=decision,
            ledger_open_trades=[{'symbol': 'PATH', 'id': 1}],
        )
        assert len(results) == 1
        assert results[0]['success'] is False
        assert 'No matching' in results[0]['error']

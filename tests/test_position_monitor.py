"""
Sprint 6 Tests — Position Monitor

Tests use mocked client and in-memory ledger.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ledger.ledger import TradingLedger
from src.orders.position_monitor import PositionMonitor


@pytest.fixture
def mock_client():
    """Create a mock TradeStationClient."""
    client = MagicMock()
    client.account_id = 'SIM123'
    client.get_positions = AsyncMock(return_value=[])
    client.get_order_status = AsyncMock(return_value={})
    client.get_quote = AsyncMock(return_value={'Last': '900.00'})
    return client


@pytest.fixture
def ledger():
    """Create an in-memory ledger."""
    l = TradingLedger(db_path=':memory:')
    yield l
    l.close()


@pytest.fixture
def monitor(mock_client, ledger):
    return PositionMonitor(mock_client, ledger)


@pytest.fixture
def open_trade(ledger):
    """Create an open trade in the ledger."""
    trade_id = ledger.record_trade_entry(
        cycle_id='test-cycle-001',
        symbol='NVDA',
        direction='LONG',
        shares=2,
        entry_price=875.40,
        stop_loss_price=851.00,
        take_profit_price=910.00,
        entry_reasoning='Test trade entry.',
        news_catalyst=None,
        entry_order_id='ENT-001',
        stop_order_id='STP-001',
        tp_order_id='LMT-001',
        phase='GROWTH',
        active_capital=1000.0,
    )
    return trade_id


# ==================== Position Fetch Tests ====================

class TestPositionFetch:
    """Test broker position fetching."""

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, monitor, mock_client):
        mock_client.get_positions.return_value = []
        positions = await monitor.get_broker_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_with_data(self, monitor, mock_client):
        mock_client.get_positions.return_value = [
            {'Symbol': 'NVDA', 'Quantity': 2},
            {'Symbol': 'AAPL', 'Quantity': 5},
        ]
        positions = await monitor.get_broker_positions()
        assert len(positions) == 2

    @pytest.mark.asyncio
    async def test_get_positions_error(self, monitor, mock_client):
        mock_client.get_positions.side_effect = Exception("API error")
        positions = await monitor.get_broker_positions()
        assert positions == []


# ==================== Exit Detection Tests ====================

class TestExitDetection:
    """Test exit detection by comparing broker vs ledger."""

    @pytest.mark.asyncio
    async def test_no_exits_when_still_open(self, monitor, mock_client, open_trade):
        """Trade still at broker = no exit detected."""
        mock_client.get_positions.return_value = [
            {'Symbol': 'NVDA', 'Quantity': 2},
        ]
        exits = await monitor.check_for_exits(current_balance=26000)
        assert len(exits) == 0

    @pytest.mark.asyncio
    async def test_exit_detected_stop_loss(self, monitor, mock_client, open_trade):
        """Trade gone from broker + SL filled = exit recorded."""
        mock_client.get_positions.return_value = []  # Position gone
        mock_client.get_order_status.side_effect = [
            # Stop loss order filled
            {'Status': 'FLL', 'FilledPrice': 851.00},
        ]

        exits = await monitor.check_for_exits(current_balance=26000)
        assert len(exits) == 1
        assert exits[0]['symbol'] == 'NVDA'
        assert exits[0]['exit_reasoning'] == 'Stop loss hit'
        assert exits[0]['pnl_dollars'] < 0  # Loss

    @pytest.mark.asyncio
    async def test_exit_detected_take_profit(self, monitor, mock_client, open_trade):
        """Trade gone from broker + TP filled = exit recorded."""
        mock_client.get_positions.return_value = []
        mock_client.get_order_status.side_effect = [
            # Stop loss NOT filled
            {'Status': 'CAN'},
            # Take profit filled
            {'Status': 'FLL', 'FilledPrice': 910.00},
        ]

        exits = await monitor.check_for_exits(current_balance=26000)
        assert len(exits) == 1
        assert exits[0]['exit_reasoning'] == 'Take profit hit'
        assert exits[0]['pnl_dollars'] > 0  # Win

    @pytest.mark.asyncio
    async def test_no_open_trades_no_exits(self, monitor, mock_client):
        """No open trades in ledger = nothing to check."""
        exits = await monitor.check_for_exits(current_balance=26000)
        assert len(exits) == 0


# ==================== Position Sync Tests ====================

class TestPositionSync:
    """Test full position sync."""

    @pytest.mark.asyncio
    async def test_sync_returns_structure(self, monitor, mock_client):
        mock_client.get_positions.return_value = []
        result = await monitor.sync_positions(current_balance=26000)
        assert 'broker_positions' in result
        assert 'exits_recorded' in result
        assert 'open_trade_count' in result

    @pytest.mark.asyncio
    async def test_sync_with_open_trade(self, monitor, mock_client, open_trade):
        mock_client.get_positions.return_value = [
            {'Symbol': 'NVDA', 'Quantity': 2},
        ]
        result = await monitor.sync_positions(current_balance=26000)
        assert result['open_trade_count'] == 1
        assert len(result['exits_recorded']) == 0

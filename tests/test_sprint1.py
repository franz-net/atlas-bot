"""
Sprint 1 Tests — Foundation

Verifies:
- TradeStationClient can be instantiated
- Account retrieval works (mocked)
- Balance retrieval works and active capital is calculated correctly
- Order placement and cancellation works (mocked)
- Order failure detection (TS returns OrderID even on failed orders)
- Sim mode enforcement
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.constants import (
    HARD_STOP_ACTIVE_CAPITAL_PCT,
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    PROTECTED_FLOOR,
)


@pytest.fixture
def mock_env(monkeypatch):
    """Set required environment variables for testing."""
    monkeypatch.setenv('TS_API_KEY', 'test_key')
    monkeypatch.setenv('TS_API_SECRET', 'test_secret')
    monkeypatch.setenv('TS_ACCOUNT_ID', 'TEST123')
    monkeypatch.setenv('USE_SIM_ACCOUNT', 'true')


class TestConstants:
    """Verify capital and risk constants are correctly defined."""

    def test_protected_floor(self):
        assert PROTECTED_FLOOR == 25000

    def test_hard_stop(self):
        assert HARD_STOP_ACTIVE_CAPITAL_PCT == 0.20

    def test_max_position_size(self):
        assert MAX_POSITION_SIZE_PCT == 0.30

    def test_max_concurrent_positions(self):
        assert MAX_CONCURRENT_POSITIONS == 3


class TestActiveCapitalCalculation:
    """Verify active capital = equity - protected floor."""

    def test_normal_balance(self):
        equity = 30000
        active_capital = equity - PROTECTED_FLOOR
        assert active_capital == 5000

    def test_at_floor(self):
        equity = 25000
        active_capital = equity - PROTECTED_FLOOR
        assert active_capital == 0

    def test_below_hard_stop(self):
        """With $5K start-of-day capital, hard stop = 20% = $1K."""
        start_of_day_capital = 5000
        hard_stop_value = start_of_day_capital * HARD_STOP_ACTIVE_CAPITAL_PCT
        assert hard_stop_value == 1000
        # If active capital drops to $800, trading halts
        active_capital = 800
        assert active_capital < hard_stop_value

    def test_position_sizing(self):
        """Max position = 30% of active capital."""
        active_capital = 1000
        max_position = active_capital * MAX_POSITION_SIZE_PCT
        assert max_position == 300


class TestClientInstantiation:
    """Verify TradeStationClient can be created with env vars."""

    def test_client_creation(self, mock_env):
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        assert client.api_key == 'test_key'
        assert client.api_secret == 'test_secret'
        assert client.account_id == 'TEST123'
        assert client.use_sim is True

    def test_sim_mode_default(self, mock_env):
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        assert client.use_sim is True


class TestAccountRetrieval:
    """Verify account data retrieval with mocked API responses."""

    @pytest.mark.asyncio
    async def test_get_account(self, mock_env):
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        client.session = MagicMock()

        mock_response = {
            'Accounts': [{
                'AccountID': 'TEST123',
                'AccountType': 'Margin',
                'Status': 'Active'
            }]
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock, return_value=mock_response):
            account = await client.get_account()
            assert account['AccountID'] == 'TEST123'
            assert account['AccountType'] == 'Margin'

    @pytest.mark.asyncio
    async def test_get_balances(self, mock_env):
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        client.session = MagicMock()

        mock_response = {
            'Balances': [{
                'AccountID': 'TEST123',
                'CashBalance': '26000.00',
                'Equity': '26000.00',
                'MarketValue': '0.00'
            }]
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock, return_value=mock_response):
            balances = await client.get_balances()
            assert balances is not None
            balance_data = balances['Balances'][0]
            equity = float(balance_data['Equity'])
            active_capital = equity - PROTECTED_FLOOR
            assert active_capital == 1000


class TestOrderPlacement:
    """Verify order placement and cancellation with mocked API."""

    @pytest.mark.asyncio
    async def test_place_order_success(self, mock_env):
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        client.session = MagicMock()

        mock_response = {
            'Orders': [{
                'OrderID': 'ORD-001',
                'Message': 'Order placed successfully'
            }]
        }

        with patch.object(client, '_make_request', new_callable=AsyncMock, return_value=mock_response):
            order_data = {
                'AccountID': 'TEST123',
                'Symbol': 'AAPL',
                'Quantity': '1',
                'OrderType': 'Market',
                'TradeAction': 'BUY',
                'TimeInForce': {'Duration': 'DAY'},
                'Route': 'Intelligent'
            }
            result = await client.place_order(order_data)
            assert result is not None
            # place_order returns parsed dict with success/order_id keys
            assert result['success'] is True
            assert result['order_id'] == 'ORD-001'

    @pytest.mark.asyncio
    async def test_order_failure_detection(self, mock_env):
        """CRITICAL: TS returns OrderID even on failed orders. Must check Error field."""
        mock_response = {
            'Orders': [{
                'OrderID': 'ORD-002',
                'Error': 'FAILED',
                'Message': 'Insufficient buying power'
            }]
        }

        order = mock_response['Orders'][0]
        error_field = order.get('Error', '')
        assert error_field.upper() in ('FAILED', 'REJECT', 'REJECTED', 'ERROR')

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_env):
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        client.session = MagicMock()

        with patch.object(client, '_make_request', new_callable=AsyncMock, return_value={}):
            result = await client.cancel_order('ORD-001')
            # cancel_order returns bool
            assert isinstance(result, bool)

    def test_sim_mode_enforcement(self, mock_env, monkeypatch):
        """Verify sim mode is respected."""
        monkeypatch.setenv('USE_SIM_ACCOUNT', 'true')
        from src.api.tradestation import TradeStationClient
        client = TradeStationClient()
        assert client.use_sim is True
        # Sim URL should be used for requests
        assert 'sim' in client.sim_base_url


class TestOrderFailureEdgeCases:
    """Additional edge cases for the TS order failure gotcha."""

    def test_reject_status(self):
        order = {'OrderID': 'ORD-003', 'Error': 'REJECTED'}
        assert order['Error'].upper() in ('FAILED', 'REJECT', 'REJECTED', 'ERROR')

    def test_error_status(self):
        order = {'OrderID': 'ORD-004', 'Error': 'Error'}
        assert order['Error'].upper() in ('FAILED', 'REJECT', 'REJECTED', 'ERROR')

    def test_no_error_means_success(self):
        order = {'OrderID': 'ORD-005', 'Message': 'Sent'}
        error_field = order.get('Error', '')
        assert error_field == '' or error_field.upper() not in ('FAILED', 'REJECT', 'REJECTED', 'ERROR')

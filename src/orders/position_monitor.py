"""
Position monitor for Project Atlas.

Polls TradeStation for open position changes, detects fills and exits,
and records trade outcomes to the ledger.
"""

import logging
from typing import Dict, List, Optional

from src.api.tradestation import TradeStationClient
from src.config.constants import PROTECTED_FLOOR
from src.ledger.ledger import TradingLedger
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'position_monitor')


class PositionMonitor:
    """
    Monitors open positions and records exits to the ledger.

    Compares TradeStation positions against ledger open trades to detect
    fills and closures (stop hit, take profit hit, manual exit).
    """

    def __init__(self, client: TradeStationClient, ledger: TradingLedger):
        """
        Initialize the position monitor.

        Args:
            client: Authenticated TradeStationClient instance
            ledger: TradingLedger for recording exits
        """
        self._client = client
        self._ledger = ledger

    async def get_broker_positions(self) -> List[Dict]:
        """
        Fetch current positions from TradeStation.

        Returns:
            List of position dicts with Symbol, Quantity, etc.
        """
        try:
            positions = await self._client.get_positions()
            if positions is None:
                return []
            logger.info(f"Broker reports {len(positions)} open positions")
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def check_for_exits(self, current_balance: float) -> List[Dict]:
        """
        Compare broker positions against ledger to detect closed trades.

        A ledger trade marked OPEN that no longer appears in broker positions
        has been filled (SL or TP hit, or manual close).

        Args:
            current_balance: Current account balance for active capital calculation

        Returns:
            List of exit records that were recorded to ledger
        """
        broker_positions = await self.get_broker_positions()
        broker_symbols = {
            p.get('Symbol', '') for p in broker_positions
        }

        ledger_open = self._ledger.get_open_trades()
        if not ledger_open:
            return []

        exits_recorded = []
        active_capital = current_balance - PROTECTED_FLOOR

        for trade in ledger_open:
            symbol = trade['symbol']
            trade_id = trade['id']

            if symbol in broker_symbols:
                # Still open at broker — no exit
                continue

            # Position closed at broker — determine exit details
            exit_info = await self._determine_exit(trade)

            if exit_info is None:
                logger.warning(
                    f"Trade {trade_id} ({symbol}) closed at broker "
                    f"but could not determine exit details"
                )
                continue

            # Record exit to ledger
            success = self._ledger.record_trade_exit(
                trade_id=trade_id,
                exit_price=exit_info['exit_price'],
                pnl_dollars=exit_info['pnl_dollars'],
                pnl_pct=exit_info['pnl_pct'],
                exit_reasoning=exit_info['exit_reasoning'],
                active_capital=active_capital,
                exit_fees=exit_info.get('exit_fees', 0.0),
            )

            if success:
                exit_record = {
                    'trade_id': trade_id,
                    'symbol': symbol,
                    **exit_info,
                }
                exits_recorded.append(exit_record)
                logger.info(
                    f"Exit recorded: {symbol} P&L=${exit_info['pnl_dollars']:.2f} "
                    f"({exit_info['exit_reasoning']})"
                )

        return exits_recorded

    async def _determine_exit(self, trade: Dict) -> Optional[Dict]:
        """
        Determine how a trade was closed by checking order statuses.

        Checks stop loss and take profit order IDs to see which filled.

        Args:
            trade: Ledger trade record

        Returns:
            Exit info dict or None if undetermined
        """
        stop_order_id = trade.get('stop_order_id')
        tp_order_id = trade.get('tp_order_id')
        entry_price = trade.get('entry_price', 0)
        shares = trade.get('shares', 0)
        direction = trade.get('direction', 'LONG')

        exit_price = 0.0
        exit_fees = 0.0
        exit_reasoning = 'Position closed'

        # Check if stop loss was filled
        if stop_order_id:
            stop_status = await self._client.get_order_status(stop_order_id)
            if stop_status and stop_status.get('Status') in ('FLL', 'FLP'):
                exit_price = float(stop_status.get('FilledPrice', 0))
                if exit_price == 0:
                    exit_price = float(stop_status.get('StopPrice', trade.get('stop_loss_price', 0)))
                exit_fees = (
                    float(stop_status.get('CommissionFee', 0) or 0)
                    + float(stop_status.get('UnbundledRouteFee', 0) or 0)
                )
                exit_reasoning = 'Stop loss hit'

        # Check if take profit was filled
        if tp_order_id and exit_price == 0:
            tp_status = await self._client.get_order_status(tp_order_id)
            if tp_status and tp_status.get('Status') in ('FLL', 'FLP'):
                exit_price = float(tp_status.get('FilledPrice', 0))
                if exit_price == 0:
                    exit_price = float(tp_status.get('LimitPrice', trade.get('take_profit_price', 0)))
                exit_fees = (
                    float(tp_status.get('CommissionFee', 0) or 0)
                    + float(tp_status.get('UnbundledRouteFee', 0) or 0)
                )
                exit_reasoning = 'Take profit hit'

        # If neither bracket order filled, try getting a quote for current price
        if exit_price == 0:
            quote = await self._client.get_quote(trade['symbol'])
            if quote:
                exit_price = float(quote.get('Last', 0))
                exit_reasoning = 'Position closed (manual or broker)'

        if exit_price <= 0 or entry_price <= 0:
            return None

        # Calculate P&L
        if direction == 'LONG':
            pnl_dollars = (exit_price - entry_price) * shares
        else:
            pnl_dollars = (entry_price - exit_price) * shares

        pnl_pct = (pnl_dollars / (entry_price * shares)) * 100 if entry_price > 0 else 0

        return {
            'exit_price': exit_price,
            'pnl_dollars': pnl_dollars,
            'pnl_pct': pnl_pct,
            'exit_reasoning': exit_reasoning,
            'exit_fees': exit_fees,
        }

    async def sync_positions(self, current_balance: float) -> Dict:
        """
        Full position sync: check exits and return current state.

        Args:
            current_balance: Current account balance

        Returns:
            Dict with broker_positions, exits_recorded, and open_trade_count
        """
        exits = await self.check_for_exits(current_balance)
        positions = await self.get_broker_positions()
        open_trades = self._ledger.get_open_trades()

        return {
            'broker_positions': positions,
            'exits_recorded': exits,
            'open_trade_count': len(open_trades),
        }

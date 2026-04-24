"""
Order manager for Project Atlas.

Translates AI trading decisions into TradeStation bracket orders.
Enforces all risk checks (capital, position limits, market hours)
BEFORE placing any order. Never places naked positions.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.api.tradestation import TradeStationClient
from src.config.constants import (
    DEFAULT_TICK_SIZE,
    MARKET_CLOSE_HOUR,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    MAX_SLIPPAGE_PCT,
    NO_ENTRY_AFTER_HOUR,
    NO_ENTRY_AFTER_MINUTE,
    NO_ENTRY_BEFORE_MINUTE,
    PROTECTED_FLOOR,
)
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'order_manager')


class OrderManager:
    """
    Executes AI trading decisions via TradeStation bracket orders.

    All risk checks happen here — the decision engine proposes trades,
    the order manager decides if they are safe to execute.
    """

    def __init__(self, client: TradeStationClient):
        """
        Initialize the order manager.

        Args:
            client: Authenticated TradeStationClient instance
        """
        self._client = client

    def is_market_hours(self) -> bool:
        """
        Window in which ENTRIES are allowed: 9:35 - 15:45 EST.

        The early buffer avoids the open-auction chaos; the late buffer
        avoids opening fresh overnight risk in the last 15 minutes.

        Returns:
            True if within the entry window.
        """
        est = ZoneInfo('US/Eastern')
        now = datetime.now(est)

        open_time = now.replace(
            hour=MARKET_OPEN_HOUR, minute=NO_ENTRY_BEFORE_MINUTE,
            second=0, microsecond=0,
        )
        close_time = now.replace(
            hour=NO_ENTRY_AFTER_HOUR, minute=NO_ENTRY_AFTER_MINUTE,
            second=0, microsecond=0,
        )

        return open_time <= now <= close_time

    def is_exchange_open(self) -> bool:
        """
        Window in which EXITS are allowed: 9:30 - 15:58 EST.

        Wider than the entry window because closing a position is risk-
        reducing — it should be allowed right up to the close, only
        stopping ~2 minutes before to leave room for the fill.

        Returns:
            True if the exchange is accepting exit orders.
        """
        est = ZoneInfo('US/Eastern')
        now = datetime.now(est)

        open_time = now.replace(
            hour=MARKET_OPEN_HOUR, minute=30, second=0, microsecond=0,
        )
        close_time = now.replace(
            hour=15, minute=58, second=0, microsecond=0,
        )

        return open_time <= now <= close_time

    def check_risk_limits(
        self,
        active_capital: float,
        open_position_count: int,
        trade: Dict,
    ) -> Optional[str]:
        """
        Validate a proposed trade against all risk rules.

        Args:
            active_capital: Current active capital (balance - floor)
            open_position_count: Number of currently open positions
            trade: Proposed trade dict from decision engine

        Returns:
            Error message if trade is rejected, None if safe
        """
        # Hard stop check — not applicable in order manager (checked in scheduler)
        # Order manager validates position-level risk only

        # Position limit check
        if open_position_count >= MAX_CONCURRENT_POSITIONS:
            return (
                f"At max positions ({open_position_count}/{MAX_CONCURRENT_POSITIONS})"
            )

        # Position size check
        shares = trade.get('shares', 0)
        entry_price = trade.get('entry_price', 0)
        if entry_price <= 0:
            return f"Invalid entry price: {entry_price}"

        position_value = shares * entry_price
        max_value = active_capital * MAX_POSITION_SIZE_PCT
        if position_value > max_value:
            return (
                f"Position ${position_value:.2f} exceeds "
                f"{int(MAX_POSITION_SIZE_PCT * 100)}% limit (${max_value:.2f})"
            )

        # Market hours check
        if not self.is_market_hours():
            return "Outside allowed trading hours (9:35-15:45 EST)"

        return None

    async def execute_trade(
        self,
        trade: Dict,
        active_capital: float,
        open_position_count: int,
        phase: str,
    ) -> Dict:
        """
        Execute a single trade from an AI decision.

        Places an entry order with bracket (stop loss + take profit)
        as a single OSO group.

        Args:
            trade: Validated trade dict from decision engine
            active_capital: Current active capital
            open_position_count: Current open position count
            phase: GROWTH or WITHDRAWAL

        Returns:
            Execution result dict with success, order IDs, and error info
        """
        symbol = trade['symbol']
        direction = trade['direction']
        shares = trade['shares']
        stop_loss = trade['stop_loss']
        take_profit = trade.get('take_profit', 0)

        result = {
            'success': False,
            'symbol': symbol,
            'direction': direction,
            'shares': shares,
            'entry_order_id': None,
            'stop_order_id': None,
            'tp_order_id': None,
            'error': None,
        }

        # Risk check
        # Use current quote price for risk validation
        quote = await self._client.get_quote(symbol)
        if not quote:
            result['error'] = f"Could not get quote for {symbol}"
            logger.error(result['error'])
            return result

        entry_price = float(quote.get('Last', 0))
        if entry_price <= 0:
            result['error'] = f"Invalid quote price for {symbol}: {entry_price}"
            logger.error(result['error'])
            return result

        # Slippage guard: reject if live price drifted too far from AI's expected entry
        expected_price = trade.get('expected_entry_price', 0)
        if expected_price > 0:
            slippage = abs(entry_price - expected_price) / expected_price
            if slippage > MAX_SLIPPAGE_PCT:
                result['error'] = (
                    f"Slippage {slippage * 100:.2f}% exceeds max "
                    f"{MAX_SLIPPAGE_PCT * 100:.1f}% "
                    f"(expected=${expected_price:.2f} live=${entry_price:.2f})"
                )
                logger.warning(f"Trade {symbol} rejected: {result['error']}")
                return result
            logger.info(
                f"Trade {symbol}: slippage {slippage * 100:.3f}% "
                f"(expected=${expected_price:.2f} live=${entry_price:.2f})"
            )

        trade_with_price = {**trade, 'entry_price': entry_price}
        risk_error = self.check_risk_limits(
            active_capital, open_position_count, trade_with_price,
        )
        if risk_error:
            result['error'] = f"Risk check failed: {risk_error}"
            logger.warning(f"Trade {symbol} rejected: {risk_error}")
            return result

        # Build entry order
        trade_action = 'BUY' if direction == 'LONG' else 'SELLSHORT'
        entry_order = {
            'AccountID': self._client.account_id,
            'Symbol': symbol,
            'Quantity': str(shares),
            'OrderType': trade.get('order_type', 'Market'),
            'TradeAction': trade_action,
            'TimeInForce': {'Duration': 'DAY'},
            'Route': 'Intelligent',
        }

        # Place bracket order (entry + SL + TP as OSO group)
        logger.info(
            f"Placing bracket order: {trade_action} {shares} {symbol} "
            f"SL={stop_loss} TP={take_profit}"
        )

        try:
            bracket_result = await self._client.place_bracket_order(
                entry_order=entry_order,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                tick_size=DEFAULT_TICK_SIZE,
            )

            if not bracket_result.get('success'):
                result['error'] = bracket_result.get('error', 'Bracket order failed')
                logger.error(
                    f"Bracket order failed for {symbol}: {result['error']}"
                )
                return result

            result['success'] = True
            result['entry_order_id'] = bracket_result.get('entry_order_id')
            result['stop_order_id'] = bracket_result.get('stop_order_id')
            result['tp_order_id'] = bracket_result.get('tp_order_id')
            result['entry_price'] = entry_price  # Quote price as fallback
            result['entry_fees'] = 0.0

            # Fetch actual fill price and fees from broker
            fill_details = await self._get_fill_details(
                result['entry_order_id'], entry_price
            )
            result['entry_price'] = fill_details['fill_price']
            result['entry_fees'] = fill_details['fees']

            logger.info(
                f"Bracket order placed: {symbol} entry={result['entry_order_id']} "
                f"fill=${fill_details['fill_price']:.2f} "
                f"fees=${fill_details['fees']:.2f} "
                f"SL={result['stop_order_id']} TP={result['tp_order_id']}"
            )
            return result

        except Exception as e:
            result['error'] = f"Order placement exception: {e}"
            logger.error(result['error'])
            return result

    async def _get_fill_details(
        self, order_id: str, fallback_price: float, retries: int = 3
    ) -> Dict:
        """
        Poll the broker for fill price and fees of a market order.

        Market orders fill almost instantly on sim, but we retry a few
        times with a brief delay in case the status hasn't updated yet.

        Args:
            order_id: Order ID to check
            fallback_price: Quote price to use if fill price unavailable
            retries: Number of attempts

        Returns:
            Dict with 'fill_price' and 'fees'
        """
        result = {'fill_price': fallback_price, 'fees': 0.0}

        if not order_id:
            return result

        for attempt in range(retries):
            try:
                await asyncio.sleep(1)
                status = await self._client.get_order_status(order_id)
                if not status:
                    continue

                # Extract fees: CommissionFee + UnbundledRouteFee
                commission = float(status.get('CommissionFee', 0) or 0)
                route_fee = float(status.get('UnbundledRouteFee', 0) or 0)
                result['fees'] = commission + route_fee

                # TS v3 fill price fields
                fill_price = status.get('FilledPrice') or status.get(
                    'AvgFilledPrice'
                )
                if fill_price:
                    price = float(fill_price)
                    if price > 0:
                        result['fill_price'] = price
                        logger.info(
                            f"Fill for {order_id}: ${price:.2f} "
                            f"fees=${result['fees']:.2f} "
                            f"(quote was ${fallback_price:.2f})"
                        )
                        return result

                # Check if order is filled
                order_status = status.get('Status', '')
                if order_status in ('FLL', 'FLP'):
                    # Filled but no price field — check legs
                    legs = status.get('Legs', [])
                    if legs:
                        exec_price = legs[0].get('ExecutionPrice',
                                    legs[0].get('ExecPrice', 0))
                        if exec_price and float(exec_price) > 0:
                            result['fill_price'] = float(exec_price)
                            logger.info(
                                f"Fill from legs for {order_id}: "
                                f"${result['fill_price']:.2f} "
                                f"fees=${result['fees']:.2f}"
                            )
                            return result

            except Exception as e:
                logger.warning(f"Fill details check attempt {attempt + 1} failed: {e}")

        logger.warning(
            f"Could not get fill details for {order_id}, "
            f"using quote ${fallback_price:.2f}"
        )
        return result

    async def execute_exit(
        self,
        exit_decision: Dict,
        open_trade: Dict,
    ) -> Dict:
        """
        Execute an AI-initiated early exit: cancel bracket, place market close.

        Args:
            exit_decision: Exit dict with symbol and reasoning
            open_trade: Matching trade from ledger (has order IDs, direction, shares)

        Returns:
            Execution result dict
        """
        symbol = exit_decision['symbol']
        direction = open_trade.get('direction', 'LONG')
        shares = open_trade.get('shares', 0)
        stop_order_id = open_trade.get('stop_order_id', '')
        tp_order_id = open_trade.get('tp_order_id', '')

        result = {
            'success': False,
            'symbol': symbol,
            'direction': direction,
            'shares': shares,
            'trade_id': open_trade.get('id'),
            'exit_price': 0,
            'exit_fees': 0,
            'close_order_id': None,
            'reasoning': exit_decision.get('reasoning', ''),
            'error': None,
            'already_closed': False,
        }

        # Verify position still exists at broker BEFORE the time guard.
        # An exit for a non-existent position is a no-op, not a policy
        # violation — and there's no point rejecting by time if there's
        # nothing to close.
        try:
            positions = await self._client.get_positions()
            broker_symbols = [
                p.get('Symbol', '') for p in (positions or [])
            ]
            if symbol not in broker_symbols:
                result['already_closed'] = True
                result['error'] = 'Position already closed by bracket fill'
                logger.info(f"EXIT {symbol}: position already closed at broker")
                return result
        except Exception as e:
            result['error'] = f"Failed to check positions: {e}"
            logger.error(result['error'])
            return result

        # Exchange-hours check (wider than the entry window). If a
        # position is open, closing it is risk-reducing and should be
        # allowed until ~2 min before close.
        if not self.is_exchange_open():
            result['error'] = "Exchange closed — cannot place exit order"
            logger.warning(f"EXIT {symbol} rejected: {result['error']}")
            return result

        # Cancel bracket orders (SL + TP)
        logger.info(f"EXIT {symbol}: canceling bracket orders SL={stop_order_id} TP={tp_order_id}")

        sl_cancelled = True
        tp_cancelled = True

        if stop_order_id:
            sl_cancelled = await self._client.cancel_order(stop_order_id)
        if tp_order_id:
            tp_cancelled = await self._client.cancel_order(tp_order_id)

        if not sl_cancelled or not tp_cancelled:
            result['error'] = (
                f"Failed to cancel bracket orders "
                f"(SL={'OK' if sl_cancelled else 'FAILED'}, "
                f"TP={'OK' if tp_cancelled else 'FAILED'})"
            )
            logger.error(f"EXIT {symbol}: {result['error']}")
            return result

        # Re-check position after cancels — bracket may have filled during cancel
        try:
            positions_after = await self._client.get_positions()
            symbols_after = [
                p.get('Symbol', '') for p in (positions_after or [])
            ]
            if symbol not in symbols_after:
                result['already_closed'] = True
                result['error'] = 'Position closed by bracket fill during cancel'
                logger.info(f"EXIT {symbol}: position gone after bracket cancel — bracket filled first")
                return result
        except Exception as e:
            logger.warning(f"EXIT {symbol}: post-cancel position check failed: {e} — proceeding with close")

        # Place market close order
        trade_action = 'SELL' if direction == 'LONG' else 'BUYTOCOVER'
        close_order = {
            'AccountID': self._client.account_id,
            'Symbol': symbol,
            'Quantity': str(shares),
            'OrderType': 'Market',
            'TradeAction': trade_action,
            'TimeInForce': {'Duration': 'DAY'},
            'Route': 'Intelligent',
        }

        logger.info(f"EXIT {symbol}: placing {trade_action} {shares} shares")

        try:
            order_result = await self._client.place_order(close_order)

            if not order_result:
                result['error'] = 'Close order returned no result'
                logger.error(f"EXIT {symbol}: {result['error']}")
                return result

            # place_order returns parsed response: {success, order_id, message, ...}
            if not order_result.get('success'):
                result['error'] = f"Close order failed: {order_result.get('error', 'unknown')}"
                logger.error(f"EXIT {symbol}: {result['error']}")
                return result

            close_order_id = order_result.get('order_id', '')
            result['close_order_id'] = close_order_id

            # Fetch fill details
            quote = await self._client.get_quote(symbol)
            fallback_price = float(quote.get('Last', 0)) if quote else 0

            fill_details = await self._get_fill_details(
                close_order_id, fallback_price
            )
            result['exit_price'] = fill_details['fill_price']
            result['exit_fees'] = fill_details['fees']
            result['success'] = True

            logger.info(
                f"EXIT {symbol}: closed {shares} shares @ ${result['exit_price']:.2f} "
                f"fees=${result['exit_fees']:.2f} order={close_order_id}"
            )
            return result

        except Exception as e:
            result['error'] = f"Close order exception: {e}"
            logger.error(result['error'])
            return result

    async def execute_exit_decisions(
        self,
        decision: Dict,
        ledger_open_trades: List[Dict],
    ) -> List[Dict]:
        """
        Execute all EXIT decisions from a cycle.

        Args:
            decision: Full decision result with exits list
            ledger_open_trades: Open trades from ledger (for order IDs)

        Returns:
            List of exit execution result dicts
        """
        exits = decision.get('exits', [])
        if not exits:
            return []

        # Build symbol → ledger trade lookup
        trade_by_symbol = {t['symbol']: t for t in ledger_open_trades}

        results = []
        for exit_decision in exits:
            symbol = exit_decision.get('symbol', '')
            open_trade = trade_by_symbol.get(symbol)

            if not open_trade:
                logger.warning(
                    f"EXIT {symbol}: no matching open trade in ledger — skipping"
                )
                results.append({
                    'success': False,
                    'symbol': symbol,
                    'error': 'No matching open trade in ledger',
                })
                continue

            exec_result = await self.execute_exit(exit_decision, open_trade)
            results.append(exec_result)

        return results

    async def execute_decisions(
        self,
        decision: Dict,
        active_capital: float,
        open_positions: List[Dict],
        phase: str,
    ) -> List[Dict]:
        """
        Execute all trades from a decision cycle.

        Args:
            decision: Full decision result from DecisionEngine
            active_capital: Current active capital
            open_positions: Current open positions from TS
            phase: GROWTH or WITHDRAWAL

        Returns:
            List of execution result dicts
        """
        trades = decision.get('trades', [])
        if not trades:
            return []

        action = decision.get('action', 'HOLD')
        if action not in ('ENTER',):
            logger.info(f"Decision action '{action}' — no orders to place")
            return []

        results = []
        current_position_count = len(open_positions)

        for trade in trades:
            if trade.get('action') != 'ENTER':
                continue

            exec_result = await self.execute_trade(
                trade=trade,
                active_capital=active_capital,
                open_position_count=current_position_count,
                phase=phase,
            )
            results.append(exec_result)

            # Update position count if order succeeded
            if exec_result['success']:
                current_position_count += 1

        return results

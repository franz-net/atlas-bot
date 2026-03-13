"""
Weekly withdrawal calculation for Project Atlas.

Activates in Phase 2 when active capital exceeds $2,000.
Calculates 1% of weekly realized profit on Fridays.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.config.constants import PROTECTED_FLOOR
from src.ledger.ledger import TradingLedger
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'withdrawal_tracker')


class WithdrawalTracker:
    """
    Calculates and records weekly profit withdrawals.

    Only runs on Fridays. Only activates when active capital > $2,000 (Phase 2).
    """

    def __init__(self, ledger: TradingLedger):
        """
        Initialize the withdrawal tracker.

        Args:
            ledger: TradingLedger instance for reading trades and recording withdrawals
        """
        self._ledger = ledger
        self._withdrawal_threshold = float(
            __import__('os').getenv('WITHDRAWAL_THRESHOLD', '2000')
        )

    def calculate_weekly_withdrawal(
        self, current_balance: float
    ) -> Optional[Dict]:
        """
        Calculate withdrawal amount for the current week.

        Only runs on Fridays. Returns None if not Friday or not in Phase 2.

        Args:
            current_balance: Current total account balance

        Returns:
            Withdrawal dict or None if not applicable
        """
        now = datetime.now()

        # Only calculate on Fridays
        if now.weekday() != 4:
            logger.info("Not Friday — skipping withdrawal calculation")
            return None

        active_capital = current_balance - PROTECTED_FLOOR
        if active_capital < self._withdrawal_threshold:
            logger.info(
                f"Active capital ${active_capital:.2f} below threshold "
                f"${self._withdrawal_threshold:.2f} — no withdrawal"
            )
            return None

        # Calculate this week's realized profit
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start_str = week_start.strftime('%Y-%m-%d')

        closed_trades = self._ledger.get_closed_trades()
        weekly_profit = sum(
            t['pnl_dollars'] or 0
            for t in closed_trades
            if t.get('exit_timestamp', '') >= week_start_str
        )

        if weekly_profit <= 0:
            logger.info(f"No positive weekly profit (${weekly_profit:.2f}) — no withdrawal")
            return None

        withdrawal_amount = weekly_profit * 0.01
        week_ending = now.strftime('%Y-%m-%d')

        result = {
            'week_ending': week_ending,
            'active_capital': active_capital,
            'weekly_profit': weekly_profit,
            'withdrawal_amount': withdrawal_amount,
        }

        logger.info(
            f"Withdrawal calculated: ${withdrawal_amount:.2f} "
            f"(1% of ${weekly_profit:.2f} weekly profit)"
        )

        return result

    def record_withdrawal(self, withdrawal: Dict) -> bool:
        """
        Record a calculated withdrawal to the ledger.

        Args:
            withdrawal: Withdrawal dict from calculate_weekly_withdrawal

        Returns:
            True if recorded successfully
        """
        mode = __import__('os').getenv('USE_SIM_ACCOUNT', 'true').lower()
        if mode == 'true':
            notes = (
                f"SIM: floor raised by ${withdrawal['withdrawal_amount']:.2f} "
                f"(effective floor = ${PROTECTED_FLOOR} + cumulative withdrawals)"
            )
        else:
            notes = 'LIVE: withdrawal recorded'

        return self._ledger.record_withdrawal(
            week_ending=withdrawal['week_ending'],
            active_capital=withdrawal['active_capital'],
            weekly_profit=withdrawal['weekly_profit'],
            withdrawal_amount=withdrawal['withdrawal_amount'],
            notes=notes,
        )

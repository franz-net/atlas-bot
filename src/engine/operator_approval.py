"""
Operator approval layer for Project Atlas.

Phase A of live transition: every trade requires explicit operator approval
before execution. The system presents the trade details and waits for
a yes/no decision.

This module is only active when OPERATOR_APPROVAL=true in .env.
In sim mode, trades execute without approval.
"""

import logging
import os
from typing import Dict, Optional

from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'operator_approval')


class OperatorApproval:
    """
    Gated trade execution requiring operator confirmation.

    Active when OPERATOR_APPROVAL=true. Presents trade details
    to the operator and blocks until approved or rejected.
    """

    def __init__(self):
        """Initialize the approval layer."""
        self._enabled = os.getenv('OPERATOR_APPROVAL', 'false').lower() == 'true'
        self._approved_count = 0
        self._rejected_count = 0
        self._auto_approve_after = int(os.getenv('AUTO_APPROVE_AFTER', '0'))
        logger.info(
            f"Operator approval: {'ENABLED' if self._enabled else 'DISABLED'}"
            + (f' (auto-approve after {self._auto_approve_after} trades)'
               if self._auto_approve_after > 0 else '')
        )

    @property
    def is_enabled(self) -> bool:
        """Whether operator approval is active."""
        return self._enabled

    @property
    def stats(self) -> Dict:
        """Return approval statistics."""
        return {
            'enabled': self._enabled,
            'approved': self._approved_count,
            'rejected': self._rejected_count,
            'total': self._approved_count + self._rejected_count,
        }

    def request_approval(self, trade: Dict, active_capital: float) -> bool:
        """
        Present a trade to the operator for approval.

        Blocks on stdin until the operator responds. In non-interactive
        environments, defaults to rejection (safe).

        Args:
            trade: Trade dict from decision engine
            active_capital: Current active capital

        Returns:
            True if approved, False if rejected
        """
        if not self._enabled:
            return True

        # Auto-approve after N successful trades (Phase B transition)
        if (self._auto_approve_after > 0
                and self._approved_count >= self._auto_approve_after):
            logger.info(
                f"Auto-approving: {self._approved_count} trades already approved"
            )
            self._approved_count += 1
            return True

        # Display trade details
        symbol = trade.get('symbol', '?')
        direction = trade.get('direction', '?')
        shares = trade.get('shares', 0)
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        reasoning = trade.get('reasoning', 'No reasoning provided')

        print()
        print("=" * 60)
        print("  OPERATOR APPROVAL REQUIRED")
        print("=" * 60)
        print(f"  Symbol:         {symbol}")
        print(f"  Direction:      {direction}")
        print(f"  Shares:         {shares}")
        print(f"  Stop Loss:      ${stop_loss:.2f}")
        print(f"  Take Profit:    ${take_profit:.2f}")
        print(f"  Active Capital: ${active_capital:.2f}")
        print(f"  Reasoning:      {reasoning[:200]}")
        print("=" * 60)

        try:
            response = input("  Approve this trade? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            logger.warning(f"Approval request for {symbol}: no input — defaulting to REJECT")
            self._rejected_count += 1
            return False

        approved = response in ('y', 'yes')

        if approved:
            self._approved_count += 1
            logger.info(f"Trade APPROVED by operator: {direction} {shares} {symbol}")
        else:
            self._rejected_count += 1
            logger.info(f"Trade REJECTED by operator: {direction} {shares} {symbol}")

        return approved

    def format_summary(self) -> str:
        """
        Format approval statistics for display.

        Returns:
            Summary string
        """
        stats = self.stats
        if not stats['enabled']:
            return "Operator approval: DISABLED (all trades auto-execute)"

        total = stats['total']
        if total == 0:
            return "Operator approval: ENABLED (no trades reviewed yet)"

        return (
            f"Operator approval: ENABLED | "
            f"Approved: {stats['approved']} | "
            f"Rejected: {stats['rejected']} | "
            f"Total: {total}"
        )

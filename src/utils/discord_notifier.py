"""
Discord webhook notifications for Project Atlas.

Sends trade alerts, daily summaries, and error notifications to a
Discord channel via webhook. Gracefully no-ops if no webhook URL
is configured.
"""

import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp

from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'discord_notifier')


class DiscordNotifier:
    """
    Send notifications to a Discord webhook.

    All methods are safe to call even if no webhook is configured —
    they return False silently. Failures never raise exceptions or
    interrupt the trading loop.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize the notifier.

        Args:
            webhook_url: Discord webhook URL. If None, reads from
                         DISCORD_WEBHOOK_URL in .env.
        """
        self._webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL', '')
        if self._webhook_url:
            logger.info("Discord notifier enabled")
        else:
            logger.info("Discord notifier disabled (no webhook URL)")

    @property
    def enabled(self) -> bool:
        """Whether notifications are active."""
        return bool(self._webhook_url)

    async def _send(self, embed: dict) -> bool:
        """
        Send an embed to the Discord webhook.

        Args:
            embed: Discord embed dict

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._webhook_url:
            return False

        payload = {"embeds": [embed]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 204):
                        return True
                    logger.warning(
                        f"Discord webhook returned {resp.status}: "
                        f"{await resp.text()}"
                    )
                    return False
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")
            return False

    async def notify_entry(
        self,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        conviction: int,
        mode: str = 'SIM',
    ) -> bool:
        """
        Send a trade entry notification.

        Args:
            symbol: Ticker symbol
            direction: LONG or SHORT
            shares: Number of shares
            entry_price: Fill price
            stop_loss: Stop loss price
            take_profit: Take profit price
            conviction: AI conviction score (1-10)
            mode: SIM or LIVE
        """
        color = 0x2ECC71 if direction == 'LONG' else 0xE74C3C
        notional = shares * entry_price

        embed = {
            "title": f"{'🟢' if direction == 'LONG' else '🔴'} {direction} {symbol}",
            "color": color,
            "fields": [
                {"name": "Shares", "value": str(shares), "inline": True},
                {"name": "Price", "value": f"${entry_price:.2f}", "inline": True},
                {"name": "Notional", "value": f"${notional:.2f}", "inline": True},
                {"name": "Stop Loss", "value": f"${stop_loss:.2f}", "inline": True},
                {"name": "Take Profit", "value": f"${take_profit:.2f}", "inline": True},
                {"name": "Conviction", "value": f"{conviction}/10", "inline": True},
            ],
            "footer": {"text": f"Atlas [{mode}]"},
            "timestamp": datetime.now(ZoneInfo('US/Eastern')).isoformat(),
        }

        return await self._send(embed)

    async def notify_exit(
        self,
        symbol: str,
        direction: str,
        pnl_dollars: float,
        pnl_pct: float,
        exit_reason: str,
        mode: str = 'SIM',
    ) -> bool:
        """
        Send a trade exit notification.

        Args:
            symbol: Ticker symbol
            direction: LONG or SHORT
            pnl_dollars: Realized P&L in dollars
            pnl_pct: Realized P&L as percentage
            exit_reason: Why the trade was closed
            mode: SIM or LIVE
        """
        win = pnl_dollars > 0
        color = 0x2ECC71 if win else 0xE74C3C
        emoji = '✅' if win else '❌'

        embed = {
            "title": f"{emoji} EXIT {symbol} {direction}",
            "color": color,
            "fields": [
                {"name": "P&L", "value": f"${pnl_dollars:+.2f} ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "Reason", "value": exit_reason[:200], "inline": False},
            ],
            "footer": {"text": f"Atlas [{mode}]"},
            "timestamp": datetime.now(ZoneInfo('US/Eastern')).isoformat(),
        }

        return await self._send(embed)

    async def notify_daily_summary(
        self,
        date: str,
        trades_today: int,
        daily_pnl: float,
        total_pnl: float,
        balance: float,
        active_capital: float,
        mode: str = 'SIM',
    ) -> bool:
        """
        Send an end-of-day summary notification.

        Args:
            date: Date string (YYYY-MM-DD)
            trades_today: Number of trades closed today
            daily_pnl: Today's realized P&L
            total_pnl: All-time realized P&L
            balance: Current account balance
            active_capital: Current active capital
            mode: SIM or LIVE
        """
        color = 0x2ECC71 if daily_pnl >= 0 else 0xE74C3C

        embed = {
            "title": f"📊 Daily Summary — {date}",
            "color": color,
            "fields": [
                {"name": "Trades Today", "value": str(trades_today), "inline": True},
                {"name": "Daily P&L", "value": f"${daily_pnl:+.2f}", "inline": True},
                {"name": "Total P&L", "value": f"${total_pnl:+.2f}", "inline": True},
                {"name": "Balance", "value": f"${balance:,.2f}", "inline": True},
                {"name": "Active Capital", "value": f"${active_capital:,.2f}", "inline": True},
            ],
            "footer": {"text": f"Atlas [{mode}]"},
            "timestamp": datetime.now(ZoneInfo('US/Eastern')).isoformat(),
        }

        return await self._send(embed)

    async def notify_error(
        self,
        event_type: str,
        message: str,
        mode: str = 'SIM',
    ) -> bool:
        """
        Send an error or alert notification.

        Args:
            event_type: Category (HARD_STOP, API_ERROR, REVIEW_FAILED, etc.)
            message: Error details
            mode: SIM or LIVE
        """
        embed = {
            "title": f"⚠️ {event_type}",
            "color": 0xE67E22,
            "description": message[:2000],
            "footer": {"text": f"Atlas [{mode}]"},
            "timestamp": datetime.now(ZoneInfo('US/Eastern')).isoformat(),
        }

        return await self._send(embed)

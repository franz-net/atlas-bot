"""
Pre-flight checks for Project Atlas.

Validates system readiness before starting the trading loop.
Provides sim validation report for live graduation assessment.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.config.constants import (
    MAX_CONCURRENT_POSITIONS,
    PROTECTED_FLOOR,
)
from src.ledger.ledger import TradingLedger
from src.utils.logging_config import setup_file_logger

logger = setup_file_logger(__name__, 'preflight')

# Sim validation thresholds (SPEC Section 10.1)
MIN_SIM_TRADING_DAYS = 30
MIN_WIN_RATE = 45.0
MAX_SINGLE_DAY_DRAWDOWN_PCT = 15.0


class PreflightCheck:
    """
    System readiness and sim validation checks.

    Run before starting the trading loop or before considering
    live graduation.
    """

    def __init__(self, ledger: TradingLedger):
        """
        Initialize with a ledger for reading trade history.

        Args:
            ledger: TradingLedger instance
        """
        self._ledger = ledger

    def run_startup_checks(self) -> Dict:
        """
        Run pre-flight checks before starting the trading loop.

        Validates environment, configuration, and account readiness.

        Returns:
            Dict with passed (bool), checks (list of results), and warnings
        """
        checks = []
        warnings = []

        # 1. Environment variables
        required_env = ['TS_API_KEY', 'TS_API_SECRET', 'TS_ACCOUNT_ID']
        for var in required_env:
            val = os.getenv(var)
            checks.append({
                'name': f'ENV: {var}',
                'passed': bool(val),
                'detail': 'Set' if val else 'MISSING',
            })

        # 2. AI provider config
        provider = os.getenv('DECISION_PROVIDER', 'claude')
        checks.append({
            'name': 'Decision provider',
            'passed': provider in ('claude', 'gemini'),
            'detail': provider,
        })

        if provider == 'claude':
            key = os.getenv('ANTHROPIC_API_KEY')
            model = os.getenv('CLAUDE_DECISION_MODEL')
            checks.append({
                'name': 'ANTHROPIC_API_KEY',
                'passed': bool(key),
                'detail': 'Set' if key else 'MISSING',
            })
            checks.append({
                'name': 'CLAUDE_DECISION_MODEL',
                'passed': bool(model),
                'detail': model or 'MISSING',
            })
        elif provider == 'gemini':
            key = os.getenv('GEMINI_API_KEY')
            model = os.getenv('GEMINI_DECISION_MODEL')
            checks.append({
                'name': 'GEMINI_API_KEY',
                'passed': bool(key),
                'detail': 'Set' if key else 'MISSING',
            })
            checks.append({
                'name': 'GEMINI_DECISION_MODEL',
                'passed': bool(model),
                'detail': model or 'MISSING',
            })

        # 3. Sim mode check
        use_sim = os.getenv('USE_SIM_ACCOUNT', 'true').lower() == 'true'
        checks.append({
            'name': 'Sim mode',
            'passed': True,  # Both sim and live are valid
            'detail': 'SIM' if use_sim else 'LIVE',
        })
        if not use_sim:
            warnings.append(
                'LIVE MODE ACTIVE — trades will use real capital. '
                'Ensure sim validation criteria are met.'
            )

        # 4. Database accessible
        try:
            summary = self._ledger.get_summary()
            checks.append({
                'name': 'Ledger database',
                'passed': True,
                'detail': f'{summary["total_trades"]} trades recorded',
            })
        except Exception as e:
            checks.append({
                'name': 'Ledger database',
                'passed': False,
                'detail': str(e),
            })

        all_passed = all(c['passed'] for c in checks)

        return {
            'passed': all_passed,
            'checks': checks,
            'warnings': warnings,
            'sim_mode': use_sim,
        }

    def run_sim_validation(self) -> Dict:
        """
        Validate sim trading record against graduation criteria.

        SPEC Section 10.1:
        - Minimum 30 trading days
        - Positive net P&L
        - Win rate > 45%
        - No single day > 15% active capital drawdown
        - All trades have complete ledger entries

        Returns:
            Validation report dict
        """
        summary = self._ledger.get_summary()
        closed_trades = self._ledger.get_closed_trades()

        criteria = []

        # 1. Trading days count
        trading_days = self._count_trading_days(closed_trades)
        criteria.append({
            'name': f'Minimum {MIN_SIM_TRADING_DAYS} trading days',
            'passed': trading_days >= MIN_SIM_TRADING_DAYS,
            'detail': f'{trading_days} days',
        })

        # 2. Positive net P&L
        criteria.append({
            'name': 'Positive net P&L',
            'passed': summary['total_pnl'] > 0,
            'detail': f"${summary['total_pnl']:.2f}",
        })

        # 3. Win rate
        criteria.append({
            'name': f'Win rate > {MIN_WIN_RATE}%',
            'passed': summary['win_rate'] > MIN_WIN_RATE,
            'detail': f"{summary['win_rate']:.1f}%",
        })

        # 4. Max single-day drawdown
        max_drawdown = self._calculate_max_daily_drawdown(closed_trades)
        criteria.append({
            'name': f'No day > {MAX_SINGLE_DAY_DRAWDOWN_PCT}% drawdown',
            'passed': max_drawdown <= MAX_SINGLE_DAY_DRAWDOWN_PCT,
            'detail': f'{max_drawdown:.1f}% worst day',
        })

        # 5. Complete ledger entries
        incomplete = self._count_incomplete_trades(closed_trades)
        criteria.append({
            'name': 'All trades have complete entries',
            'passed': incomplete == 0,
            'detail': f'{incomplete} incomplete' if incomplete else 'All complete',
        })

        all_passed = all(c['passed'] for c in criteria)

        return {
            'ready_for_live': all_passed,
            'criteria': criteria,
            'summary': {
                'total_trades': summary['total_trades'],
                'trading_days': trading_days,
                'win_rate': summary['win_rate'],
                'total_pnl': summary['total_pnl'],
                'max_daily_drawdown': max_drawdown,
                'total_api_cost': summary['total_api_cost'],
            },
        }

    def _count_trading_days(self, trades: List[Dict]) -> int:
        """Count unique trading days from closed trades."""
        days = set()
        for t in trades:
            ts = t.get('entry_timestamp', '')
            if ts:
                days.add(ts[:10])  # YYYY-MM-DD
        return len(days)

    def _calculate_max_daily_drawdown(self, trades: List[Dict]) -> float:
        """
        Calculate the worst single-day drawdown as % of active capital.

        Returns:
            Max daily drawdown percentage (positive number)
        """
        daily_pnl = {}
        daily_capital = {}

        for t in trades:
            ts = t.get('exit_timestamp', '')
            if not ts:
                continue
            date = ts[:10]
            pnl = t.get('pnl_dollars', 0) or 0
            capital = t.get('active_capital_at_entry', 1000) or 1000

            daily_pnl[date] = daily_pnl.get(date, 0) + pnl
            if date not in daily_capital:
                daily_capital[date] = capital

        if not daily_pnl:
            return 0.0

        max_drawdown = 0.0
        for date, pnl in daily_pnl.items():
            if pnl < 0:
                capital = daily_capital.get(date, 1000)
                drawdown_pct = abs(pnl) / capital * 100 if capital > 0 else 0
                max_drawdown = max(max_drawdown, drawdown_pct)

        return max_drawdown

    def _count_incomplete_trades(self, trades: List[Dict]) -> int:
        """Count trades missing required fields."""
        incomplete = 0
        for t in trades:
            if not t.get('entry_reasoning'):
                incomplete += 1
            elif not t.get('exit_reasoning'):
                incomplete += 1
            elif t.get('pnl_dollars') is None:
                incomplete += 1
        return incomplete

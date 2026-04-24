"""
End-of-day review for Project Atlas.

Uses Claude Opus to analyze the day's trading activity, identify patterns,
and suggest strategy adjustments. Always uses Claude — never Gemini.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.engine.providers.claude import ClaudeProvider
from src.ledger.ledger import TradingLedger
from src.utils.logging_config import setup_file_logger
from strategy.prompts import REVIEW_SYSTEM_PROMPT

logger = setup_file_logger(__name__, 'eod_review')


class EODReview:
    """
    End-of-day portfolio review using Claude Opus.

    Always uses Claude Opus regardless of DECISION_PROVIDER setting.
    Produces actionable analysis of the day's trading.
    """

    def __init__(self, ledger: TradingLedger):
        """
        Initialize the EOD review engine.

        Args:
            ledger: TradingLedger for reading trade data
        """
        self._ledger = ledger
        review_model = os.getenv('CLAUDE_REVIEW_MODEL', 'claude-opus-4-20250514')
        self._provider = ClaudeProvider(model_override=review_model)
        logger.info(f"EOD review initialized: model={review_model}")

    def _build_review_data(self, date: str = None) -> Dict:
        """
        Assemble the week's trading data for Opus review.

        Pulls all trades from Monday through the given date (typically Friday).

        Args:
            date: Date string YYYY-MM-DD (defaults to today)

        Returns:
            Review data dict
        """
        if date is None:
            est = ZoneInfo('US/Eastern')
            date = datetime.now(est).strftime('%Y-%m-%d')

        # Calculate Monday of this week
        from datetime import timedelta
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        monday = date_obj - timedelta(days=date_obj.weekday())
        monday_str = monday.strftime('%Y-%m-%d')

        # Get this week's closed trades
        all_closed = self._ledger.get_closed_trades()
        todays_closed = [
            t for t in all_closed
            if monday_str <= (t.get('exit_timestamp', '')[:10] or '') <= date
        ]

        # Get this week's entries (may still be open)
        all_trades = self._ledger.get_all_trades()
        todays_entries = [
            t for t in all_trades
            if monday_str <= (t.get('entry_timestamp', '')[:10] or '') <= date
        ]

        # Get open trades
        open_trades = self._ledger.get_open_trades()

        # Get this week's cycles
        recent_cycles = self._ledger.get_recent_cycles(limit=500)
        todays_cycles = [
            c for c in recent_cycles
            if monday_str <= (c.get('timestamp', '')[:10] or '') <= date
        ]

        # Summary stats
        summary = self._ledger.get_summary()
        daily_cost = sum(
            self._ledger.get_daily_api_cost(
                (monday + timedelta(days=d)).strftime('%Y-%m-%d')
            )
            for d in range(5)
        )

        # Near-miss shadow data: which gates rejected candidates this week,
        # and did those rejections move in the direction the trade would
        # have taken?
        near_miss_rollup = self._ledger.get_near_miss_rollup(monday_str, date)
        near_miss_summary = []
        for row in near_miss_rollup:
            total = row['total'] or 0
            backfilled = row['backfilled'] or 0
            wins = row['directional_wins_1d'] or 0
            near_miss_summary.append({
                'gate': row['gate'],
                'would_be_direction': row['would_be_direction'],
                'rejections': total,
                'backfilled': backfilled,
                'mean_1d_return_pct': round((row['mean_1d'] or 0) * 100, 3),
                'mean_3d_return_pct': round((row['mean_3d'] or 0) * 100, 3),
                'directional_hit_rate_1d': (
                    round(wins / backfilled * 100, 1)
                    if backfilled else None
                ),
            })

        return {
            'week_ending': date,
            'week_starting': monday_str,
            'closed_trades_this_week': [
                {
                    'symbol': t['symbol'],
                    'direction': t['direction'],
                    'shares': t['shares'],
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'pnl_dollars': t['pnl_dollars'],
                    'pnl_pct': t['pnl_pct'],
                    'entry_reasoning': t['entry_reasoning'],
                    'exit_reasoning': t['exit_reasoning'],
                    'news_catalyst': t['news_catalyst'],
                    'stop_loss_price': t['stop_loss_price'],
                    'take_profit_price': t['take_profit_price'],
                }
                for t in todays_closed
            ],
            'entries_this_week': [
                {
                    'symbol': t['symbol'],
                    'direction': t['direction'],
                    'shares': t['shares'],
                    'entry_price': t['entry_price'],
                    'entry_reasoning': t['entry_reasoning'],
                    'status': t['status'],
                }
                for t in todays_entries
            ],
            'open_positions': [
                {
                    'symbol': t['symbol'],
                    'direction': t['direction'],
                    'shares': t['shares'],
                    'entry_price': t['entry_price'],
                    'stop_loss_price': t['stop_loss_price'],
                    'take_profit_price': t['take_profit_price'],
                }
                for t in open_trades
            ],
            'cycles_this_week': len(todays_cycles),
            'holds_this_week': sum(
                1 for c in todays_cycles
                if c.get('action_taken') == 'HOLD'
            ),
            'api_cost_this_week': daily_cost,
            'near_miss_rollup': near_miss_summary,
            'overall_summary': {
                'total_trades': summary['total_trades'],
                'win_rate': summary['win_rate'],
                'total_pnl': summary['total_pnl'],
                'total_api_cost': summary['total_api_cost'],
            },
        }

    async def run_review(self, date: str = None) -> Dict:
        """
        Run the end-of-day review.

        Args:
            date: Date to review (defaults to today)

        Returns:
            Review result dict with parsed analysis
        """
        review_data = self._build_review_data(date)

        logger.info(
            f"Running weekly review for {review_data['week_starting']} to "
            f"{review_data['week_ending']}: "
            f"{len(review_data['closed_trades_this_week'])} closed, "
            f"{len(review_data['entries_this_week'])} entries, "
            f"{review_data['cycles_this_week']} cycles"
        )

        # No trades this week — skip Opus call
        if (not review_data['closed_trades_this_week']
                and not review_data['entries_this_week']
                and not review_data['open_positions']):
            logger.info("No trading activity this week — skipping Opus review")
            return {
                'success': True,
                'date': review_data['week_ending'],
                'review': None,
                'skipped': True,
                'reason': 'No trading activity this week',
                'cost_estimate': 0,
            }

        user_prompt = json.dumps(review_data, indent=2, default=str)

        response = await self._provider.review(
            REVIEW_SYSTEM_PROMPT, user_prompt
        )

        result = {
            'success': response.get('success', False),
            'date': review_data['week_ending'],
            'review': None,
            'skipped': False,
            'reason': None,
            'cost_estimate': response.get('cost_estimate', 0),
            'prompt_tokens': response.get('prompt_tokens', 0),
            'completion_tokens': response.get('completion_tokens', 0),
        }

        if response.get('success'):
            parsed = self._parse_review(response.get('content', ''))
            result['review'] = parsed
            if parsed:
                logger.info(
                    f"EOD review complete: grade={parsed.get('overall_grade', '?')} "
                    f"trades={len(parsed.get('trades_reviewed', []))} "
                    f"recommendations={len(parsed.get('recommendations', []))}"
                )
            else:
                result['reason'] = 'Failed to parse review response'
                logger.error("EOD review returned unparseable response")
        else:
            result['reason'] = response.get('error', 'Provider call failed')
            logger.error(f"EOD review failed: {result['reason']}")

        return result

    def _parse_review(self, content: str) -> Optional[Dict]:
        """
        Parse JSON from Opus review response.

        Args:
            content: Raw response text

        Returns:
            Parsed review dict or None
        """
        if not content:
            return None

        cleaned = content.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Review JSON parse error: {e}")
            logger.error(f"Raw content: {content[:500]}")
            return None

    def format_review(self, review: Dict) -> str:
        """
        Format a review dict into human-readable text for CLI output.

        Args:
            review: Parsed review dict from Opus

        Returns:
            Formatted string
        """
        if not review:
            return "No review data available."

        lines = []
        lines.append(f"Date: {review.get('date', 'Unknown')}")
        lines.append(f"Grade: {review.get('overall_grade', 'N/A')}")
        lines.append(f"\nSummary: {review.get('summary', 'N/A')}")

        trades = review.get('trades_reviewed', [])
        if trades:
            lines.append(f"\nTrades Reviewed ({len(trades)}):")
            for t in trades:
                pnl = t.get('pnl', 0)
                lines.append(
                    f"  {t.get('symbol', '?')} — {t.get('result', '?')} "
                    f"${pnl:.2f} | Entry: {t.get('entry_quality', '?')} "
                    f"Exit: {t.get('exit_quality', '?')}"
                )
                if t.get('notes'):
                    lines.append(f"    {t['notes']}")

        patterns = review.get('patterns_identified', [])
        if patterns:
            lines.append(f"\nPatterns:")
            for p in patterns:
                lines.append(f"  - {p}")

        risk = review.get('risk_assessment', {})
        if risk:
            lines.append(f"\nRisk Assessment:")
            lines.append(f"  Rules followed: {'Yes' if risk.get('rules_followed') else 'NO'}")
            lines.append(f"  Sizing quality: {risk.get('position_sizing_quality', '?')}")
            violations = risk.get('violations', [])
            if violations:
                for v in violations:
                    lines.append(f"  VIOLATION: {v}")

        recs = review.get('recommendations', [])
        if recs:
            lines.append(f"\nRecommendations:")
            for r in recs:
                lines.append(f"  - {r}")

        wl = review.get('watchlist_suggestions', {})
        if wl and (wl.get('add') or wl.get('remove')):
            lines.append(f"\nWatchlist Suggestions:")
            if wl.get('add'):
                lines.append(f"  Add: {', '.join(wl['add'])}")
            if wl.get('remove'):
                lines.append(f"  Remove: {', '.join(wl['remove'])}")
            if wl.get('reasoning'):
                lines.append(f"  Reason: {wl['reasoning']}")

        return '\n'.join(lines)

"""
Weekly watchlist rotation for Project Atlas.

Uses Claude Opus to analyze which symbols should be added/removed from the
watchlist based on screening pass/fail data and market conditions. Runs
every Friday after close alongside the weekly review.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.engine.providers.claude import ClaudeProvider
from src.ledger.ledger import TradingLedger
from src.screener.screener import StockScreener
from src.utils.logging_config import setup_file_logger
from strategy.prompts import ROTATION_SYSTEM_PROMPT

logger = setup_file_logger(__name__, 'watchlist_rotation')

MAX_ADDITIONS_PER_WEEK = 10
MAX_REMOVALS_PER_WEEK = 10


class WatchlistRotation:
    """
    Weekly watchlist rotation using Claude Opus.

    Analyzes screening pass/fail data and recommends symbol additions/removals.
    Always uses Claude Opus regardless of DECISION_PROVIDER setting.
    """

    def __init__(
        self,
        ledger: TradingLedger,
        screener: StockScreener,
    ):
        """
        Initialize the watchlist rotation engine.

        Args:
            ledger: TradingLedger for reading trade data
            screener: StockScreener for cold list stats
        """
        self._ledger = ledger
        self._screener = screener
        self._watchlist_path = Path('data/watchlist.json')
        review_model = os.getenv('CLAUDE_REVIEW_MODEL', 'claude-opus-4-20250514')
        self._provider = ClaudeProvider(model_override=review_model)
        logger.info(f"Watchlist rotation initialized: model={review_model}")

    def _load_watchlist(self) -> Dict:
        """
        Load the current watchlist file.

        Returns:
            Full watchlist dict with version, notes, and symbols list
        """
        try:
            with open(self._watchlist_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")
            return {'version': '1.0', 'symbols': []}

    def _save_watchlist(self, watchlist: Dict) -> bool:
        """
        Save the watchlist file atomically.

        Args:
            watchlist: Full watchlist dict to write

        Returns:
            True if write succeeded
        """
        try:
            watchlist['updated'] = datetime.now().strftime('%Y-%m-%d')
            # Write to temp file first, then rename (atomic on same filesystem)
            tmp_path = self._watchlist_path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(watchlist, f, indent=2)
                f.write('\n')
            tmp_path.rename(self._watchlist_path)
            logger.info(f"Watchlist saved: {len(watchlist.get('symbols', []))} symbols")
            return True
        except Exception as e:
            logger.error(f"Failed to save watchlist: {e}")
            return False

    def _build_rotation_data(self, week_ending: str) -> Dict:
        """
        Assemble data for Opus to make watchlist rotation decisions.

        Args:
            week_ending: Friday date (YYYY-MM-DD)

        Returns:
            Rotation data dict
        """
        watchlist = self._load_watchlist()
        current_symbols = [
            {'symbol': s['symbol'], 'sector': s.get('sector', 'Unknown')}
            for s in watchlist.get('symbols', [])
        ]

        # Cold list stats from screener
        cold_stats = self._screener.get_cold_stats()
        high_failure_symbols = {
            sym: count for sym, count in cold_stats.items()
            if count >= 6  # Failed at least 6 consecutive cycles
        }

        # Trades this week from ledger
        date_obj = datetime.strptime(week_ending, '%Y-%m-%d')
        monday = date_obj - timedelta(days=date_obj.weekday())
        monday_str = monday.strftime('%Y-%m-%d')

        all_trades = self._ledger.get_all_trades()
        week_trades = [
            {
                'symbol': t['symbol'],
                'direction': t['direction'],
                'pnl_dollars': t.get('pnl_dollars'),
                'status': t['status'],
            }
            for t in all_trades
            if monday_str <= (t.get('entry_timestamp', '')[:10] or '') <= week_ending
        ]

        # Open positions (cannot remove these symbols)
        open_trades = self._ledger.get_open_trades()
        open_symbols = list({t['symbol'] for t in open_trades})

        # Sector distribution
        sector_counts: Dict[str, int] = {}
        for s in current_symbols:
            sector = s['sector']
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return {
            'week_ending': week_ending,
            'current_watchlist': current_symbols,
            'watchlist_size': len(current_symbols),
            'target_size_range': [130, 160],
            'sector_distribution': sector_counts,
            'high_failure_symbols': high_failure_symbols,
            'trades_this_week': week_trades,
            'open_positions': open_symbols,
        }

    def _validate_diff(
        self,
        diff: Dict,
        current_symbols: List[str],
        open_symbols: List[str],
    ) -> Dict:
        """
        Sanitize and validate the Opus rotation diff.

        Enforces caps, prevents removing open positions, and deduplicates.

        Args:
            diff: Raw diff from Opus
            current_symbols: Current watchlist symbols
            open_symbols: Symbols with open positions (cannot remove)

        Returns:
            Cleaned diff dict
        """
        current_set = set(current_symbols)
        open_set = set(open_symbols)

        # Validate removals
        removals = diff.get('remove', [])
        if not isinstance(removals, list):
            removals = []
        valid_removals = []
        for sym in removals:
            if not isinstance(sym, str):
                continue
            sym = sym.upper().strip()
            if sym not in current_set:
                logger.warning(f"Rotation: cannot remove {sym} — not in watchlist")
                continue
            if sym in open_set:
                logger.warning(f"Rotation: cannot remove {sym} — has open position")
                continue
            valid_removals.append(sym)
        valid_removals = valid_removals[:MAX_REMOVALS_PER_WEEK]

        # Validate additions
        additions = diff.get('add', [])
        if not isinstance(additions, list):
            additions = []
        valid_additions = []
        for item in additions:
            if isinstance(item, str):
                item = {'symbol': item.upper().strip(), 'sector': 'Unknown'}
            if not isinstance(item, dict) or 'symbol' not in item:
                continue
            sym = item['symbol'].upper().strip()
            if sym in current_set and sym not in valid_removals:
                logger.warning(f"Rotation: cannot add {sym} — already in watchlist")
                continue
            valid_additions.append({
                'symbol': sym,
                'sector': item.get('sector', 'Unknown'),
            })
        valid_additions = valid_additions[:MAX_ADDITIONS_PER_WEEK]

        return {
            'remove': valid_removals,
            'add': valid_additions,
            'reasoning': diff.get('reasoning', {}),
            'keep_on_watchlist': diff.get('keep_on_watchlist', []),
            'notes': diff.get('notes', ''),
        }

    def _apply_diff(self, diff: Dict) -> bool:
        """
        Apply the validated diff to the watchlist file.

        Args:
            diff: Validated diff with 'remove' and 'add' lists

        Returns:
            True if applied successfully
        """
        watchlist = self._load_watchlist()
        symbols = watchlist.get('symbols', [])

        removal_set = set(diff.get('remove', []))
        if removal_set:
            symbols = [s for s in symbols if s['symbol'] not in removal_set]
            logger.info(f"Rotation: removed {len(removal_set)} symbols: {sorted(removal_set)}")

        additions = diff.get('add', [])
        for item in additions:
            symbols.append({
                'symbol': item['symbol'],
                'sector': item.get('sector', 'Unknown'),
            })
        if additions:
            added_syms = [a['symbol'] for a in additions]
            logger.info(f"Rotation: added {len(additions)} symbols: {sorted(added_syms)}")

        watchlist['symbols'] = symbols
        watchlist['notes'] = (
            f"Updated by weekly rotation on {datetime.now().strftime('%Y-%m-%d')}. "
            f"{diff.get('notes', '')}"
        )

        return self._save_watchlist(watchlist)

    async def run_rotation(self, week_ending: str) -> Dict:
        """
        Run the weekly watchlist rotation.

        Args:
            week_ending: Friday date (YYYY-MM-DD)

        Returns:
            Result dict with success status, changes made, and cost
        """
        rotation_data = self._build_rotation_data(week_ending)

        # Skip if no cold stats yet (first day of running)
        if not rotation_data['high_failure_symbols'] and not rotation_data['trades_this_week']:
            logger.info("Rotation: no screening data yet — skipping")
            return {
                'success': True,
                'skipped': True,
                'reason': 'No screening data accumulated yet',
                'added': [],
                'removed': [],
                'cost_estimate': 0,
            }

        user_prompt = json.dumps(rotation_data, indent=2, default=str)

        logger.info(
            f"Running watchlist rotation: {rotation_data['watchlist_size']} symbols, "
            f"{len(rotation_data['high_failure_symbols'])} high-failure, "
            f"{len(rotation_data['trades_this_week'])} trades this week"
        )

        response = await self._provider.review(
            ROTATION_SYSTEM_PROMPT, user_prompt
        )

        result = {
            'success': False,
            'skipped': False,
            'reason': None,
            'added': [],
            'removed': [],
            'cost_estimate': response.get('cost_estimate', 0),
            'prompt_tokens': response.get('prompt_tokens', 0),
            'completion_tokens': response.get('completion_tokens', 0),
        }

        if not response.get('success'):
            result['reason'] = response.get('error', 'Provider call failed')
            logger.error(f"Rotation failed: {result['reason']}")
            return result

        # Parse JSON response
        parsed = self._parse_response(response.get('content', ''))
        if parsed is None:
            result['reason'] = 'Failed to parse JSON from Opus response'
            logger.error("Rotation: malformed JSON response")
            return result

        # Validate diff
        current_symbols = [s['symbol'] for s in rotation_data['current_watchlist']]
        open_symbols = rotation_data['open_positions']
        validated = self._validate_diff(parsed, current_symbols, open_symbols)

        # Apply diff if there are any changes
        if validated['remove'] or validated['add']:
            applied = self._apply_diff(validated)
            if not applied:
                result['reason'] = 'Failed to write watchlist file'
                return result

        result['success'] = True
        result['added'] = [a['symbol'] for a in validated['add']]
        result['removed'] = validated['remove']

        # Record to ledger
        reasoning_parts = []
        for sym, reason in validated.get('reasoning', {}).items():
            reasoning_parts.append(f"{sym}: {reason}")
        reasoning_text = '; '.join(reasoning_parts) if reasoning_parts else validated.get('notes', '')

        self._ledger.record_watchlist_change(
            week_ending=week_ending,
            symbols_added=json.dumps(result['added']),
            symbols_removed=json.dumps(result['removed']),
            reasoning=reasoning_text,
            full_response=response.get('content', ''),
            prompt_tokens=result['prompt_tokens'],
            completion_tokens=result['completion_tokens'],
            cost_estimate=result['cost_estimate'],
        )

        logger.info(
            f"Rotation complete: +{len(result['added'])} -{len(result['removed'])} "
            f"symbols | Cost ${result['cost_estimate']:.4f}"
        )

        return result

    def _parse_response(self, content: str) -> Optional[Dict]:
        """
        Parse JSON from Opus rotation response.

        Args:
            content: Raw response text

        Returns:
            Parsed dict or None
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
            logger.error(f"Rotation JSON parse error: {e}")
            logger.error(f"Raw content: {content[:500]}")
            return None

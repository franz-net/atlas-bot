"""
Claude/Gemini decision engine for Project Atlas.

Takes screened candidate packages and asks the AI provider for trading
decisions. Never imports AI SDKs directly — all calls go through the
provider abstraction in src/engine/providers/.

Share sizing is calculated in Python, not by the LLM. The AI only
provides expected_entry_price, stop_loss, take_profit, and reasoning.
"""

import json
import math
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from src.config.constants import (
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    PROTECTED_FLOOR,
)
from src.engine.providers import DecisionProvider, create_provider
from src.engine.schemas import AIResponse, ExitDecision, TradeDecision
from src.utils.logging_config import setup_file_logger
from strategy.prompts import DECISION_SYSTEM_PROMPT as SYSTEM_PROMPT
from strategy.thresholds import (
    ATR_SL_MULTIPLIER,
    LONG_MIN_RSI,
    LONG_REVERSAL_MAX_RSI,
    MAX_SL_DISTANCE_PCT,
    MAX_TP_DISTANCE_PCT,
    MIN_SL_DISTANCE_PCT,
    MIN_TP_DISTANCE_PCT,
    SHORT_MAX_RSI,
    TARGET_POSITION_PCT,
    TARGET_RR_RATIO,
)

logger = setup_file_logger(__name__, 'decision_engine')


class DecisionEngine:
    """
    AI-powered trading decision engine.

    Uses the provider abstraction to call Claude or Gemini for trading
    decisions. Never imports AI SDKs directly. Share sizing is calculated
    in Python based on the AI's expected_entry_price and stop_loss.
    """

    def __init__(
        self,
        provider: Optional[DecisionProvider] = None,
        ledger=None,
    ):
        """
        Initialize the decision engine.

        Args:
            provider: Optional pre-configured provider. If None, creates one
                      from DECISION_PROVIDER env var.
            ledger: Optional TradingLedger for near-miss shadow logging when
                    AI-proposed trades are rejected by Python gates.
        """
        if provider:
            self._provider = provider
        else:
            provider_name = os.getenv('DECISION_PROVIDER', 'claude')
            self._provider = create_provider(provider_name)
        self._ledger = ledger

        logger.info(
            f"Decision engine initialized: provider={self._provider.provider_name} "
            f"model={self._provider.model_name}"
        )

    def _log_near_miss(
        self,
        cycle_id: str,
        symbol: str,
        gate: str,
        gate_detail: str,
        entry_price: Optional[float] = None,
        would_be_direction: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ) -> None:
        """Log a near-miss when an AI-proposed trade is rejected post-hoc."""
        if self._ledger is None:
            return
        try:
            self._ledger.record_near_miss(
                symbol=symbol,
                gate=gate,
                gate_detail=gate_detail,
                entry_price=entry_price,
                would_be_direction=would_be_direction,
                metrics=metrics,
                cycle_id=cycle_id,
            )
        except Exception as e:
            logger.warning(f"Near-miss log failed for {symbol}/{gate}: {e}")

    async def decide(
        self,
        candidates: List[Dict],
        account_state: Dict,
        open_positions: List[Dict],
        recent_symbols: Optional[List[str]] = None,
        ledger_open_trades: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Make a trading decision based on candidate packages and account state.

        Args:
            candidates: List of candidate packages from CandidateBuilder
            account_state: Current account info (balance, active_capital, phase)
            open_positions: List of currently open positions
            recent_symbols: Symbols to exclude due to cooldown (recently traded)

        Returns:
            Decision dict with cycle_id, parsed action, raw response, and usage stats
        """
        cycle_id = str(uuid.uuid4())
        logger.info(f"Decision cycle {cycle_id}: {len(candidates)} candidates")

        # No candidates AND no open positions = automatic HOLD, skip AI call
        if not candidates and not open_positions:
            logger.info(f"Cycle {cycle_id}: no candidates, no positions — skipping AI call")
            return {
                'cycle_id': cycle_id,
                'timestamp': datetime.now().isoformat(),
                'candidates_evaluated': 0,
                'provider': self._provider.provider_name,
                'model': self._provider.model_name,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cost_estimate': 0,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'raw_response': '',
                'action': 'HOLD',
                'trades': [],
                'exits': [],
                'cycle_notes': 'No candidates passed screening filters.',
                'parse_error': None,
            }

        # Build prompts
        system_prompt = self._build_system_prompt(account_state)
        user_prompt = self._build_user_prompt(
            candidates, account_state, open_positions,
            ledger_open_trades=ledger_open_trades,
        )

        # Call provider
        response = await self._provider.decide(system_prompt, user_prompt)

        # Parse and validate
        active_capital = account_state.get('active_capital', 1000)
        result = self._build_cycle_result(
            cycle_id, candidates, response, active_capital,
            recent_symbols=recent_symbols,
        )

        logger.info(
            f"Cycle {cycle_id}: action={result['action']} "
            f"trades={len(result.get('trades', []))} "
            f"cost=${result['cost_estimate']:.4f}"
        )

        return result

    def _build_system_prompt(self, account_state: Dict) -> str:
        """
        Build the system prompt with current risk parameters.

        Args:
            account_state: Current account state

        Returns:
            Formatted system prompt string
        """
        active_capital = account_state.get('active_capital', 1000)
        return SYSTEM_PROMPT.format(
            max_positions=MAX_CONCURRENT_POSITIONS,
            max_position_pct=int(MAX_POSITION_SIZE_PCT * 100),
            max_position_dollars=active_capital * MAX_POSITION_SIZE_PCT,
            active_capital=active_capital,
        )

    def _build_user_prompt(
        self,
        candidates: List[Dict],
        account_state: Dict,
        open_positions: List[Dict],
        ledger_open_trades: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build the user prompt with account state and candidate data.

        Merges broker position data with ledger trade data to give
        the AI full context for exit decisions.

        Args:
            candidates: Candidate packages
            account_state: Account info
            open_positions: Current positions from broker
            ledger_open_trades: Open trades from ledger (for exit context)

        Returns:
            Formatted user prompt string
        """
        est = ZoneInfo('US/Eastern')
        now = datetime.now(est)

        from src.screener.candidate_builder import CandidateBuilder

        # Build enriched open_positions by merging broker + ledger data
        ledger_by_symbol = {}
        if ledger_open_trades:
            for t in ledger_open_trades:
                ledger_by_symbol[t.get('symbol', '')] = t

        enriched_positions = []
        for p in open_positions:
            symbol = p.get('Symbol', '')
            ledger_trade = ledger_by_symbol.get(symbol, {})

            position = {
                'symbol': symbol,
                'direction': ledger_trade.get('direction', 'LONG'),
                'shares': int(p.get('Quantity', 0)),
                'entry_price': ledger_trade.get('entry_price', p.get('AveragePrice', 0)),
                'current_price': float(p.get('AveragePrice', 0)),
                'unrealized_pnl': float(p.get('UnrealizedProfitLoss', 0)),
                'stop_loss': ledger_trade.get('stop_loss_price', 0),
                'take_profit': ledger_trade.get('take_profit_price', 0),
                'entry_reasoning': ledger_trade.get('entry_reasoning', ''),
                'entry_timestamp': ledger_trade.get('entry_timestamp', ''),
            }

            # Compute unrealized P&L percentage
            entry = position['entry_price']
            if entry > 0:
                pnl = position['unrealized_pnl']
                position_value = entry * position['shares']
                position['unrealized_pnl_pct'] = round(
                    (pnl / position_value) * 100, 2
                ) if position_value > 0 else 0

            # Compute bars_held (cycles since entry, ~5 min each)
            entry_ts = ledger_trade.get('entry_timestamp', '')
            if entry_ts:
                try:
                    entry_dt = datetime.fromisoformat(entry_ts)
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=est)
                    elapsed_minutes = (now - entry_dt).total_seconds() / 60
                    position['bars_held'] = max(1, int(elapsed_minutes / 5))
                except (ValueError, TypeError):
                    position['bars_held'] = 0
            else:
                position['bars_held'] = 0

            enriched_positions.append(position)

        prompt_data = {
            'timestamp': now.isoformat(),
            'day_of_week': now.strftime('%A'),
            'session_phase': CandidateBuilder.get_session_phase(),
            'account': {
                'balance': account_state.get('balance', 0),
                'active_capital': account_state.get('active_capital', 0),
                'buying_power': account_state.get('buying_power', 0),
                'phase': account_state.get('phase', 'GROWTH'),
                'open_position_count': len(open_positions),
                'max_positions': MAX_CONCURRENT_POSITIONS,
            },
            'open_positions': enriched_positions,
            'candidates': candidates,
        }

        return json.dumps(prompt_data, indent=2, default=str)

    def _build_cycle_result(
        self,
        cycle_id: str,
        candidates: List[Dict],
        response: Dict,
        active_capital: float,
        recent_symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        Parse the provider response into a structured cycle result.

        Uses Pydantic schema validation for strict type checking and
        business rule enforcement. Filters out trades for symbols on cooldown.

        Args:
            cycle_id: UUID for this decision cycle
            candidates: Candidates that were evaluated
            response: Normalized provider response
            active_capital: Current active capital for share sizing
            recent_symbols: Symbols to exclude due to cooldown

        Returns:
            Cycle result dict
        """
        result = {
            'cycle_id': cycle_id,
            'timestamp': datetime.now().isoformat(),
            'candidates_evaluated': len(candidates),
            'provider': response.get('provider', ''),
            'model': response.get('model', ''),
            'prompt_tokens': response.get('prompt_tokens', 0),
            'completion_tokens': response.get('completion_tokens', 0),
            'cost_estimate': response.get('cost_estimate', 0),
            'cache_creation_input_tokens': response.get('cache_creation_input_tokens', 0),
            'cache_read_input_tokens': response.get('cache_read_input_tokens', 0),
            'raw_response': response.get('content', ''),
            'action': 'HOLD',
            'trades': [],
            'exits': [],
            'cycle_notes': '',
            'parse_error': None,
        }

        if not response.get('success'):
            result['parse_error'] = response.get('error', 'Provider call failed')
            logger.error(f"Cycle {cycle_id}: provider error: {result['parse_error']}")
            return result

        # Parse JSON response
        parsed = self._parse_response(response.get('content', ''))
        if parsed is None:
            result['parse_error'] = 'Failed to parse JSON from provider response'
            logger.error(f"Cycle {cycle_id}: malformed JSON response")
            return result

        # Validate with Pydantic
        ai_response = self._validate_response(parsed)
        if ai_response is None:
            # Log per-trade pydantic rejections so we can see which business
            # rule (R:R, conviction, SL/TP distance) is killing candidates.
            self._log_pydantic_rejections(cycle_id, parsed)
            result['parse_error'] = 'Pydantic validation failed'
            logger.error(f"Cycle {cycle_id}: response failed schema validation")
            return result

        result['action'] = ai_response.action
        result['cycle_notes'] = ai_response.cycle_notes

        # Build per-symbol lookup maps from candidates
        direction_map = {}
        rsi_map = {}
        regime_map = {}
        atr_map = {}
        for c in candidates:
            sym = c.get('symbol', '')
            direction_map[sym] = c.get('allowed_direction', 'LONG_ONLY')
            rsi_map[sym] = c.get('rsi_14', c.get('rsi', 0))
            regime_map[sym] = c.get('regime', '')
            atr_map[sym] = c.get('atr_pct', 0)

        # Convert validated trades to dicts with computed shares
        valid_trades = []
        for trade in ai_response.trades:
            candidate_rsi = rsi_map.get(trade.symbol, 0)
            regime = regime_map.get(trade.symbol, '')
            rejection_metrics = {
                'conviction': trade.conviction,
                'rsi': candidate_rsi,
                'regime': regime,
                'direction': trade.direction,
            }

            # Enforce VWAP regime direction constraint
            allowed = direction_map.get(trade.symbol)
            if allowed:
                expected_dir = 'LONG' if allowed == 'LONG_ONLY' else 'SHORT'
                if trade.direction != expected_dir:
                    logger.warning(
                        f"Cycle {cycle_id}: trade {trade.symbol} {trade.direction} "
                        f"rejected — allowed_direction is {allowed}"
                    )
                    self._log_near_miss(
                        cycle_id=cycle_id,
                        symbol=trade.symbol,
                        gate='direction_mismatch',
                        gate_detail=f'proposed={trade.direction}, allowed={allowed}',
                        entry_price=trade.expected_entry_price,
                        would_be_direction=trade.direction,
                        metrics=rejection_metrics,
                    )
                    continue

            # Enforce RSI direction gate (AI ignores prompt guidance)
            if trade.direction == 'LONG':
                is_reversal = regime == 'REVERSAL_LONG'
                if is_reversal:
                    if candidate_rsi > LONG_REVERSAL_MAX_RSI:
                        logger.warning(
                            f"Cycle {cycle_id}: trade {trade.symbol} LONG reversal "
                            f"rejected — RSI {candidate_rsi:.1f} > {LONG_REVERSAL_MAX_RSI} "
                            f"(not oversold enough for reversal)"
                        )
                        self._log_near_miss(
                            cycle_id=cycle_id,
                            symbol=trade.symbol,
                            gate='rsi_direction',
                            gate_detail=f'LONG reversal rsi={candidate_rsi:.1f} > {LONG_REVERSAL_MAX_RSI}',
                            entry_price=trade.expected_entry_price,
                            would_be_direction='LONG',
                            metrics=rejection_metrics,
                        )
                        continue
                else:
                    if candidate_rsi < LONG_MIN_RSI:
                        logger.warning(
                            f"Cycle {cycle_id}: trade {trade.symbol} LONG "
                            f"rejected — RSI {candidate_rsi:.1f} < {LONG_MIN_RSI} "
                            f"(no bullish momentum)"
                        )
                        self._log_near_miss(
                            cycle_id=cycle_id,
                            symbol=trade.symbol,
                            gate='rsi_direction',
                            gate_detail=f'LONG rsi={candidate_rsi:.1f} < {LONG_MIN_RSI}',
                            entry_price=trade.expected_entry_price,
                            would_be_direction='LONG',
                            metrics=rejection_metrics,
                        )
                        continue
            elif trade.direction == 'SHORT':
                if candidate_rsi > SHORT_MAX_RSI:
                    logger.warning(
                        f"Cycle {cycle_id}: trade {trade.symbol} SHORT "
                        f"rejected — RSI {candidate_rsi:.1f} > {SHORT_MAX_RSI} "
                        f"(no bearish momentum)"
                    )
                    self._log_near_miss(
                        cycle_id=cycle_id,
                        symbol=trade.symbol,
                        gate='rsi_direction',
                        gate_detail=f'SHORT rsi={candidate_rsi:.1f} > {SHORT_MAX_RSI}',
                        entry_price=trade.expected_entry_price,
                        would_be_direction='SHORT',
                        metrics=rejection_metrics,
                    )
                    continue

            trade_dict = self._trade_to_dict(
                trade, active_capital, atr_map.get(trade.symbol, 0),
            )
            if trade_dict:
                valid_trades.append(trade_dict)

        # Filter out symbols on cooldown
        cooldown_set = set(recent_symbols) if recent_symbols else set()
        if cooldown_set:
            filtered_trades = []
            for trade in valid_trades:
                if trade['symbol'] in cooldown_set:
                    logger.warning(
                        f"Cycle {cycle_id}: trade {trade['symbol']} filtered "
                        f"— symbol on cooldown (recently traded)"
                    )
                    self._log_near_miss(
                        cycle_id=cycle_id,
                        symbol=trade['symbol'],
                        gate='cooldown',
                        gate_detail='recently traded',
                        entry_price=trade.get('expected_entry_price'),
                        would_be_direction=trade.get('direction'),
                        metrics={
                            'conviction': trade.get('conviction'),
                            'direction': trade.get('direction'),
                        },
                    )
                else:
                    filtered_trades.append(trade)
            valid_trades = filtered_trades

        result['trades'] = valid_trades

        # If all trades were filtered out, downgrade action to HOLD
        if result['action'] == 'ENTER' and not valid_trades:
            result['action'] = 'HOLD'

        # Handle EXIT action — cap at 1 exit per cycle
        if ai_response.action == 'EXIT' and ai_response.exits:
            first_exit = ai_response.exits[0]
            result['exits'] = [{
                'action': 'EXIT',
                'symbol': first_exit.symbol,
                'reasoning': first_exit.reasoning,
            }]

            if len(ai_response.exits) > 1:
                logger.warning(
                    f"Cycle {cycle_id}: AI proposed {len(ai_response.exits)} exits, "
                    f"taking only first ({first_exit.symbol}). "
                    f"Ignored: {[e.symbol for e in ai_response.exits[1:]]}"
                )

        return result

    def _parse_response(self, content: str) -> Optional[Dict]:
        """
        Parse JSON from provider response, handling common issues.

        Args:
            content: Raw response text

        Returns:
            Parsed dict, or None if unparseable
        """
        if not content:
            return None

        # Strip markdown code fences if present
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
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Raw content: {content[:500]}")
            return None

    def _log_pydantic_rejections(self, cycle_id: str, parsed: Dict) -> None:
        """
        When bulk AIResponse validation fails, try each trade individually
        and log which rule rejected it. Best-effort — missing fields are
        silently skipped.
        """
        trades = parsed.get('trades') if isinstance(parsed, dict) else None
        if not isinstance(trades, list):
            return
        for t in trades:
            if not isinstance(t, dict):
                continue
            symbol = t.get('symbol', '?')
            try:
                TradeDecision.model_validate(t)
            except ValidationError as e:
                first_err = e.errors()[0] if e.errors() else {'msg': 'unknown'}
                self._log_near_miss(
                    cycle_id=cycle_id,
                    symbol=symbol,
                    gate='pydantic',
                    gate_detail=str(first_err.get('msg', ''))[:200],
                    entry_price=t.get('expected_entry_price'),
                    would_be_direction=t.get('direction'),
                    metrics={
                        'conviction': t.get('conviction'),
                        'stop_loss': t.get('stop_loss'),
                        'take_profit': t.get('take_profit'),
                        'direction': t.get('direction'),
                    },
                )

    def _validate_response(self, parsed: Dict) -> Optional[AIResponse]:
        """
        Validate parsed JSON against the Pydantic AIResponse schema.

        Logs all validation errors with detail for debugging.

        Args:
            parsed: Parsed JSON dict from LLM response

        Returns:
            Validated AIResponse, or None if validation fails
        """
        try:
            return AIResponse.model_validate(parsed)
        except ValidationError as e:
            for error in e.errors():
                field = ' -> '.join(str(loc) for loc in error['loc'])
                logger.warning(f"Schema validation: {field}: {error['msg']}")
            return None

    def _trade_to_dict(
        self,
        trade: TradeDecision,
        active_capital: float,
        atr_pct: float,
    ) -> Optional[Dict]:
        """
        Convert a validated TradeDecision to a trade dict with Python-
        computed shares, stop_loss, and take_profit.

        Share sizing: position value = 20% of active capital (midpoint of
        15-30% range), capped at MAX_POSITION_SIZE_PCT. Minimum 1 share.

        Bracket (SL/TP) sizing (v1.4): derived from the candidate's ATR%
        rather than proposed by the AI. SL distance = ATR_SL_MULTIPLIER ×
        atr_pct, clamped to [MIN_SL_DISTANCE_PCT, MAX_SL_DISTANCE_PCT].
        TP distance = TARGET_RR_RATIO × SL distance, clamped to
        [MIN_TP_DISTANCE_PCT, MAX_TP_DISTANCE_PCT].

        Args:
            trade: Validated Pydantic TradeDecision
            active_capital: Current active capital for sizing
            atr_pct: Candidate's ATR as a percentage of price (e.g. 3.5
                     means 3.5% daily range). Falls back to MIN_SL_DISTANCE_PCT
                     when missing or non-positive.

        Returns:
            Trade dict ready for order execution, or None if sizing fails
        """
        entry = trade.expected_entry_price
        if entry <= 0:
            logger.warning(f"Trade {trade.symbol}: invalid entry price {entry}")
            return None

        # Calculate shares: target 20% of active capital, cap at 30%
        target_value = active_capital * TARGET_POSITION_PCT
        max_value = active_capital * MAX_POSITION_SIZE_PCT
        position_value = min(target_value, max_value)
        shares = max(1, math.floor(position_value / entry))

        # Verify the position doesn't exceed max
        if shares * entry > max_value:
            shares = math.floor(max_value / entry)
            if shares < 1:
                logger.warning(
                    f"Trade {trade.symbol}: share price ${entry:.2f} exceeds "
                    f"max position size ${max_value:.2f} — skipping"
                )
                return None

        # ATR-based bracket (v1.4)
        atr = atr_pct if atr_pct and atr_pct > 0 else MIN_SL_DISTANCE_PCT
        sl_dist_pct = max(
            MIN_SL_DISTANCE_PCT,
            min(MAX_SL_DISTANCE_PCT, ATR_SL_MULTIPLIER * atr),
        )
        tp_dist_pct = max(
            MIN_TP_DISTANCE_PCT,
            min(MAX_TP_DISTANCE_PCT, TARGET_RR_RATIO * sl_dist_pct),
        )

        if trade.direction == 'LONG':
            stop_loss = round(entry * (1 - sl_dist_pct / 100), 2)
            take_profit = round(entry * (1 + tp_dist_pct / 100), 2)
        else:
            stop_loss = round(entry * (1 + sl_dist_pct / 100), 2)
            take_profit = round(entry * (1 - tp_dist_pct / 100), 2)

        logger.info(
            f"Trade {trade.symbol}: sized {shares} shares @ ${entry:.2f} "
            f"= ${shares * entry:.2f} ({shares * entry / active_capital * 100:.1f}% "
            f"of ${active_capital:.2f} active capital)"
        )
        logger.info(
            f"Trade {trade.symbol}: bracket SL=${stop_loss} ({sl_dist_pct:.2f}%) "
            f"TP=${take_profit} ({tp_dist_pct:.2f}%) "
            f"from atr={atr:.2f}% × {ATR_SL_MULTIPLIER} × RR {TARGET_RR_RATIO}"
        )

        return {
            'action': trade.action,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'shares': shares,
            'order_type': 'Market',
            'expected_entry_price': trade.expected_entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'conviction': trade.conviction,
            'reasoning': trade.reasoning,
        }

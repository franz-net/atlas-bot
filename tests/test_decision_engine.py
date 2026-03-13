"""
Sprint 4 Tests — Decision Engine

Tests use mocked provider responses — no live AI API calls.
Tests cover Pydantic schema validation, share sizing, slippage guard,
and full decision cycle with mocked providers.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.engine.decision_engine import DecisionEngine
from src.engine.schemas import AIResponse, ExitDecision, TradeDecision


@pytest.fixture
def mock_provider():
    """Create a mock DecisionProvider."""
    provider = MagicMock()
    provider.provider_name = 'claude'
    provider.model_name = 'claude-sonnet-4-20250514'
    provider.decide = AsyncMock()
    provider.get_cost_estimate = MagicMock(return_value=0.005)
    return provider


@pytest.fixture
def engine(mock_provider):
    return DecisionEngine(provider=mock_provider)


@pytest.fixture
def sample_candidates():
    return [
        {
            'symbol': 'NVDA',
            'price': 875.40,
            'bid': 875.35,
            'ask': 875.45,
            'spread_pct': 0.011,
            'relative_volume': 2.3,
            'atr_pct': 2.1,
            'rsi_14': 58,
            'vwap': 872.10,
            'vwap_distance_pct': 0.38,
            'momentum_5d': 0.74,
            'recent_high': 891.0,
            'recent_low': 851.2,
            'bars_daily_5': [],
            'regime': 'TREND_LONG',
            'allowed_direction': 'LONG_ONLY',
            'news': [{'headline': 'NVDA upgrade', 'sentiment': 'positive',
                       'source': 'Benzinga', 'timestamp': '2026-03-10T10:00:00Z'}],
        }
    ]


@pytest.fixture
def sample_account_state():
    return {
        'balance': 30000,
        'active_capital': 5000,
        'buying_power': 30000,
        'phase': 'GROWTH',
    }


# ─── Pydantic Schema Tests ───────────────────────────────────────────


class TestTradeDecisionSchema:
    """Test Pydantic TradeDecision validation."""

    def test_valid_long_trade(self):
        trade = TradeDecision(
            action='ENTER',
            symbol='NVDA',
            direction='LONG',
            expected_entry_price=875.0,
            stop_loss=855.0,
            take_profit=920.0,
            reasoning='Breaking above 20-day high on 2.3x relative volume with Benzinga upgrade catalyst. RSI 58 not extended.',
        )
        assert trade.symbol == 'NVDA'
        assert trade.expected_entry_price == 875.0

    def test_valid_short_trade(self):
        trade = TradeDecision(
            action='ENTER',
            symbol='TSLA',
            direction='SHORT',
            expected_entry_price=250.0,
            stop_loss=260.0,
            take_profit=232.0,
            reasoning='Breaking below support at $252 with RSI at 32 and 1.8x relative volume. Negative earnings revision catalyst.',
        )
        assert trade.direction == 'SHORT'

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                # missing direction, expected_entry_price, stop_loss, take_profit
                reasoning='Some reasoning that is long enough to pass validation check.',
            )

    def test_invalid_action_enum(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='BUY',  # Invalid — must be 'ENTER'
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with Benzinga upgrade catalyst. RSI 58 not extended.',
            )

    def test_invalid_direction_enum(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='UP',  # Invalid — must be 'LONG' or 'SHORT'
                expected_entry_price=875.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with Benzinga upgrade catalyst. RSI 58 not extended.',
            )

    def test_reasoning_too_short(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Looks good.',
            )

    def test_symbol_too_long(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='ENTER',
                symbol='TOOLONG',  # > 5 chars
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58.',
            )

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=-10.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58.',
            )

    def test_zero_stop_loss_rejected(self):
        with pytest.raises(ValidationError):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58.',
            )

    def test_rr_ratio_too_low(self):
        """R:R of ~1:1 should be rejected (need 1:2)."""
        with pytest.raises(ValidationError, match='R:R too low'):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=860.0,  # $15 risk
                take_profit=890.0,  # $15 reward — 1:1 R:R
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )

    def test_tp_too_close_scalp_rejected(self):
        """TP < 0.8% from entry should be rejected as a scalp."""
        with pytest.raises(ValidationError, match='scalp'):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=872.0,  # $3 risk
                take_profit=881.0,  # $6 reward, 0.69% distance — scalp
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )

    def test_sl_too_tight_rejected(self):
        """SL < 0.3% from entry should be rejected."""
        with pytest.raises(ValidationError, match='too tight'):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=874.0,  # $1 = 0.11% — too tight
                take_profit=920.0,  # plenty of reward
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )

    def test_long_sl_above_entry_rejected(self):
        """LONG stop_loss above entry price makes no sense."""
        with pytest.raises(ValidationError, match='must be below'):
            TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=880.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )

    def test_short_sl_below_entry_rejected(self):
        """SHORT stop_loss below entry price makes no sense."""
        with pytest.raises(ValidationError, match='must be above'):
            TradeDecision(
                action='ENTER',
                symbol='TSLA',
                direction='SHORT',
                expected_entry_price=250.0,
                stop_loss=245.0,
                take_profit=230.0,
                reasoning='Breaking below support with RSI at 32 and 1.8x relative volume. Negative revision catalyst confirmed.',
            )

    def test_tp_too_far_rejected(self):
        """TP > 8% from entry should be rejected as unrealistic."""
        with pytest.raises(ValidationError, match='TP too far'):
            TradeDecision(
                action='ENTER',
                symbol='PATH',
                direction='LONG',
                expected_entry_price=11.52,
                stop_loss=11.10,  # 3.6% SL — OK
                take_profit=15.76,  # 36.8% — way too far
                reasoning='Momentum breakout above resistance with 2.1x volume. RSI 55 rising. SL below VWAP at 11.10.',
            )

    def test_sl_too_wide_rejected(self):
        """SL > 4% from entry should be rejected as too much risk."""
        with pytest.raises(ValidationError, match='SL too wide'):
            TradeDecision(
                action='ENTER',
                symbol='PATH',
                direction='LONG',
                expected_entry_price=100.0,
                stop_loss=94.0,  # 6% SL — exceeds 4% max
                take_profit=115.0,  # 15% TP — R:R 2.5:1, passes R:R check
                reasoning='Momentum breakout above resistance with 2.1x volume. RSI 55 rising. SL below structure at 94.',
            )

    def test_short_tp_too_far_rejected(self):
        """SHORT TP > 8% from entry should be rejected."""
        with pytest.raises(ValidationError, match='TP too far'):
            TradeDecision(
                action='ENTER',
                symbol='TSLA',
                direction='SHORT',
                expected_entry_price=250.0,
                stop_loss=258.0,  # 3.2% SL — OK
                take_profit=220.0,  # 12% — too far
                reasoning='Breaking below support with RSI at 32 and 1.8x relative volume. Negative revision catalyst confirmed.',
            )


class TestAIResponseSchema:
    """Test Pydantic AIResponse validation."""

    def test_valid_hold(self):
        response = AIResponse(
            action='HOLD',
            trades=[],
            cycle_notes='No setups meeting criteria.',
        )
        assert response.action == 'HOLD'
        assert len(response.trades) == 0

    def test_valid_enter(self):
        response = AIResponse(
            action='ENTER',
            trades=[TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )],
            cycle_notes='Entering NVDA.',
        )
        assert response.action == 'ENTER'
        assert len(response.trades) == 1

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            AIResponse(action='BUY', trades=[], cycle_notes='test')

    def test_hold_with_trades_clears_trades(self):
        """HOLD action should clear any trades the AI included."""
        response = AIResponse(
            action='HOLD',
            trades=[TradeDecision(
                action='ENTER',
                symbol='NVDA',
                direction='LONG',
                expected_entry_price=875.0,
                stop_loss=855.0,
                take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )],
            cycle_notes='Holding but AI included trades anyway.',
        )
        assert len(response.trades) == 0


# ─── Engine Integration Tests ────────────────────────────────────────


class TestEngineInit:
    """Test engine initialization."""

    def test_init_with_provider(self, mock_provider):
        engine = DecisionEngine(provider=mock_provider)
        assert engine._provider.provider_name == 'claude'

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv('DECISION_PROVIDER', 'claude')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        monkeypatch.setenv('CLAUDE_DECISION_MODEL', 'claude-sonnet-4-20250514')
        engine = DecisionEngine()
        assert engine._provider.provider_name == 'claude'


class TestResponseParsing:
    """Test JSON response parsing and validation."""

    def test_parse_valid_json(self, engine):
        content = json.dumps({
            'action': 'ENTER',
            'trades': [{'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                         'expected_entry_price': 875.0,
                         'stop_loss': 855.0, 'take_profit': 920.0,
                         'reasoning': 'Strong setup with 2.3x relative volume and Benzinga upgrade catalyst. RSI at 58 not extended.'}],
            'cycle_notes': 'Entering NVDA on upgrade catalyst.'
        })
        result = engine._parse_response(content)
        assert result is not None
        assert result['action'] == 'ENTER'

    def test_parse_json_with_code_fences(self, engine):
        content = '```json\n{"action": "HOLD", "trades": [], "cycle_notes": "No setups."}\n```'
        result = engine._parse_response(content)
        assert result is not None
        assert result['action'] == 'HOLD'

    def test_parse_empty_content(self, engine):
        assert engine._parse_response('') is None
        assert engine._parse_response(None) is None

    def test_parse_malformed_json(self, engine):
        assert engine._parse_response('not json at all') is None


class TestPydanticValidation:
    """Test _validate_response with Pydantic schema."""

    def test_valid_response(self, engine):
        parsed = {
            'action': 'ENTER',
            'trades': [{
                'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                'expected_entry_price': 875.0,
                'stop_loss': 855.0, 'take_profit': 920.0,
                'reasoning': 'Strong setup with 2.3x relative volume and Benzinga upgrade catalyst. RSI at 58 not extended.',
            }],
            'cycle_notes': 'Entering NVDA.',
        }
        result = engine._validate_response(parsed)
        assert result is not None
        assert result.action == 'ENTER'
        assert len(result.trades) == 1

    def test_invalid_action_returns_none(self, engine):
        parsed = {'action': 'BUY', 'trades': [], 'cycle_notes': 'test'}
        assert engine._validate_response(parsed) is None

    def test_bad_trade_in_list_returns_none(self, engine):
        parsed = {
            'action': 'ENTER',
            'trades': [{
                'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                'expected_entry_price': 875.0,
                'stop_loss': 855.0, 'take_profit': 920.0,
                'reasoning': 'Too short.',  # < 50 chars
            }],
            'cycle_notes': 'test',
        }
        assert engine._validate_response(parsed) is None


class TestShareSizing:
    """Test Python-side share calculation in _trade_to_dict."""

    def test_share_sizing_basic(self, engine):
        trade = TradeDecision(
            action='ENTER', symbol='NVDA', direction='LONG',
            expected_entry_price=100.0, stop_loss=97.0, take_profit=106.0,
            reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
        )
        # active_capital=1000, target 20% = $200, $200 / $100 = 2 shares
        result = engine._trade_to_dict(trade, active_capital=1000)
        assert result is not None
        assert result['shares'] == 2
        assert result['expected_entry_price'] == 100.0

    def test_share_sizing_expensive_stock_rejected(self, engine):
        trade = TradeDecision(
            action='ENTER', symbol='NVDA', direction='LONG',
            expected_entry_price=875.0, stop_loss=855.0, take_profit=920.0,
            reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
        )
        # active_capital=1000, max 30% = $300. $875 > $300 -> rejected
        result = engine._trade_to_dict(trade, active_capital=1000)
        assert result is None

    def test_share_sizing_expensive_stock_with_enough_capital(self, engine):
        trade = TradeDecision(
            action='ENTER', symbol='NVDA', direction='LONG',
            expected_entry_price=875.0, stop_loss=855.0, take_profit=920.0,
            reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
        )
        # active_capital=5000, max 30% = $1500. $1500 / $875 = 1 share
        result = engine._trade_to_dict(trade, active_capital=5000)
        assert result is not None
        assert result['shares'] == 1

    def test_share_sizing_cheap_stock(self, engine):
        trade = TradeDecision(
            action='ENTER', symbol='F', direction='LONG',
            expected_entry_price=10.0, stop_loss=9.65, take_profit=10.70,
            reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
        )
        # active_capital=1000, target 20% = $200, $200 / $10 = 20 shares
        # But max 30% = $300, $300 / $10 = 30. 20 < 30 so 20 shares.
        result = engine._trade_to_dict(trade, active_capital=1000)
        assert result is not None
        assert result['shares'] == 20

    def test_share_sizing_respects_max_pct(self, engine):
        trade = TradeDecision(
            action='ENTER', symbol='F', direction='LONG',
            expected_entry_price=10.0, stop_loss=9.65, take_profit=10.70,
            reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
        )
        # active_capital=100, target 20% = $20, $20 / $10 = 2 shares
        # max 30% = $30, 2 * $10 = $20 < $30, OK
        result = engine._trade_to_dict(trade, active_capital=100)
        assert result is not None
        assert result['shares'] == 2
        assert result['shares'] * 10 <= 100 * 0.30

    def test_trade_dict_has_all_fields(self, engine):
        trade = TradeDecision(
            action='ENTER', symbol='AAPL', direction='LONG',
            expected_entry_price=150.0, stop_loss=145.0, take_profit=161.0,
            reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
        )
        result = engine._trade_to_dict(trade, active_capital=1000)
        assert result is not None
        assert 'action' in result
        assert 'symbol' in result
        assert 'direction' in result
        assert 'shares' in result
        assert 'order_type' in result
        assert 'expected_entry_price' in result
        assert 'stop_loss' in result
        assert 'take_profit' in result
        assert 'reasoning' in result


class TestDecisionCycle:
    """Test full decision cycle with mocked provider."""

    @pytest.mark.asyncio
    async def test_no_candidates_skips_ai_call(self, engine, mock_provider, sample_account_state):
        """No candidates = automatic HOLD, no AI API call made."""
        result = await engine.decide([], sample_account_state, [])

        assert result['action'] == 'HOLD'
        assert result['candidates_evaluated'] == 0
        assert result['cost_estimate'] == 0
        assert result['prompt_tokens'] == 0
        assert 'No candidates' in result['cycle_notes']
        # Provider should NOT have been called
        mock_provider.decide.assert_not_called()

    @pytest.mark.asyncio
    async def test_enter_decision(self, engine, mock_provider, sample_candidates, sample_account_state):
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 855.0,
                    'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms technical setup.',
                }],
                'cycle_notes': 'Entering NVDA on upgrade catalyst.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 200,
            'cost_estimate': 0.009,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 1500,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1
        assert result['trades'][0]['symbol'] == 'NVDA'
        assert result['trades'][0]['stop_loss'] == 855.0
        assert result['trades'][0]['expected_entry_price'] == 875.0
        # Shares computed by Python: $5000 * 20% / $875 = 1 share
        assert result['trades'][0]['shares'] == 1
        assert result['cycle_id'] is not None
        assert result['candidates_evaluated'] == 1
        assert result['parse_error'] is None

    @pytest.mark.asyncio
    async def test_hold_decision(self, engine, mock_provider, sample_candidates, sample_account_state):
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'HOLD',
                'trades': [],
                'cycle_notes': 'No setups meeting criteria this cycle.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 50,
            'cost_estimate': 0.007,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 1500,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert result['action'] == 'HOLD'
        assert len(result['trades']) == 0
        assert result['cache_read_input_tokens'] == 1500

    @pytest.mark.asyncio
    async def test_provider_failure(self, engine, mock_provider, sample_candidates, sample_account_state):
        mock_provider.decide.return_value = {
            'success': False,
            'content': '',
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cost_estimate': 0,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': 'API rate limit exceeded',
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert result['action'] == 'HOLD'
        assert result['parse_error'] is not None
        assert len(result['trades']) == 0

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, engine, mock_provider, sample_candidates, sample_account_state):
        mock_provider.decide.return_value = {
            'success': True,
            'content': 'I think you should buy NVDA because...',  # Not JSON
            'prompt_tokens': 2000,
            'completion_tokens': 100,
            'cost_estimate': 0.008,
            'provider': 'gemini',
            'model': 'gemini-2.0-flash',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert result['action'] == 'HOLD'
        assert result['parse_error'] is not None

    @pytest.mark.asyncio
    async def test_trade_with_weak_reasoning_rejected(self, engine, mock_provider, sample_candidates, sample_account_state):
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 855.0, 'take_profit': 920.0,
                    'reasoning': 'Looks good.',  # Too short — Pydantic rejects
                }],
                'cycle_notes': 'Entering.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 50,
            'cost_estimate': 0.007,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        # Entire response fails Pydantic validation (trade reasoning too short)
        assert len(result['trades']) == 0

    @pytest.mark.asyncio
    async def test_bad_rr_ratio_rejected(self, engine, mock_provider, sample_candidates, sample_account_state):
        """AI returns a trade with bad R:R — Pydantic rejects the entire response."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 860.0, 'take_profit': 890.0,  # 1:1 R:R
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms setup.',
                }],
                'cycle_notes': 'Entering.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 50,
            'cost_estimate': 0.007,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert len(result['trades']) == 0
        assert result['parse_error'] == 'Pydantic validation failed'

    @pytest.mark.asyncio
    async def test_invalid_action_enum_rejected(self, engine, mock_provider, sample_candidates, sample_account_state):
        """AI returns action='BUY' instead of 'ENTER' — Pydantic catches it."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'BUY',  # Invalid enum
                'trades': [],
                'cycle_notes': 'Buying NVDA.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 50,
            'cost_estimate': 0.007,
            'provider': 'gemini',
            'model': 'gemini-2.0-flash',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert result['action'] == 'HOLD'
        assert result['parse_error'] == 'Pydantic validation failed'

    @pytest.mark.asyncio
    async def test_legacy_shares_field_ignored(self, engine, mock_provider, sample_candidates, sample_account_state):
        """AI includes 'shares' field — it's ignored, Python calculates shares."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'shares': 999,  # Should be ignored
                    'stop_loss': 855.0, 'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms setup.',
                }],
                'cycle_notes': 'Entering.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 50,
            'cost_estimate': 0.007,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])

        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1
        # Shares computed by Python, not 999
        assert result['trades'][0]['shares'] == 1


class TestSystemPrompt:
    """Test system prompt construction."""

    def test_prompt_includes_risk_rules(self, engine, sample_account_state):
        prompt = engine._build_system_prompt(sample_account_state)
        assert '3' in prompt  # MAX_CONCURRENT_POSITIONS
        assert '30' in prompt  # MAX_POSITION_SIZE_PCT

    def test_prompt_specifies_json_format(self, engine, sample_account_state):
        prompt = engine._build_system_prompt(sample_account_state)
        assert 'JSON' in prompt
        assert 'ENTER' in prompt
        assert 'HOLD' in prompt

    def test_prompt_no_shares_instruction(self, engine, sample_account_state):
        prompt = engine._build_system_prompt(sample_account_state)
        assert 'shares' in prompt.lower()
        assert 'expected_entry_price' in prompt

    def test_prompt_includes_active_capital(self, engine, sample_account_state):
        prompt = engine._build_system_prompt(sample_account_state)
        assert '$5000.00' in prompt


class TestUserPrompt:
    """Test user prompt construction."""

    def test_user_prompt_includes_candidates(self, engine, sample_candidates, sample_account_state):
        prompt = engine._build_user_prompt(sample_candidates, sample_account_state, [])
        data = json.loads(prompt)
        assert len(data['candidates']) == 1
        assert data['candidates'][0]['symbol'] == 'NVDA'

    def test_user_prompt_includes_account(self, engine, sample_candidates, sample_account_state):
        prompt = engine._build_user_prompt(sample_candidates, sample_account_state, [])
        data = json.loads(prompt)
        assert data['account']['active_capital'] == 5000
        assert data['account']['phase'] == 'GROWTH'

    def test_user_prompt_includes_positions(self, engine, sample_candidates, sample_account_state):
        positions = [{'Symbol': 'AAPL', 'Quantity': 2, 'AveragePrice': 150.0,
                       'UnrealizedProfitLoss': 10.0}]
        prompt = engine._build_user_prompt(sample_candidates, sample_account_state, positions)
        data = json.loads(prompt)
        assert len(data['open_positions']) == 1
        assert data['open_positions'][0]['symbol'] == 'AAPL'


# ─── Symbol Cooldown Tests ──────────────────────────────────────────


class TestSymbolCooldown:
    """Test symbol cooldown filtering in decision engine."""

    @pytest.mark.asyncio
    async def test_cooldown_filters_recent_symbol(self, engine, mock_provider, sample_candidates, sample_account_state):
        """Trade for a symbol on cooldown should be filtered out."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 855.0,
                    'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms technical setup.',
                }],
                'cycle_notes': 'Entering NVDA on upgrade catalyst.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 200,
            'cost_estimate': 0.009,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            sample_candidates, sample_account_state, [],
            recent_symbols=['NVDA'],
        )

        assert result['action'] == 'HOLD'
        assert len(result['trades']) == 0

    @pytest.mark.asyncio
    async def test_cooldown_allows_non_recent_symbol(self, engine, mock_provider, sample_candidates, sample_account_state):
        """Trade for a symbol NOT on cooldown should pass through."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 855.0,
                    'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms technical setup.',
                }],
                'cycle_notes': 'Entering NVDA on upgrade catalyst.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 200,
            'cost_estimate': 0.009,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            sample_candidates, sample_account_state, [],
            recent_symbols=['AAPL', 'TSLA'],
        )

        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1
        assert result['trades'][0]['symbol'] == 'NVDA'

    @pytest.mark.asyncio
    async def test_cooldown_empty_list_no_filtering(self, engine, mock_provider, sample_candidates, sample_account_state):
        """Empty recent_symbols list should not filter any trades."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 855.0,
                    'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms technical setup.',
                }],
                'cycle_notes': 'Entering NVDA on upgrade catalyst.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 200,
            'cost_estimate': 0.009,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            sample_candidates, sample_account_state, [],
            recent_symbols=[],
        )

        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1

    @pytest.mark.asyncio
    async def test_cooldown_none_no_filtering(self, engine, mock_provider, sample_candidates, sample_account_state):
        """None recent_symbols (default) should not filter any trades."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER',
                    'symbol': 'NVDA',
                    'direction': 'LONG',
                    'expected_entry_price': 875.0,
                    'stop_loss': 855.0,
                    'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms technical setup.',
                }],
                'cycle_notes': 'Entering NVDA on upgrade catalyst.',
            }),
            'prompt_tokens': 2000,
            'completion_tokens': 200,
            'cost_estimate': 0.009,
            'provider': 'claude',
            'model': 'claude-sonnet-4-20250514',
            'error': None,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            sample_candidates, sample_account_state, [],
        )

        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1


# ─── VWAP Direction Enforcement Tests ──────────────────────────────


class TestVWAPDirectionEnforcement:
    """Test that decision engine enforces allowed_direction from VWAP regime."""

    @pytest.mark.asyncio
    async def test_long_trade_matching_long_only_passes(self, engine, mock_provider, sample_candidates, sample_account_state):
        """LONG trade with LONG_ONLY allowed_direction should pass."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'LONG',
                    'expected_entry_price': 875.0, 'stop_loss': 855.0, 'take_profit': 920.0,
                    'reasoning': 'Breaking above 20-day high on 2.3x relative volume. Benzinga upgrade catalyst confirms setup.',
                }],
                'cycle_notes': 'Entering NVDA.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 200, 'cost_estimate': 0.009,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(sample_candidates, sample_account_state, [])
        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1

    @pytest.mark.asyncio
    async def test_short_trade_rejected_when_long_only(self, engine, mock_provider, sample_account_state):
        """SHORT trade with LONG_ONLY allowed_direction should be rejected."""
        candidates = [{
            'symbol': 'NVDA', 'price': 875.40, 'allowed_direction': 'LONG_ONLY',
            'atr_pct': 2.1, 'rsi_14': 58, 'vwap': 872.10,
        }]

        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'NVDA', 'direction': 'SHORT',
                    'expected_entry_price': 875.0, 'stop_loss': 895.0, 'take_profit': 835.0,
                    'reasoning': 'Shorting NVDA at resistance with declining volume. RSI overbought at 72 with bearish divergence confirmed.',
                }],
                'cycle_notes': 'Shorting NVDA.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 200, 'cost_estimate': 0.009,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(candidates, sample_account_state, [])
        assert result['action'] == 'HOLD'
        assert len(result['trades']) == 0

    @pytest.mark.asyncio
    async def test_long_trade_rejected_when_short_only(self, engine, mock_provider, sample_account_state):
        """LONG trade with SHORT_ONLY allowed_direction should be rejected."""
        candidates = [{
            'symbol': 'ORCL', 'price': 150.0, 'allowed_direction': 'SHORT_ONLY',
            'atr_pct': 1.8, 'rsi_14': 35, 'vwap': 152.0,
        }]

        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'ORCL', 'direction': 'LONG',
                    'expected_entry_price': 150.0, 'stop_loss': 145.0, 'take_profit': 161.0,
                    'reasoning': 'Buying ORCL at support level with high relative volume and news catalyst. RSI at 35 oversold bounce.',
                }],
                'cycle_notes': 'Entering ORCL long.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 200, 'cost_estimate': 0.009,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(candidates, sample_account_state, [])
        assert result['action'] == 'HOLD'
        assert len(result['trades']) == 0

    @pytest.mark.asyncio
    async def test_short_trade_matching_short_only_passes(self, engine, mock_provider, sample_account_state):
        """SHORT trade with SHORT_ONLY allowed_direction should pass."""
        candidates = [{
            'symbol': 'ORCL', 'price': 150.0, 'allowed_direction': 'SHORT_ONLY',
            'atr_pct': 1.8, 'rsi_14': 35, 'vwap': 152.0,
        }]

        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'ENTER',
                'trades': [{
                    'action': 'ENTER', 'symbol': 'ORCL', 'direction': 'SHORT',
                    'expected_entry_price': 150.0, 'stop_loss': 155.0, 'take_profit': 139.0,
                    'reasoning': 'Shorting ORCL below declining VWAP with high volume breakdown. RSI at 35 confirms bearish momentum.',
                }],
                'cycle_notes': 'Shorting ORCL.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 200, 'cost_estimate': 0.009,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(candidates, sample_account_state, [])
        assert result['action'] == 'ENTER'
        assert len(result['trades']) == 1
        assert result['trades'][0]['direction'] == 'SHORT'


class TestExitDecisionSchema:
    """Test Pydantic ExitDecision validation."""

    def test_valid_exit(self):
        exit_dec = ExitDecision(
            action='EXIT',
            symbol='PATH',
            reasoning='Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain before thesis invalidates.',
        )
        assert exit_dec.symbol == 'PATH'
        assert exit_dec.action == 'EXIT'

    def test_exit_short_reasoning_rejected(self):
        """Reasoning must be 50+ chars."""
        with pytest.raises(ValidationError):
            ExitDecision(
                action='EXIT',
                symbol='PATH',
                reasoning='Too short',
            )

    def test_exit_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            ExitDecision(
                action='ENTER',
                symbol='PATH',
                reasoning='Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain.',
            )


class TestAIResponseExit:
    """Test AIResponse with EXIT action."""

    def test_valid_exit_response(self):
        response = AIResponse(
            action='EXIT',
            exits=[ExitDecision(
                action='EXIT',
                symbol='PATH',
                reasoning='Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain before thesis invalidates.',
            )],
            cycle_notes='Exiting PATH.',
        )
        assert response.action == 'EXIT'
        assert len(response.exits) == 1
        assert len(response.trades) == 0

    def test_exit_with_no_exits_downgrades_to_hold(self):
        response = AIResponse(
            action='EXIT',
            exits=[],
            cycle_notes='Wanted to exit but no targets.',
        )
        assert response.action == 'HOLD'
        assert len(response.exits) == 0

    def test_exit_clears_trades(self):
        """EXIT action should clear any trades list."""
        response = AIResponse(
            action='EXIT',
            trades=[TradeDecision(
                action='ENTER', symbol='NVDA', direction='LONG',
                expected_entry_price=875.0, stop_loss=855.0, take_profit=920.0,
                reasoning='Breaking above 20-day high on 2.3x relative volume with catalyst. RSI 58 not extended.',
            )],
            exits=[ExitDecision(
                action='EXIT', symbol='PATH',
                reasoning='Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain.',
            )],
            cycle_notes='Mixed signals.',
        )
        assert response.action == 'EXIT'
        assert len(response.trades) == 0
        assert len(response.exits) == 1

    def test_hold_clears_exits(self):
        """HOLD action should clear exits list."""
        response = AIResponse(
            action='HOLD',
            exits=[ExitDecision(
                action='EXIT', symbol='PATH',
                reasoning='Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain.',
            )],
            cycle_notes='Holding.',
        )
        assert response.action == 'HOLD'
        assert len(response.exits) == 0


class TestExitDecisionCycle:
    """Test full EXIT decision cycle with mocked provider."""

    @pytest.fixture
    def sample_open_positions(self):
        return [{'Symbol': 'PATH', 'Quantity': 86, 'AveragePrice': 11.52, 'UnrealizedProfitLoss': 18.49}]

    @pytest.fixture
    def sample_ledger_trades(self):
        return [{
            'id': 1,
            'symbol': 'PATH',
            'direction': 'LONG',
            'shares': 86,
            'entry_price': 11.52,
            'stop_loss_price': 11.10,
            'take_profit_price': 12.10,
            'entry_reasoning': 'Momentum breakout above resistance with 2.1x volume.',
            'entry_timestamp': '2026-03-12T10:15:00',
            'stop_order_id': 'SL123',
            'tp_order_id': 'TP456',
        }]

    @pytest.mark.asyncio
    async def test_exit_decision_parsed(self, engine, mock_provider, sample_account_state, sample_open_positions, sample_ledger_trades):
        """EXIT response should produce exits list capped at 1."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'EXIT',
                'trades': [],
                'exits': [{
                    'action': 'EXIT', 'symbol': 'PATH',
                    'reasoning': 'Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain before thesis invalidates.',
                }],
                'cycle_notes': 'Exiting PATH.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 100, 'cost_estimate': 0.007,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            [], sample_account_state, sample_open_positions,
            ledger_open_trades=sample_ledger_trades,
        )
        assert result['action'] == 'EXIT'
        assert len(result['exits']) == 1
        assert result['exits'][0]['symbol'] == 'PATH'
        assert len(result['trades']) == 0

    @pytest.mark.asyncio
    async def test_exit_caps_at_one(self, engine, mock_provider, sample_account_state, sample_open_positions, sample_ledger_trades):
        """Multiple exits should be capped at 1."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'EXIT',
                'trades': [],
                'exits': [
                    {'action': 'EXIT', 'symbol': 'PATH',
                     'reasoning': 'Momentum stalled, price consolidating for 3+ candles, volume dried up. Taking gain.'},
                    {'action': 'EXIT', 'symbol': 'NVDA',
                     'reasoning': 'Volume dried up and price is consolidating after failed breakout. Protecting remaining gain.'},
                ],
                'cycle_notes': 'Multiple exits.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 100, 'cost_estimate': 0.007,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            [], sample_account_state, sample_open_positions,
            ledger_open_trades=sample_ledger_trades,
        )
        assert result['action'] == 'EXIT'
        assert len(result['exits']) == 1
        assert result['exits'][0]['symbol'] == 'PATH'

    @pytest.mark.asyncio
    async def test_no_candidates_with_positions_still_calls_ai(self, engine, mock_provider, sample_account_state, sample_open_positions, sample_ledger_trades):
        """When no candidates but positions exist, AI should still be called (for exit evaluation)."""
        mock_provider.decide.return_value = {
            'success': True,
            'content': json.dumps({
                'action': 'HOLD',
                'trades': [],
                'exits': [],
                'cycle_notes': 'Positions look fine.',
            }),
            'prompt_tokens': 2000, 'completion_tokens': 50, 'cost_estimate': 0.005,
            'provider': 'claude', 'model': 'claude-sonnet-4-20250514', 'error': None,
            'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0,
        }

        result = await engine.decide(
            [], sample_account_state, sample_open_positions,
            ledger_open_trades=sample_ledger_trades,
        )
        # AI was called (not short-circuited)
        mock_provider.decide.assert_called_once()
        assert result['action'] == 'HOLD'

    @pytest.mark.asyncio
    async def test_no_candidates_no_positions_skips_ai(self, engine, mock_provider, sample_account_state):
        """When no candidates AND no positions, AI call should be skipped."""
        result = await engine.decide([], sample_account_state, [])
        mock_provider.decide.assert_not_called()
        assert result['action'] == 'HOLD'
        assert result['exits'] == []


class TestEnrichedPositions:
    """Test enriched open_positions payload for AI."""

    def test_positions_include_ledger_data(self, engine):
        """Open positions should be enriched with ledger trade data."""
        broker_positions = [
            {'Symbol': 'PATH', 'Quantity': 86, 'AveragePrice': 11.74, 'UnrealizedProfitLoss': 18.49},
        ]
        ledger_trades = [{
            'symbol': 'PATH', 'direction': 'LONG', 'shares': 86,
            'entry_price': 11.52, 'stop_loss_price': 11.10,
            'take_profit_price': 12.10,
            'entry_reasoning': 'Momentum breakout above resistance.',
            'entry_timestamp': '2026-03-12T10:15:00',
        }]
        account_state = {'balance': 30000, 'active_capital': 5000, 'buying_power': 30000, 'phase': 'GROWTH'}

        prompt = engine._build_user_prompt(
            [], account_state, broker_positions,
            ledger_open_trades=ledger_trades,
        )
        data = json.loads(prompt)

        pos = data['open_positions'][0]
        assert pos['symbol'] == 'PATH'
        assert pos['direction'] == 'LONG'
        assert pos['entry_price'] == 11.52
        assert pos['stop_loss'] == 11.10
        assert pos['take_profit'] == 12.10
        assert pos['entry_reasoning'] == 'Momentum breakout above resistance.'
        assert pos['unrealized_pnl'] == 18.49
        assert 'bars_held' in pos
        assert 'unrealized_pnl_pct' in pos

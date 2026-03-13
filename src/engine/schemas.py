"""
Pydantic models for AI decision engine responses.

Provides strict schema validation for LLM output. If the AI returns
anything that doesn't match these schemas, the trade is rejected
and the cycle falls back to HOLD.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from strategy.thresholds import (
    MAX_SL_DISTANCE_PCT,
    MAX_TP_DISTANCE_PCT,
    MIN_REASONING_LENGTH,
    MIN_REWARD_RISK_RATIO,
    MIN_SL_DISTANCE_PCT,
    MIN_TP_DISTANCE_PCT,
)


class TradeDecision(BaseModel):
    """
    A single trade proposed by the AI decision engine.

    The AI provides expected_entry_price, stop_loss, and take_profit.
    Share calculation is done by the Python execution engine, not the LLM.
    """

    action: Literal['ENTER']
    symbol: str = Field(..., min_length=1, max_length=5)
    direction: Literal['LONG', 'SHORT']
    expected_entry_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    reasoning: str = Field(..., min_length=MIN_REASONING_LENGTH)

    @model_validator(mode='after')
    def validate_trade_levels(self) -> 'TradeDecision':
        """
        Validate R:R ratio, TP distance, and SL distance.

        Returns:
            Self if valid

        Raises:
            ValueError: If any validation fails
        """
        entry = self.expected_entry_price
        sl = self.stop_loss
        tp = self.take_profit

        if self.direction == 'LONG':
            if sl >= entry:
                raise ValueError(
                    f'LONG stop_loss ({sl}) must be below '
                    f'expected_entry_price ({entry})'
                )
            if tp <= entry:
                raise ValueError(
                    f'LONG take_profit ({tp}) must be above '
                    f'expected_entry_price ({entry})'
                )
            risk = entry - sl
            reward = tp - entry
        else:
            if sl <= entry:
                raise ValueError(
                    f'SHORT stop_loss ({sl}) must be above '
                    f'expected_entry_price ({entry})'
                )
            if tp >= entry:
                raise ValueError(
                    f'SHORT take_profit ({tp}) must be below '
                    f'expected_entry_price ({entry})'
                )
            risk = sl - entry
            reward = entry - tp

        # R:R minimum check
        if risk <= 0:
            raise ValueError(f'Invalid risk calculation: risk={risk:.2f}')

        tp_distance_pct = (reward / entry) * 100
        sl_distance_pct = (risk / entry) * 100

        # Maximum distance checks first — reject absurdly wide brackets
        if sl_distance_pct > MAX_SL_DISTANCE_PCT:
            raise ValueError(
                f'SL too wide ({sl_distance_pct:.2f}%, max {MAX_SL_DISTANCE_PCT}%). '
                f'Risk too large — tighten stop to nearby structure.'
            )

        if tp_distance_pct > MAX_TP_DISTANCE_PCT:
            raise ValueError(
                f'TP too far ({tp_distance_pct:.2f}%, max {MAX_TP_DISTANCE_PCT}%). '
                f'Unrealistic target — size bracket to actual ATR.'
            )

        # R:R ratio check
        rr_ratio = reward / risk
        if rr_ratio < MIN_REWARD_RISK_RATIO:
            raise ValueError(
                f'R:R too low ({rr_ratio:.2f}:1, need 2:1). '
                f'Risk=${risk:.2f} Reward=${reward:.2f}'
            )

        # Minimum TP distance from entry
        if tp_distance_pct < MIN_TP_DISTANCE_PCT:
            raise ValueError(
                f'TP too close ({tp_distance_pct:.2f}%, need {MIN_TP_DISTANCE_PCT}%+). '
                f'This is a scalp, not a swing trade.'
            )

        # Minimum SL distance from entry
        if sl_distance_pct < MIN_SL_DISTANCE_PCT:
            raise ValueError(
                f'SL too tight ({sl_distance_pct:.2f}%, need {MIN_SL_DISTANCE_PCT}%+). '
                f'Will get stopped on noise.'
            )

        return self


class ExitDecision(BaseModel):
    """
    A request to close an open position early.

    The AI proposes EXIT when the original trade thesis has invalidated.
    No price levels needed — exits are market orders.
    """

    action: Literal['EXIT']
    symbol: str = Field(..., min_length=1, max_length=5)
    reasoning: str = Field(..., min_length=MIN_REASONING_LENGTH)


class AIResponse(BaseModel):
    """
    Top-level AI decision response.

    The action field determines what the AI wants to do:
    - HOLD: no action
    - ENTER: open new positions (trades list)
    - EXIT: close existing positions (exits list)
    """

    action: Literal['ENTER', 'HOLD', 'EXIT']
    trades: List[TradeDecision] = Field(default_factory=list)
    exits: List[ExitDecision] = Field(default_factory=list)
    cycle_notes: str = ''

    @model_validator(mode='after')
    def validate_action_consistency(self) -> 'AIResponse':
        """
        Ensure action matches trades/exits lists.

        Returns:
            Self if valid
        """
        if self.action == 'HOLD':
            self.trades = []
            self.exits = []
        elif self.action == 'ENTER':
            self.exits = []
        elif self.action == 'EXIT':
            self.trades = []
            if not self.exits:
                # EXIT with no exits → downgrade to HOLD
                self.action = 'HOLD'
        return self

"""
Pydantic models for AI decision engine responses.

As of v1.4, the AI no longer proposes stop_loss or take_profit levels.
Those are computed in Python from the candidate's ATR% after validation
(see decision_engine._trade_to_dict). This removes a class of LLM
hallucinations where TP was arithmetically derived from SL to pass R:R
checks, rather than picked from real technical levels.

What the AI still provides per trade:
  action, symbol, direction, expected_entry_price, conviction, reasoning
"""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strategy.thresholds import (
    MIN_CONVICTION_SCORE,
    MIN_REASONING_LENGTH,
)


class TradeDecision(BaseModel):
    """
    A single trade proposed by the AI decision engine.

    The AI only chooses direction + entry price + conviction.
    Share sizing, SL, and TP are all computed by Python.
    """

    # Accept but ignore legacy fields from older AI responses (stop_loss,
    # take_profit). Pydantic would otherwise just silently ignore; being
    # explicit documents the contract.
    model_config = ConfigDict(extra='ignore')

    action: Literal['ENTER']
    symbol: str = Field(..., min_length=1, max_length=5)
    direction: Literal['LONG', 'SHORT']
    expected_entry_price: float = Field(..., gt=0)
    conviction: int = Field(..., ge=1, le=10)
    reasoning: str = Field(..., min_length=MIN_REASONING_LENGTH)

    @model_validator(mode='after')
    def validate_conviction(self) -> 'TradeDecision':
        """Conviction must clear the minimum threshold."""
        if self.conviction < MIN_CONVICTION_SCORE:
            raise ValueError(
                f'Conviction too low ({self.conviction}/10, need {MIN_CONVICTION_SCORE}+). '
                f'Not a high-quality setup — HOLD instead.'
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

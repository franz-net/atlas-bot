"""
Screener filter thresholds and validation parameters — EXAMPLE TEMPLATE.

Copy strategy.example/ to strategy/ and tune these values for your strategy.
"""

# ==================== Screener Filters ====================
# These determine which stocks from the watchlist reach the AI

SCREENER_PRICE_MIN = 5.0           # Exclude penny stocks
SCREENER_PRICE_MAX = 500.0         # Exclude ultra-high-priced stocks
SCREENER_MIN_RELATIVE_VOLUME = 1.5 # Volume vs 20-day average
SCREENER_MAX_VWAP_DISTANCE_PCT = 2.0  # Max distance from VWAP (%)
SCREENER_MIN_ATR_PCT = 1.0        # Minimum daily range (%)
SCREENER_RSI_MIN = 30             # Lower RSI bound
SCREENER_RSI_MAX = 70             # Upper RSI bound
SCREENER_MAX_SPREAD_PCT = 0.20    # Max bid-ask spread (%)
SCREENER_QUOTE_BATCH_SIZE = 40    # Symbols per API quote batch
SCREENER_TOP_CANDIDATES = 10      # Max candidates sent to AI

# ==================== Candidate Ranking Weights ====================
# Scoring formula: (rel_vol * VOL_WEIGHT) + (momentum * MOM_WEIGHT) + (atr * ATR_WEIGHT)

RANKING_WEIGHT_VOLUME = 0.4       # Relative volume importance
RANKING_WEIGHT_MOMENTUM = 0.3     # 5-day momentum importance
RANKING_WEIGHT_ATR = 0.3          # Volatility importance

# ==================== Trade Validation (Pydantic) ====================
# These are enforced in code — the AI cannot bypass them

MIN_REWARD_RISK_RATIO = 1.8       # Minimum R:R (1.8 with buffer for 2:1 target)
MIN_TP_DISTANCE_PCT = 0.8         # Minimum take-profit distance from entry (%)
MIN_SL_DISTANCE_PCT = 0.3         # Minimum stop-loss distance from entry (%)
MAX_TP_DISTANCE_PCT = 8.0         # Maximum take-profit distance from entry (%)
MAX_SL_DISTANCE_PCT = 4.0         # Maximum stop-loss distance from entry (%)
MIN_REASONING_LENGTH = 50         # Minimum characters in trade reasoning

# ==================== Position Sizing ====================
# Engine calculates shares — AI only provides price levels

TARGET_POSITION_PCT = 0.20        # Target position size (20% of active capital)

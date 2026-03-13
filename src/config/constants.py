"""
Constants for Project Atlas — AI-Powered Equity Trading System

Shared constants used across all modules. Inherited from Midas with
equity-specific additions for Atlas.
"""

# ==================== API Rate Limiting ====================
API_RATE_LIMIT_CALLS = 500
API_RATE_LIMIT_PERIOD_SECONDS = 60

# ==================== API Request Defaults ====================
DEFAULT_API_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3

# ==================== Async Sleep Intervals ====================
BRIEF_ASYNC_SLEEP_SECONDS = 0.5
SHORT_ASYNC_SLEEP_SECONDS = 2.0

# ==================== Order Verification ====================
ORDER_VERIFICATION_TIMEOUT_SECONDS = 30

# ==================== Tick Sizes ====================
DEFAULT_TICK_SIZE = 0.01  # Equity tick size

# ==================== Capital and Risk ====================
PROTECTED_FLOOR = 25000
HARD_STOP_ACTIVE_CAPITAL_PCT = 0.20  # Halt if active capital drops below 20% of start-of-day capital
MAX_POSITION_SIZE_PCT = 0.30
MAX_CONCURRENT_POSITIONS = 3
MAX_SLIPPAGE_PCT = 0.005  # 0.5% max allowed slippage from AI's expected entry

# ==================== Market Hours (EST) ====================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
NO_ENTRY_BEFORE_MINUTE = 35  # No entries 9:30-9:35
MARKET_CLOSE_HOUR = 16
NO_ENTRY_AFTER_HOUR = 15
NO_ENTRY_AFTER_MINUTE = 45  # No entries after 15:45

# ==================== Trading Loop ====================
DEFAULT_LOOP_INTERVAL_SECONDS = 300  # 5 minutes

# ==================== Symbol Cooldown ====================
SYMBOL_COOLDOWN_CYCLES = 6  # 30 minutes at 5-min intervals

# ==================== Screener Filters (from strategy) ====================
# These are re-exported here so all modules import from constants.py
# Customize values in strategy/thresholds.py
from strategy.thresholds import (  # noqa: E402
    SCREENER_PRICE_MIN,
    SCREENER_PRICE_MAX,
    SCREENER_MIN_RELATIVE_VOLUME,
    SCREENER_MAX_VWAP_DISTANCE_PCT,
    SCREENER_MIN_ATR_PCT,
    SCREENER_RSI_MIN,
    SCREENER_RSI_MAX,
    SCREENER_MAX_SPREAD_PCT,
    SCREENER_QUOTE_BATCH_SIZE,
    SCREENER_TOP_CANDIDATES,
)

"""
Technical indicator calculations for the Atlas stock screener.

Pure computation functions operating on pandas DataFrames. No API calls,
no side effects. All functions return Optional[float] for graceful handling
of insufficient data.
"""

from typing import Optional, Tuple

import pandas as pd
import pandas_ta as ta


def calculate_atr(daily_bars: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range from daily OHLCV bars.

    Args:
        daily_bars: DataFrame with Open, High, Low, Close columns
        period: ATR lookback period

    Returns:
        Latest ATR value, or None if insufficient data
    """
    if len(daily_bars) < period + 1:
        return None

    try:
        atr_series = ta.atr(
            daily_bars['High'], daily_bars['Low'], daily_bars['Close'],
            length=period,
        )
        if atr_series is None or atr_series.empty:
            return None
        return float(atr_series.iloc[-1])
    except Exception:
        return None


def calculate_atr_pct(
    daily_bars: pd.DataFrame, current_price: float, period: int = 14
) -> Optional[float]:
    """
    Calculate ATR as a percentage of current price.

    Args:
        daily_bars: DataFrame with OHLCV columns
        current_price: Current stock price
        period: ATR lookback period

    Returns:
        ATR percentage (e.g., 2.1 means 2.1%), or None if insufficient data
    """
    if current_price <= 0:
        return None

    atr = calculate_atr(daily_bars, period)
    if atr is None:
        return None

    return (atr / current_price) * 100


def calculate_vwap(intraday_bars: pd.DataFrame) -> Optional[float]:
    """
    Calculate Volume Weighted Average Price from intraday bars.

    Args:
        intraday_bars: DataFrame with High, Low, Close, Volume columns
                       (typically 1-minute bars for current session)

    Returns:
        Latest VWAP value, or None if insufficient data
    """
    if intraday_bars is None or intraday_bars.empty:
        return None

    try:
        vwap_series = ta.vwap(
            intraday_bars['High'], intraday_bars['Low'],
            intraday_bars['Close'], intraday_bars['Volume'],
        )
        if vwap_series is None or vwap_series.empty:
            return None
        return float(vwap_series.iloc[-1])
    except Exception:
        return None


def calculate_vwap_distance_pct(vwap: float, current_price: float) -> Optional[float]:
    """
    Calculate distance from VWAP as a percentage.

    Args:
        vwap: Current VWAP value
        current_price: Current stock price

    Returns:
        Absolute percentage distance from VWAP, or None if inputs invalid
    """
    if vwap is None or vwap <= 0 or current_price <= 0:
        return None

    return abs(current_price - vwap) / vwap * 100


def calculate_rsi(daily_bars: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index from daily close prices.

    Args:
        daily_bars: DataFrame with Close column
        period: RSI lookback period

    Returns:
        Latest RSI value (0-100), or None if insufficient data
    """
    if len(daily_bars) < period + 1:
        return None

    try:
        rsi_series = ta.rsi(daily_bars['Close'], length=period)
        if rsi_series is None or rsi_series.empty:
            return None
        value = rsi_series.iloc[-1]
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def calculate_relative_volume(
    current_volume: float,
    daily_bars: pd.DataFrame,
    period: int = 20,
    minutes_since_open: int = 0,
) -> Optional[float]:
    """
    Calculate relative volume compared to 20-day average.

    Normalizes current intraday volume by time-of-day to avoid the
    morning low-volume bias. Projects current volume to a full-day
    equivalent before comparing.

    Args:
        current_volume: Current intraday volume from quote
        daily_bars: DataFrame with Volume column (last 20+ daily bars)
        period: Number of days for average volume
        minutes_since_open: Minutes elapsed since 9:30 EST (0-390)

    Returns:
        Relative volume ratio (e.g., 2.3 means 2.3x average), or None
    """
    if current_volume <= 0 or len(daily_bars) < period:
        return None

    try:
        avg_volume = float(daily_bars['Volume'].tail(period).mean())
        if avg_volume <= 0:
            return None

        # Normalize by time of day if market is in session
        if 0 < minutes_since_open < 390:
            day_fraction = minutes_since_open / 390
            projected_volume = current_volume / day_fraction
        else:
            projected_volume = current_volume

        return projected_volume / avg_volume
    except Exception:
        return None


def calculate_momentum_5d(daily_bars: pd.DataFrame) -> Optional[float]:
    """
    Calculate 5-day price momentum as percentage change.

    Args:
        daily_bars: DataFrame with Close column (needs at least 6 rows)

    Returns:
        5-day percentage change (e.g., 0.05 means +5%), or None
    """
    if len(daily_bars) < 6:
        return None

    try:
        close_today = float(daily_bars['Close'].iloc[-1])
        close_5d = float(daily_bars['Close'].iloc[-6])
        if close_5d <= 0:
            return None
        return (close_today - close_5d) / close_5d
    except Exception:
        return None


def determine_market_regime(
    intraday_bars: pd.DataFrame, current_price: float, vwap: float,
    rsi: Optional[float] = None,
) -> str:
    """
    Determine market regime archetype for a candidate.

    Archetypes:
        TREND_LONG: Price above rising VWAP, bullish continuation
        TREND_SHORT: Price below declining VWAP, bearish continuation
        REVERSAL_LONG: Price below VWAP but selling is decelerating —
                       RSI oversold, MACD histogram ticking up, green candle
        AVOID: No clear setup or falling knife (no momentum shift)

    Args:
        intraday_bars: 1-minute OHLCV DataFrame for current session
        current_price: Current stock price
        vwap: Current VWAP value
        rsi: Pre-computed RSI-14 value (from daily bars)

    Returns:
        One of 'TREND_LONG', 'TREND_SHORT', 'REVERSAL_LONG', or 'AVOID'
    """
    if vwap is None or vwap <= 0 or current_price <= 0:
        return 'AVOID'

    if intraday_bars is None or len(intraday_bars) < 10:
        return 'AVOID'

    try:
        # Compute VWAP series for slope check
        vwap_series = ta.vwap(
            intraday_bars['High'], intraday_bars['Low'],
            intraday_bars['Close'], intraday_bars['Volume'],
        )
        if vwap_series is None or len(vwap_series) < 4:
            return 'AVOID'

        vwap_slope = float(vwap_series.iloc[-1]) - float(vwap_series.iloc[-4])

        # --- Trend regimes ---
        if current_price > vwap and vwap_slope > 0:
            return 'TREND_LONG'
        if current_price < vwap and vwap_slope < 0:
            # Check for reversal gate before tagging as TREND_SHORT
            if _check_reversal_gate(intraday_bars, rsi):
                return 'REVERSAL_LONG'
            return 'TREND_SHORT'

        # Mixed signals (price above declining VWAP, or below rising VWAP)
        return 'AVOID'
    except Exception:
        return 'AVOID'


def _check_reversal_gate(
    intraday_bars: pd.DataFrame, rsi: Optional[float] = None,
) -> bool:
    """
    Check if a stock below declining VWAP qualifies as a reversal candidate.

    Requires ALL of:
        1. RSI oversold (< 35)
        2. MACD histogram ticking up (momentum shift)
        3. Latest candle is green (Close > Open)

    Args:
        intraday_bars: 1-minute OHLCV DataFrame
        rsi: Pre-computed RSI-14 value

    Returns:
        True if reversal gate passes
    """
    # Gate 1: RSI must be oversold
    if rsi is None or rsi >= 35:
        return False

    try:
        # Gate 2: MACD histogram must be rising (momentum shift)
        macd_result = ta.macd(intraday_bars['Close'])
        if macd_result is None or macd_result.empty:
            return False

        # MACD histogram column name from pandas_ta
        hist_col = [c for c in macd_result.columns if 'MACDh' in c]
        if not hist_col:
            return False

        hist = macd_result[hist_col[0]]
        if len(hist) < 2 or pd.isna(hist.iloc[-1]) or pd.isna(hist.iloc[-2]):
            return False

        histogram_rising = float(hist.iloc[-1]) > float(hist.iloc[-2])
        if not histogram_rising:
            return False

        # Gate 3: Latest candle must be green (Close > Open)
        last_close = float(intraday_bars['Close'].iloc[-1])
        last_open = float(intraday_bars['Open'].iloc[-1])
        if last_close <= last_open:
            return False

        return True
    except Exception:
        return False


# Keep backward-compatible alias
def determine_vwap_regime(
    intraday_bars: pd.DataFrame, current_price: float, vwap: float
) -> str:
    """
    Legacy wrapper — maps new regime archetypes to old direction labels.

    Args:
        intraday_bars: 1-minute OHLCV DataFrame for current session
        current_price: Current stock price
        vwap: Current VWAP value

    Returns:
        'LONG_ONLY', 'SHORT_ONLY', or 'NONE'
    """
    regime = determine_market_regime(intraday_bars, current_price, vwap)
    if regime in ('TREND_LONG', 'REVERSAL_LONG'):
        return 'LONG_ONLY'
    elif regime == 'TREND_SHORT':
        return 'SHORT_ONLY'
    return 'NONE'


def calculate_macd_histogram(
    intraday_bars: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate current and previous MACD histogram values from intraday bars.

    Args:
        intraday_bars: 1-minute OHLCV DataFrame

    Returns:
        Tuple of (current_histogram, previous_histogram), or (None, None)
    """
    if intraday_bars is None or len(intraday_bars) < 30:
        return None, None

    try:
        macd_result = ta.macd(intraday_bars['Close'])
        if macd_result is None or macd_result.empty:
            return None, None

        hist_col = [c for c in macd_result.columns if 'MACDh' in c]
        if not hist_col:
            return None, None

        hist = macd_result[hist_col[0]]
        if len(hist) < 2 or pd.isna(hist.iloc[-1]) or pd.isna(hist.iloc[-2]):
            return None, None

        return float(hist.iloc[-1]), float(hist.iloc[-2])
    except Exception:
        return None, None


def calculate_spread_pct(bid: float, ask: float) -> Optional[float]:
    """
    Calculate bid-ask spread as a percentage.

    Args:
        bid: Current bid price
        ask: Current ask price

    Returns:
        Spread percentage (e.g., 0.011 means 0.011%), or None if invalid
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return None

    midpoint = (bid + ask) / 2
    return (ask - bid) / midpoint * 100

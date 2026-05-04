"""
Crypto Trading Bot - Candlestick Pattern Recognition
Japanese candlestick pattern recognition
"""

import pandas as pd
import numpy as np


def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect all candlestick patterns"""
    df = df.copy()

    # Basic candle properties
    df["body"] = df["close"] - df["open"]
    df["body_abs"] = df["body"].abs()
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_pct"] = df["body_abs"] / (df["high"] - df["low"]).replace(0, np.nan)
    df["is_bullish"] = (df["close"] > df["open"]).astype(bool)

    avg_body = df["body_abs"].rolling(window=20).mean()

    # === SINGLE CANDLE PATTERNS ===

    # Doji
    df["pat_doji"] = (df["body_abs"] < avg_body * 0.1).astype(int)

    # Hammer
    df["pat_hammer"] = (
        (df["lower_shadow"] > df["body_abs"] * 2) &
        (df["upper_shadow"] < df["body_abs"] * 0.5) &
        (df["body_abs"] > avg_body * 0.3)
    ).astype(int)

    # Inverted Hammer
    df["pat_inv_hammer"] = (
        (df["upper_shadow"] > df["body_abs"] * 2) &
        (df["lower_shadow"] < df["body_abs"] * 0.5) &
        (df["body_abs"] > avg_body * 0.3)
    ).astype(int)

    # Marubozu
    df["pat_marubozu"] = (
        (df["body_pct"] > 0.9) &
        (df["body_abs"] > avg_body * 1.5)
    ).astype(int)

    # Spinning Top
    df["pat_spinning_top"] = (
        (df["body_abs"] < avg_body * 0.5) &
        (df["upper_shadow"] > df["body_abs"]) &
        (df["lower_shadow"] > df["body_abs"])
    ).astype(int)

    # === TWO-CANDLE PATTERNS ===

    prev_bullish = df["is_bullish"].shift(1).fillna(False).astype(bool)
    prev2_bullish = df["is_bullish"].shift(2).fillna(False).astype(bool)

    # Bullish Engulfing
    df["pat_bull_engulfing"] = (
        (~prev_bullish) &                 # Previous bearish
        (df["is_bullish"]) &              # Current bullish
        (df["open"] < df["close"].shift(1)) &
        (df["close"] > df["open"].shift(1)) &
        (df["body_abs"] > df["body_abs"].shift(1))
    ).astype(int)

    # Bearish Engulfing
    df["pat_bear_engulfing"] = (
        (prev_bullish) &     # Previous bullish
        (~df["is_bullish"]) &              # Current bearish
        (df["open"] > df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1)) &
        (df["body_abs"] > df["body_abs"].shift(1))
    ).astype(int)

    # Piercing Line
    df["pat_piercing"] = (
        (~prev_bullish) &
        (df["is_bullish"]) &
        (df["open"] < df["low"].shift(1)) &
        (df["close"] > (df["open"].shift(1) + df["close"].shift(1)) / 2) &
        (df["close"] < df["open"].shift(1))
    ).astype(int)

    # Dark Cloud Cover
    df["pat_dark_cloud"] = (
        (prev_bullish) &
        (~df["is_bullish"]) &
        (df["open"] > df["high"].shift(1)) &
        (df["close"] < (df["open"].shift(1) + df["close"].shift(1)) / 2) &
        (df["close"] > df["close"].shift(1))
    ).astype(int)

    # Tweezer Top
    df["pat_tweezer_top"] = (
        (prev_bullish) &
        (~df["is_bullish"]) &
        (abs(df["high"] - df["high"].shift(1)) < avg_body * 0.1)
    ).astype(int)

    # Tweezer Bottom
    df["pat_tweezer_bottom"] = (
        (~prev_bullish) &
        (df["is_bullish"]) &
        (abs(df["low"] - df["low"].shift(1)) < avg_body * 0.1)
    ).astype(int)

    # === THREE-CANDLE PATTERNS ===

    # Morning Star
    df["pat_morning_star"] = (
        (~prev2_bullish) &                    # 1. bearish
        (df["body_abs"].shift(1) < avg_body * 0.5) &      # 2. tiny body
        (df["is_bullish"]) &                               # 3. bullish
        (df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2)
    ).astype(int)

    # Evening Star
    df["pat_evening_star"] = (
        (prev2_bullish) &                      # 1. bullish
        (df["body_abs"].shift(1) < avg_body * 0.5) &       # 2. tiny body
        (~df["is_bullish"]) &                              # 3. bearish
        (df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2)
    ).astype(int)

    # Three White Soldiers
    df["pat_three_white_soldiers"] = (
        (df["is_bullish"]) &
        (prev_bullish) &
        (prev2_bullish) &
        (df["close"] > df["close"].shift(1)) &
        (df["close"].shift(1) > df["close"].shift(2)) &
        (df["body_abs"] > avg_body * 0.7) &
        (df["body_abs"].shift(1) > avg_body * 0.7)
    ).astype(int)

    # Three Black Crows
    df["pat_three_black_crows"] = (
        (~df["is_bullish"]) &
        (~prev_bullish) &
        (~prev2_bullish) &
        (df["close"] < df["close"].shift(1)) &
        (df["close"].shift(1) < df["close"].shift(2)) &
        (df["body_abs"] > avg_body * 0.7) &
        (df["body_abs"].shift(1) > avg_body * 0.7)
    ).astype(int)

    # Cleanup - remove helper columns
    helper_cols = ["body", "body_abs", "upper_shadow", "lower_shadow",
                   "body_pct", "is_bullish"]
    df.drop(columns=helper_cols, inplace=True, errors="ignore")

    return df


def get_pattern_signals(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate patterns into a single signal value.
    Positive = bullish, Negative = bearish, 0 = neutral

    Returns:
        Series: signal between -1.0 and 1.0
    """
    bullish_patterns = [
        "pat_hammer", "pat_bull_engulfing", "pat_piercing",
        "pat_tweezer_bottom", "pat_morning_star", "pat_three_white_soldiers"
    ]
    bearish_patterns = [
        "pat_inv_hammer", "pat_bear_engulfing", "pat_dark_cloud",
        "pat_tweezer_top", "pat_evening_star", "pat_three_black_crows"
    ]

    # Weighted sum
    weights = {
        "pat_hammer": 0.6, "pat_bull_engulfing": 0.8, "pat_piercing": 0.5,
        "pat_tweezer_bottom": 0.5, "pat_morning_star": 0.9,
        "pat_three_white_soldiers": 1.0,
        "pat_inv_hammer": -0.4, "pat_bear_engulfing": -0.8,
        "pat_dark_cloud": -0.5, "pat_tweezer_top": -0.5,
        "pat_evening_star": -0.9, "pat_three_black_crows": -1.0,
    }

    signal = pd.Series(0.0, index=df.index)
    for col, weight in weights.items():
        if col in df.columns:
            signal += df[col] * weight

    # Normalize to [-1, 1] range
    max_bull = sum(w for w in weights.values() if w > 0)
    max_bear = abs(sum(w for w in weights.values() if w < 0))
    max_val = max(max_bull, max_bear)

    return (signal / max_val).clip(-1, 1)

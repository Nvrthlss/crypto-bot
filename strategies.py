"""
Crypto Trading Bot - Advanced Trading Strategies
DCA, Limit Orders, Kelly Criterion, Mean Reversion + Trend Following
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import config


@dataclass
class DCAOrder:
    """Dollar Cost Averaging order"""
    symbol: str
    total_budget: float
    n_steps: int
    step_interval_seconds: int
    completed_steps: int = 0
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    prices: list = field(default_factory=list)
    status: str = "ACTIVE"  # ACTIVE, COMPLETED, CANCELLED


class KellyCriterion:
    """
    Position sizing based on the Kelly criterion.

    The Kelly formula determines what percentage of capital
    should be risked given a specific win rate and reward/risk
    ratio in order to maximize long-term growth.

    f* = (p * b - q) / b

    where:
    - f* = optimal betting ratio
    - p = probability of winning
    - b = average profit / average loss
    - q = 1 - p (probability of losing)
    """

    @staticmethod
    def calculate(win_rate: float, avg_win: float, avg_loss: float,
                  fractional: float = 0.25) -> float:
        """
        Kelly-criterion calculation.

        Args:
            win_rate: Win rate (0-1)
            avg_win: Average win ($)
            avg_loss: Average loss ($ - pozitív szám!)
            fractional: Fractional Kelly (0.25 = quarter Kelly, more conservative)

        Returns:
            float: Recommended position size as % of capital (0-1)
        """
        if avg_loss == 0 or win_rate <= 0:
            return 0.01  # Minimum

        b = abs(avg_win) / abs(avg_loss)  # Reward/risk ratio
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b

        # Fractional Kelly (less agressive)
        position = max(0.005, min(kelly * fractional, 0.05))

        return position

    @staticmethod
    def from_trade_history(trades: list, fractional: float = 0.25) -> float:
        """Kelly számítás tényleges trade történetből"""
        if len(trades) < 10:
            return 0.01  # Not enough data

        pnls = [t.pnl for t in trades if t.status != "OPEN"]
        if not pnls:
            return 0.01

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        if not wins or not losses:
            return 0.01

        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        return KellyCriterion.calculate(win_rate, avg_win, avg_loss, fractional)


class StrategyEngine:
    """
    Advanced trading strategy combination.

    Strategies:
    1. Trend Following - trading in the direction of the trend
    2. Mean Reversion - trading in the opposite direction when prices reach extreme levels
    3. Breakout - a breakout following a period of reduced volatility
    4. Sentiment Contrarian - trading contrary to market sentiment
    """

    def __init__(self):
        self.active_strategy = None
        self.dca_orders: list[DCAOrder] = []

    def detect_market_regime(self, df: pd.DataFrame) -> dict:
        """
        Market regime detection.
        Different strategies for trending vs ranging markets.

        Returns:
            {"regime": str, "confidence": float, "details": dict}
        """
        latest = df.iloc[-1]
        lookback = df.tail(50)

        adx = latest.get("adx", 0)
        bb_width = latest.get("bb_width", 0)
        atr_pct = latest.get("atr_pct", 0)

        # Volatility
        volatility = lookback["close"].pct_change().std() * 100
        avg_volatility = df["close"].pct_change().rolling(200).std().iloc[-1] * 100 if len(df) > 200 else volatility

        # Trend strength
        ema_50 = latest.get("ema_50", 0)
        ema_200 = latest.get("ema_200", 0)
        ema_distance = abs(ema_50 - ema_200) / latest["close"] * 100 if latest["close"] > 0 else 0

        # Regime classification
        if adx > 30 and ema_distance > 2:
            regime = "STRONG_TREND"
            confidence = min(1.0, adx / 50)
            recommended = "trend_following"
        elif adx > 20 and ema_distance > 1:
            regime = "MODERATE_TREND"
            confidence = 0.6
            recommended = "trend_following"
        elif bb_width < df["bb_width"].rolling(100).quantile(0.2).iloc[-1] if "bb_width" in df.columns and len(df) > 100 else False:
            regime = "SQUEEZE"
            confidence = 0.7
            recommended = "breakout"
        elif volatility > avg_volatility * 1.5:
            regime = "HIGH_VOLATILITY"
            confidence = 0.5
            recommended = "mean_reversion"
        else:
            regime = "RANGING"
            confidence = 0.5
            recommended = "mean_reversion"

        return {
            "regime": regime,
            "confidence": confidence,
            "recommended_strategy": recommended,
            "details": {
                "adx": round(adx, 2),
                "bb_width": round(bb_width, 4) if bb_width else 0,
                "volatility": round(volatility, 4),
                "ema_distance_pct": round(ema_distance, 2),
            }
        }

    def trend_following_signal(self, df: pd.DataFrame) -> dict:
        """
        Trend Following strategy.
        The trend is your friend.

        Entry:
        - EMA crossover (fast > slow = long)
        - ADX > 25 (strong trend)
        - Price above/under Ichimoku cloud

        Exit:
        - Opposite EMA crossover
        - ADX falls below 20
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        signal = 0
        confidence = 0.0
        reasons = []

        # EMA crossover
        ema_9 = latest.get("ema_9", 0)
        ema_21 = latest.get("ema_21", 0)
        prev_ema_9 = prev.get("ema_9", 0)
        prev_ema_21 = prev.get("ema_21", 0)

        bullish_cross = ema_9 > ema_21 and prev_ema_9 <= prev_ema_21
        bearish_cross = ema_9 < ema_21 and prev_ema_9 >= prev_ema_21

        if bullish_cross:
            signal = 1
            confidence += 0.3
            reasons.append("EMA 9/21 bullish cross")
        elif bearish_cross:
            signal = -1
            confidence += 0.3
            reasons.append("EMA 9/21 bearish cross")

        # ADX confirmation
        adx = latest.get("adx", 0)
        if adx > 25:
            plus_di = latest.get("plus_di", 0)
            minus_di = latest.get("minus_di", 0)
            if plus_di > minus_di and signal >= 0:
                signal = max(signal, 1)
                confidence += 0.2
                reasons.append(f"ADX {adx:.0f} bullish")
            elif minus_di > plus_di and signal <= 0:
                signal = min(signal, -1)
                confidence += 0.2
                reasons.append(f"ADX {adx:.0f} bearish")

        # Golden/Death cross
        ema_50 = latest.get("ema_50", 0)
        ema_200 = latest.get("ema_200", 0)
        if ema_50 > ema_200 and signal >= 0:
            confidence += 0.15
            reasons.append("Above 200 EMA")
        elif ema_50 < ema_200 and signal <= 0:
            confidence += 0.15
            reasons.append("Below 200 EMA")

        # MACD momentum
        macd_hist = latest.get("macd_histogram", 0)
        prev_macd_hist = prev.get("macd_histogram", 0)
        if macd_hist > 0 and macd_hist > prev_macd_hist and signal >= 0:
            confidence += 0.1
            reasons.append("MACD momentum bullish")
        elif macd_hist < 0 and macd_hist < prev_macd_hist and signal <= 0:
            confidence += 0.1
            reasons.append("MACD momentum bearish")

        return {
            "strategy": "trend_following",
            "signal": signal,
            "confidence": min(1.0, confidence),
            "reasons": reasons
        }

    def mean_reversion_signal(self, df: pd.DataFrame) -> dict:
        """
        Mean Reversion Strategy.
        Price reverts to the mean when it deviates too far from it.

        Entry:
        - RSI < 25 (oversold) = BUY
        - RSI > 75 (overbought) = SELL
        - Price reaches the lower/upper Bollinger Band
        """
        latest = df.iloc[-1]
        signal = 0
        confidence = 0.0
        reasons = []

        # RSI
        rsi = latest.get("rsi", 50)
        if rsi < 25:
            signal = 1
            confidence += 0.35
            reasons.append(f"RSI extreme oversold ({rsi:.0f})")
        elif rsi < 30:
            signal = 1
            confidence += 0.2
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 75:
            signal = -1
            confidence += 0.35
            reasons.append(f"RSI extreme overbought ({rsi:.0f})")
        elif rsi > 70:
            signal = -1
            confidence += 0.2
            reasons.append(f"RSI overbought ({rsi:.0f})")

        # Bollinger Bands
        bb_pct = latest.get("bb_pct", 0.5)
        if bb_pct < 0.05:
            signal = max(signal, 1) if signal >= 0 else signal
            confidence += 0.25
            reasons.append(f"Below lower BB ({bb_pct:.2f})")
        elif bb_pct > 0.95:
            signal = min(signal, -1) if signal <= 0 else signal
            confidence += 0.25
            reasons.append(f"Above upper BB ({bb_pct:.2f})")

        # Stochastic
        stoch_k = latest.get("stoch_k", 50)
        if stoch_k < 15 and signal >= 0:
            confidence += 0.15
            reasons.append(f"Stochastic oversold ({stoch_k:.0f})")
        elif stoch_k > 85 and signal <= 0:
            confidence += 0.15
            reasons.append(f"Stochastic overbought ({stoch_k:.0f})")

        return {
            "strategy": "mean_reversion",
            "signal": signal,
            "confidence": min(1.0, confidence),
            "reasons": reasons
        }

    def breakout_signal(self, df: pd.DataFrame) -> dict:
        """
        Breakout strategy.
        After a Bollinger Band squeeze, we wait for the breakout.

        Signal:
        - BB squeeze → setup
        - Volume spike + direction → breakout confirmation
        """
        latest = df.iloc[-1]
        signal = 0
        confidence = 0.0
        reasons = []

        bb_width = latest.get("bb_width", 0)
        if bb_width == 0:
            return {"strategy": "breakout", "signal": 0, "confidence": 0, "reasons": []}

        # Squeeze detection
        bb_width_history = df["bb_width"].tail(50) if "bb_width" in df.columns else pd.Series()
        if len(bb_width_history) > 20:
            is_squeeze = bb_width < bb_width_history.quantile(0.2)
        else:
            is_squeeze = False

        if not is_squeeze:
            return {"strategy": "breakout", "signal": 0, "confidence": 0,
                    "reasons": ["No squeeze detected"]}

        # Volume spike
        volume_ratio = latest.get("volume", 0) / df["volume"].rolling(20).mean().iloc[-1] if df["volume"].rolling(20).mean().iloc[-1] > 0 else 1
        has_volume = volume_ratio > 1.5

        # Direction
        close = latest["close"]
        bb_upper = latest.get("bb_upper", close)
        bb_lower = latest.get("bb_lower", close)
        bb_middle = latest.get("bb_middle", close)

        if close > bb_middle and has_volume:
            signal = 1
            confidence = 0.4
            reasons.append("Bullish breakout from squeeze")
            if close > bb_upper:
                confidence += 0.2
                reasons.append("Breaking above upper BB")
        elif close < bb_middle and has_volume:
            signal = -1
            confidence = 0.4
            reasons.append("Bearish breakout from squeeze")
            if close < bb_lower:
                confidence += 0.2
                reasons.append("Breaking below lower BB")

        if has_volume:
            confidence += 0.15
            reasons.append(f"Volume spike ({volume_ratio:.1f}x)")

        return {
            "strategy": "breakout",
            "signal": signal,
            "confidence": min(1.0, confidence),
            "reasons": reasons
        }

    def get_combined_strategy_signal(self, df: pd.DataFrame,
                                      sentiment_score: float = 0.0) -> dict:
        """
        Combine all strategies based on market regime.

        1. Detect market regime
        2. Prioritize recommended strategy signal
        3. The other strategies serve as reinforcement
        4. Sentiment modifies final confidence
        """
        regime = self.detect_market_regime(df)
        recommended = regime["recommended_strategy"]

        # Minden stratégia jelzése
        trend = self.trend_following_signal(df)
        reversion = self.mean_reversion_signal(df)
        breakout = self.breakout_signal(df)

        strategies = {
            "trend_following": trend,
            "mean_reversion": reversion,
            "breakout": breakout,
        }

        primary = strategies.get(recommended, trend)

        # Strategy weights based on regime
        if recommended == "trend_following":
            weights = {"trend_following": 0.55, "mean_reversion": 0.20, "breakout": 0.25}
        elif recommended == "mean_reversion":
            weights = {"trend_following": 0.20, "mean_reversion": 0.55, "breakout": 0.25}
        else:  # breakout
            weights = {"trend_following": 0.25, "mean_reversion": 0.20, "breakout": 0.55}

        # Weighted signal
        weighted_signal = sum(
            strategies[s]["signal"] * w for s, w in weights.items()
        )
        weighted_confidence = sum(
            strategies[s]["confidence"] * w for s, w in weights.items()
            if strategies[s]["signal"] != 0
        )

        # Final signal
        if weighted_signal > 0.2:
            final_signal = 1
        elif weighted_signal < -0.2:
            final_signal = -1
        else:
            final_signal = 0

        # Sentiment adjustment
        if sentiment_score != 0:
            if (final_signal == 1 and sentiment_score > 0.2) or \
               (final_signal == -1 and sentiment_score < -0.2):
                weighted_confidence = min(1.0, weighted_confidence * 1.15)
            elif (final_signal == 1 and sentiment_score < -0.3) or \
                 (final_signal == -1 and sentiment_score > 0.3):
                weighted_confidence *= 0.8

        # Collected reasons
        all_reasons = []
        for name, strat in strategies.items():
            if strat["signal"] != 0 and strat["reasons"]:
                all_reasons.extend([f"[{name}] {r}" for r in strat["reasons"]])

        return {
            "signal": final_signal,
            "confidence": round(weighted_confidence, 4),
            "regime": regime,
            "primary_strategy": recommended,
            "strategies": strategies,
            "reasons": all_reasons,
            "sentiment_impact": round(sentiment_score, 4),
        }

    def create_dca_order(self, symbol: str, total_budget: float,
                         n_steps: int = 5,
                         interval_seconds: int = 3600) -> DCAOrder:
        """
        Create DCA order.
        Instead of buying at once, we average over N steps.
        """
        order = DCAOrder(
            symbol=symbol,
            total_budget=total_budget,
            n_steps=n_steps,
            step_interval_seconds=interval_seconds,
        )
        self.dca_orders.append(order)
        return order

    def calculate_limit_price(self, current_price: float, side: str,
                               order_book: dict = None,
                               slippage_bps: int = 5) -> float:
        """
        Calculate limit order price.

        Instead of buying at market price (worse price),
        we set a slightly better limit price.

        Args:
            slippage_bps: Basis points offset (5 = 0.05%)
        """
        offset = current_price * slippage_bps / 10000

        if side == "BUY":
            # Place slightly below current price
            limit_price = current_price - offset
        else:
            # Place slightly above current price
            limit_price = current_price + offset

        # Use order book if available
        if order_book:
            if side == "BUY" and order_book.get("bids"):
                best_bid = float(order_book["bids"][0][0])
                limit_price = max(limit_price, best_bid)
            elif side == "SELL" and order_book.get("asks"):
                best_ask = float(order_book["asks"][0][0])
                limit_price = min(limit_price, best_ask)

        return round(limit_price, 2)

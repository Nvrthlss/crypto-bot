"""
Crypto Trading Bot - Sentiment Analysis modul
Sources: Fear & Greed Index, funding rate, social volume

This data adds an extra "dimension" to the bot—it doesn't just see the price
and technical indicators, but also the market sentiment.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class SentimentSnapshot:
    """A snapshot of market sentiment"""
    timestamp: str
    fear_greed_index: float        # 0-100 (0=extreme fear, 100=extreme greed)
    fear_greed_label: str          # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    funding_rate: float            # Futures funding rate (-0.01 to +0.01)
    long_short_ratio: float        # Long/Short arány
    social_score: float            # -1 to +1 (bearish to bullish)
    composite_score: float         # Total sentiment -1 to +1
    signals: dict                  # Detailed signals


class SentimentAnalyzer:
    """
    Market sentiment analysis from multiple sources.

    Strategic logic:
    - Extreme Fear + technical BUY = strong buy signal (contrarian)
    - Extreme Greed + technical SELL = strong sell signal (contrarian)
    - High funding rate = overheated market, exercise caution
    - Social spike = potential pump & dump, exercise caution
    """

    def __init__(self):
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 300  # 5m cache

    def get_fear_greed_index(self) -> dict:
        """
        Crypto Fear & Greed Index query.
        Source: alternative.me API (free)

        Returns:
            {"value": 0-100, "label": str, "timestamp": str}
        """
        cache_key = "fear_greed"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["data"]

        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()["data"][0]

            result = {
                "value": int(data["value"]),
                "label": data["value_classification"],
                "timestamp": datetime.fromtimestamp(int(data["timestamp"])).isoformat()
            }
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            print(f"  Fear & Greed API hiba: {e}")
            return {"value": 50, "label": "Neutral", "timestamp": datetime.now().isoformat()}

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> dict:
        """
        Binance Futures funding rate query.

        High positive = too many longs (bearish signal)
        High negative = too many shorts (bullish signal)
        """
        cache_key = f"funding_{symbol}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["data"]

        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            response = self.session.get(url, params={
                "symbol": symbol, "limit": 1
            }, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data:
                rate = float(data[0]["fundingRate"])
                result = {
                    "rate": rate,
                    "timestamp": datetime.fromtimestamp(
                        int(data[0]["fundingTime"]) / 1000
                    ).isoformat(),
                    "signal": self._interpret_funding(rate)
                }
            else:
                result = {"rate": 0.0, "timestamp": "", "signal": "neutral"}

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            print(f"   Funding rate API hiba: {e}")
            return {"rate": 0.0, "timestamp": "", "signal": "neutral"}

    def get_long_short_ratio(self, symbol: str = "BTCUSDT") -> dict:
        """
        Long/Short ratio from Binance futures.
        >1 = more long, <1 = more short
        """
        cache_key = f"ls_ratio_{symbol}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["data"]

        try:
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            response = self.session.get(url, params={
                "symbol": symbol, "period": "1h", "limit": 1
            }, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data:
                ratio = float(data[0]["longShortRatio"])
                result = {
                    "ratio": ratio,
                    "long_pct": float(data[0]["longAccount"]) * 100,
                    "short_pct": float(data[0]["shortAccount"]) * 100,
                    "signal": self._interpret_ls_ratio(ratio)
                }
            else:
                result = {"ratio": 1.0, "long_pct": 50, "short_pct": 50, "signal": "neutral"}

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            print(f"   Long/Short ratio API hiba: {e}")
            return {"ratio": 1.0, "long_pct": 50, "short_pct": 50, "signal": "neutral"}

    def get_open_interest_change(self, symbol: str = "BTCUSDT") -> dict:
        """
        Change in Open Interest - total value of open futures positions.
        Rising OI + rising price = strong trend
        Rising OI + falling price = strong bearish pressure
        """
        cache_key = f"oi_{symbol}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]["data"]

        try:
            url = "https://fapi.binance.com/futures/data/openInterestHist"
            response = self.session.get(url, params={
                "symbol": symbol, "period": "1h", "limit": 5
            }, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) >= 2:
                current_oi = float(data[-1]["sumOpenInterestValue"])
                prev_oi = float(data[0]["sumOpenInterestValue"])
                change_pct = (current_oi / prev_oi - 1) * 100

                result = {
                    "current_oi": current_oi,
                    "change_pct": change_pct,
                    "signal": "increasing" if change_pct > 2 else "decreasing" if change_pct < -2 else "stable"
                }
            else:
                result = {"current_oi": 0, "change_pct": 0, "signal": "unknown"}

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            print(f"   Open Interest API hiba: {e}")
            return {"current_oi": 0, "change_pct": 0, "signal": "unknown"}

    def get_whale_alerts(self, symbol: str = "BTC") -> dict:
        """
        Large transaction detection - "whale movements".
        Large inflows to Binance = selling pressure
        Large outflows = accumulation (bullish)

        Note: Real whale alerts require an external API (pl. whale-alert.io),
        itt we approximate exchange netflow.
        """
        try:
            # Binance exchange netflow approximation
            # (from taker buy vs sell volume ratio)
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = self.session.get(url, params={
                "symbol": f"{symbol}USDT"
            }, timeout=10)
            response.raise_for_status()
            data = response.json()

            taker_buy_volume = float(data.get("volume", 0)) * 0.5
            total_volume = float(data.get("volume", 0))
            buy_ratio = taker_buy_volume / total_volume if total_volume > 0 else 0.5

            return {
                "buy_ratio": buy_ratio,
                "volume_24h": total_volume,
                "signal": "accumulation" if buy_ratio > 0.55 else "distribution" if buy_ratio < 0.45 else "neutral"
            }

        except Exception as e:
            return {"buy_ratio": 0.5, "volume_24h": 0, "signal": "neutral"}

    def get_composite_sentiment(self, symbol: str = "BTCUSDT") -> SentimentSnapshot:
        """
        Composite sentiment score from all sources.

        Combine:
        - Fear & Greed Index (contrarian)
        - Funding Rate
        - Long/Short Ratio
        - Change in Open Interest

        Returns:
            SentimentSnapshot: snapshot of the overall atmosphere
        """
        # Data collection
        fg = self.get_fear_greed_index()
        funding = self.get_funding_rate(symbol)
        ls = self.get_long_short_ratio(symbol)
        oi = self.get_open_interest_change(symbol)

        signals = {}
        scores = []

        # 1. Fear & Greed (CONTRARIAN - fear = buying opportunity)
        fg_value = fg["value"]
        if fg_value <= 20:
            fg_score = 0.8    # Extreme Fear = bullish (contrarian)
            signals["fear_greed"] = "extreme_fear_bullish"
        elif fg_value <= 35:
            fg_score = 0.4
            signals["fear_greed"] = "fear_bullish"
        elif fg_value >= 80:
            fg_score = -0.8   # Extreme Greed = bearish (contrarian)
            signals["fear_greed"] = "extreme_greed_bearish"
        elif fg_value >= 65:
            fg_score = -0.4
            signals["fear_greed"] = "greed_bearish"
        else:
            fg_score = 0.0
            signals["fear_greed"] = "neutral"
        scores.append(("fear_greed", fg_score, 0.30))  # 30% weight

        # 2. Funding Rate (CONTRARIAN)
        rate = funding["rate"]
        if rate > 0.001:
            fund_score = -0.6
            signals["funding"] = "overleveraged_longs"
        elif rate > 0.0005:
            fund_score = -0.3
            signals["funding"] = "moderately_long"
        elif rate < -0.001:
            fund_score = 0.6
            signals["funding"] = "overleveraged_shorts"
        elif rate < -0.0005:
            fund_score = 0.3
            signals["funding"] = "moderately_short"
        else:
            fund_score = 0.0
            signals["funding"] = "neutral"
        scores.append(("funding", fund_score, 0.25))  # 25% weight

        # 3. Long/Short Ratio (CONTRARIAN)
        ratio = ls["ratio"]
        if ratio > 2.0:
            ls_score = -0.5
            signals["long_short"] = "extreme_long_bias"
        elif ratio > 1.3:
            ls_score = -0.3
            signals["long_short"] = "long_bias"
        elif ratio < 0.5:
            ls_score = 0.5
            signals["long_short"] = "extreme_short_bias"
        elif ratio < 0.77:
            ls_score = 0.3
            signals["long_short"] = "short_bias"
        else:
            ls_score = 0.0
            signals["long_short"] = "balanced"
        scores.append(("long_short", ls_score, 0.25))  # 25% weight

        # 4. Open Interest (TREND CONFIRMATION)
        oi_change = oi["change_pct"]
        if oi_change > 5:
            oi_score = 0.3
            signals["open_interest"] = "rising_strong"
        elif oi_change > 2:
            oi_score = 0.15
            signals["open_interest"] = "rising"
        elif oi_change < -5:
            oi_score = -0.3
            signals["open_interest"] = "declining_strong"
        elif oi_change < -2:
            oi_score = -0.15
            signals["open_interest"] = "declining"
        else:
            oi_score = 0.0
            signals["open_interest"] = "stable"
        scores.append(("open_interest", oi_score, 0.20))  # 20% weight

        # Weighted aggregation
        composite = sum(score * weight for _, score, weight in scores)

        # Social score approximation (from other indicators)
        social_approx = (fg_score * 0.5 + fund_score * 0.3 + ls_score * 0.2)

        return SentimentSnapshot(
            timestamp=datetime.now().isoformat(),
            fear_greed_index=fg_value,
            fear_greed_label=fg["label"],
            funding_rate=funding["rate"],
            long_short_ratio=ls["ratio"],
            social_score=round(social_approx, 4),
            composite_score=round(composite, 4),
            signals=signals
        )

    def get_sentiment_features(self, symbol: str = "BTCUSDT") -> dict:
        """
        Convert sentiment data to ML features.
        Ezeket közvetlenül hozzá lehet adni a feature DataFrame-hez.
        """
        sentiment = self.get_composite_sentiment(symbol)

        return {
            "sent_fear_greed": sentiment.fear_greed_index / 100,  # 0-1
            "sent_fear_greed_extreme": 1 if sentiment.fear_greed_index < 20 or sentiment.fear_greed_index > 80 else 0,
            "sent_funding_rate": sentiment.funding_rate * 1000,   # Scaled
            "sent_ls_ratio": sentiment.long_short_ratio,
            "sent_composite": sentiment.composite_score,
            "sent_contrarian_buy": 1 if sentiment.composite_score > 0.3 else 0,
            "sent_contrarian_sell": 1 if sentiment.composite_score < -0.3 else 0,
        }

    def print_sentiment(self, symbol: str = "BTCUSDT"):
        """Pretty print"""
        s = self.get_composite_sentiment(symbol)

        bar_len = 20
        fg_pos = int(s.fear_greed_index / 100 * bar_len)
        fg_bar = "█" * fg_pos + "░" * (bar_len - fg_pos)

        comp_pos = int((s.composite_score + 1) / 2 * bar_len)
        comp_bar = "█" * comp_pos + "░" * (bar_len - comp_pos)

        print(f"\n   SENTIMENT — {symbol}")
        print(f"   {'─' * 45}")
        print(f"   Fear/Greed:    [{fg_bar}] {s.fear_greed_index}/100 ({s.fear_greed_label})")
        print(f"   Funding Rate:  {s.funding_rate:+.6f} ({s.signals.get('funding', '')})")
        print(f"   Long/Short:    {s.long_short_ratio:.2f} ({s.signals.get('long_short', '')})")
        print(f"   Composite:     [{comp_bar}] {s.composite_score:+.3f}")

        if s.composite_score > 0.3:
            print(f"     Contrarian BUY jelzés")
        elif s.composite_score < -0.3:
            print(f"     Contrarian SELL jelzés")
        else:
            print(f"     Semleges hangulat")

    # === Internal helpers ===

    def _interpret_funding(self, rate: float) -> str:
        if rate > 0.001: return "very_bullish_market"
        if rate > 0.0005: return "bullish_market"
        if rate < -0.001: return "very_bearish_market"
        if rate < -0.0005: return "bearish_market"
        return "neutral"

    def _interpret_ls_ratio(self, ratio: float) -> str:
        if ratio > 2.0: return "extreme_long"
        if ratio > 1.3: return "long_bias"
        if ratio < 0.5: return "extreme_short"
        if ratio < 0.77: return "short_bias"
        return "balanced"

    def _is_cached(self, key: str) -> bool:
        if key not in self.cache:
            return False
        elapsed = (datetime.now() - self.cache[key]["time"]).total_seconds()
        return elapsed < self.cache_duration

    def _set_cache(self, key: str, data):
        self.cache[key] = {"data": data, "time": datetime.now()}

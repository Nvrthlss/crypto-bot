"""
Crypto Trading Bot - Multi-Timeframe Analyzer
"""

import pandas as pd
import numpy as np
from indicators import add_all_indicators
from patterns import detect_all_patterns, get_pattern_signals
from ml_model import TradingMLModel
import config


class MultiTimeframeAnalyzer:
    def __init__(self):
        self.models = {}  # {(symbol, timeframe): TradingMLModel}
        self.trained_pairs = set()

    def _get_model_key(self, symbol: str, timeframe: str) -> tuple:
        return (symbol, timeframe)

    def _get_or_create_model(self, symbol: str, timeframe: str) -> TradingMLModel:
        """Get or create model"""
        key = self._get_model_key(symbol, timeframe)
        if key not in self.models:
            self.models[key] = TradingMLModel()
        return self.models[key]

    def train_all(self, client, symbol: str) -> dict:
        """
        Train all timeframes for a coin.

        Args:
            client: BinanceClient vagy PaperTradingClient
            symbol: pl. "BTCUSDT"

        Returns:
            dict: timeframe -> training metrics
        """
        results = {}
        print(f"\n{'━' * 50}")
        print(f"{symbol} model training")
        print(f"{'━' * 50}")

        for tf_name, tf_interval in config.TIMEFRAMES.items():
            lookback = config.LOOKBACK_PERIODS.get(tf_interval, 500)
            print(f"\n   {tf_name.upper()} ({tf_interval}) - {lookback} candle(s)")

            try:
                df = client.get_klines(
                    symbol=symbol,
                    interval=tf_interval,
                    limit=lookback
                )

                df = add_all_indicators(df)
                df = detect_all_patterns(df)

                model = self._get_or_create_model(symbol, tf_interval)
                metrics = model.train(df)
                results[tf_interval] = metrics

                print(f"    Accuracy: {metrics['ensemble_accuracy']:.1%}")

            except Exception as e:
                print(f"    Error: {e}")
                results[tf_interval] = None

        self.trained_pairs.add(symbol)
        return results

    def analyze_symbol(self, client, symbol: str) -> dict:
        """
        Full multi-timeframe analysis for a coin.

        Returns:
            dict: combined signal + TF
        """
        tf_results = {}

        for tf_name, tf_interval in config.TIMEFRAMES.items():
            try:
                lookback = config.LOOKBACK_PERIODS.get(tf_interval, 500)
                df = client.get_klines(
                    symbol=symbol,
                    interval=tf_interval,
                    limit=lookback
                )

                df = add_all_indicators(df)
                df = detect_all_patterns(df)

                # Pattern signal
                pat_signal = get_pattern_signals(df).iloc[-1]

                # ML prediction
                model = self._get_or_create_model(symbol, tf_interval)
                if model.is_trained:
                    ml_signal, ml_confidence = model.predict(df)
                else:
                    ml_signal, ml_confidence = 0, 0.0

                # Indicator snapshot
                latest = df.iloc[-1]
                indicators = {
                    "rsi": round(latest.get("rsi", 0), 2),
                    "macd_hist": round(latest.get("macd_histogram", 0), 4),
                    "adx": round(latest.get("adx", 0), 2),
                    "bb_pct": round(latest.get("bb_pct", 0), 4),
                    "ema_trend": "bullish" if latest.get("ema_50", 0) > latest.get("ema_200", 0) else "bearish",
                }

                # Trend direction (simple indicator-based voting)
                trend_votes = 0
                if latest.get("rsi", 50) > 50: trend_votes += 1
                else: trend_votes -= 1
                if latest.get("macd_histogram", 0) > 0: trend_votes += 1
                else: trend_votes -= 1
                if latest.get("ema_9", 0) > latest.get("ema_21", 0): trend_votes += 1
                else: trend_votes -= 1
                if latest.get("close", 0) > latest.get("ema_200", 0): trend_votes += 1
                else: trend_votes -= 1
                if latest.get("adx", 0) > 25:
                    if latest.get("plus_di", 0) > latest.get("minus_di", 0):
                        trend_votes += 2
                    else:
                        trend_votes -= 2

                # Normalized trend score [-1, 1]
                trend_score = max(-1, min(1, trend_votes / 6))

                tf_results[tf_name] = {
                    "timeframe": tf_interval,
                    "ml_signal": ml_signal,
                    "ml_confidence": ml_confidence,
                    "pattern_signal": pat_signal,
                    "trend_score": trend_score,
                    "indicators": indicators,
                    "price": latest["close"],
                    "atr": latest.get("atr", 0),
                }

            except Exception as e:
                tf_results[tf_name] = {
                    "timeframe": tf_interval,
                    "error": str(e),
                    "ml_signal": 0,
                    "ml_confidence": 0,
                    "trend_score": 0,
                }

        # === COMBINED SIGNAL ===
        combined = self._combine_signals(tf_results)
        combined["symbol"] = symbol
        combined["timeframe_details"] = tf_results

        return combined

    def _combine_signals(self, tf_results: dict) -> dict:
        """
        Aggregate timeframe signals via weighted voting.
        Works dynamically with any number of timeframes.

        Rules:
        1. The "primary" TF gives the BUY/SELL signal
           (in case no primary, using the "scalp" TF)
        2. A higher TFs strengthen or weaken
        3. If higher TFs don't go together, confidence decreases
        4. If all go together, confidence grows
        """
        # Entry signal: primary, or scalp
        entry_tf = tf_results.get("primary", tf_results.get("scalp", {}))
        signal = entry_tf.get("ml_signal", 0)
        confidence = entry_tf.get("ml_confidence", 0)

        # Scalp confirmation (if exists, and not entry)
        scalp = tf_results.get("scalp", {})
        if "primary" in tf_results and scalp.get("ml_signal", 0) == signal and signal != 0:
            confidence = min(1.0, confidence * 1.05)

        # Weighted trend calculation - dynamic for both TF
        weighted_trend = 0.0
        for tf_name, tf_data in tf_results.items():
            if "error" in tf_data:
                continue
            weight = config.TIMEFRAME_WEIGHTS.get(tf_name, 0)
            trend = tf_data.get("trend_score", 0)
            weighted_trend += trend * weight

        # Collect higher TF data (everything that's not scalp/primary)
        higher_tf_names = [name for name in config.TIMEFRAMES.keys()
                          if name not in ("scalp", "primary")]
        higher_tfs = [tf_results.get(name, {}) for name in higher_tf_names]
        higher_trends = [tf.get("trend_score", 0) for tf in higher_tfs
                        if "error" not in tf]

        # Alignment calculation
        if signal != 0 and higher_trends:
            if signal == 1:
                agreeing = sum(1 for t in higher_trends if t > 0.2)
                disagreeing = sum(1 for t in higher_trends if t < -0.2)
            else:  # signal == -1
                agreeing = sum(1 for t in higher_trends if t < -0.2)
                disagreeing = sum(1 for t in higher_trends if t > 0.2)

            agree_ratio = agreeing / len(higher_trends)
            disagree_ratio = disagreeing / len(higher_trends)

            # All agree → +25% confidence
            if agree_ratio >= 0.8:
                confidence = min(1.0, confidence * 1.25)
                alignment = "STRONG_ALIGNMENT"
            # Majority agrees → +10%
            elif agree_ratio >= 0.5:
                confidence = min(1.0, confidence * 1.10)
                alignment = "PARTIAL_ALIGNMENT"
            # Majority disagrees → -30%
            elif disagree_ratio >= 0.5:
                confidence *= 0.70
                alignment = "DIVERGENCE"
            # Mixed → none
            else:
                alignment = "NEUTRAL"
        else:
            alignment = "NEUTRAL"

        # Pattern confirmation (from primary)
        primary_pattern = entry_tf.get("pattern_signal", 0)
        if signal == 1 and primary_pattern > 0.3:
            confidence = min(1.0, confidence * 1.10)
        elif signal == -1 and primary_pattern < -0.3:
            confidence = min(1.0, confidence * 1.10)
        elif signal != 0 and abs(primary_pattern) > 0.3 and np.sign(primary_pattern) != signal:
            confidence *= 0.85

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "weighted_trend": round(weighted_trend, 4),
            "alignment": alignment,
            "price": entry_tf.get("price", 0),
            "atr": entry_tf.get("atr", 0),
        }


class MultiCoinScanner:
    """
    Parallel multi-coin scanning and ranking.

    Workflow:
    1. Analyzes all coins multi-timeframe-en
    2. Ranks signals based on confidence
    3. Correlation-filtering
    4. Returns top trading opportunities
    """

    def __init__(self, client):
        self.client = client
        self.analyzer = MultiTimeframeAnalyzer()
        self.correlation_cache = {}

    def train_all_models(self):
        """Teaching models for all coins"""
        print("\n" + "=" * 60)
        print(" TRAINING ALL MODELS")
        print(f"   {len(config.TRADING_PAIRS)} coin × "
              f"{len(config.TIMEFRAMES)} timeframe = "
              f"{len(config.TRADING_PAIRS) * len(config.TIMEFRAMES)} model")
        print("=" * 60)

        all_metrics = {}
        for symbol in config.TRADING_PAIRS:
            metrics = self.analyzer.train_all(self.client, symbol)
            all_metrics[symbol] = metrics

        # Summary
        print("\n" + "=" * 60)
        print(" TRAINING SUMMARY")
        print(f"{'Coin':<12} {'1h':>8} {'4h':>8} {'1d':>8}")
        print("─" * 40)
        for symbol, metrics in all_metrics.items():
            vals = []
            for tf in config.TIMEFRAMES.values():
                m = metrics.get(tf)
                if m:
                    vals.append(f"{m['ensemble_accuracy']:.1%}")
                else:
                    vals.append("  N/A")
            print(f"{symbol:<12} {'  '.join(vals)}")
        print("=" * 60)

        return all_metrics

    def scan(self) -> list[dict]:
        """
        Scanning all coins, ranking based on confidence

        Returns:
            Sorted list about best opportunities
        """
        opportunities = []

        for symbol in config.TRADING_PAIRS:
            if symbol not in self.analyzer.trained_pairs:
                continue

            try:
                result = self.analyzer.analyze_symbol(self.client, symbol)

                # Coin-specific 
                overrides = config.PAIR_OVERRIDES.get(symbol, {})
                min_conf = overrides.get("min_confidence", config.ML_CONFIDENCE_THRESHOLD)

                result["min_confidence"] = min_conf
                result["tradeable"] = (
                    result["signal"] != 0 and
                    result["confidence"] >= min_conf
                )

                opportunities.append(result)

            except Exception as e:
                print(f"  ⚠️ {symbol} scan hiba: {e}")

        # Ranking
        opportunities.sort(
            key=lambda x: (x.get("tradeable", False), x.get("confidence", 0)),
            reverse=True
        )

        return opportunities

    def get_correlation(self, client, sym1: str, sym2: str,
                        interval: str = "1h", periods: int = 100) -> float:
        """Price correlation between two coins"""
        cache_key = f"{sym1}_{sym2}_{interval}"
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]

        try:
            df1 = client.get_klines(symbol=sym1, interval=interval, limit=periods)
            df2 = client.get_klines(symbol=sym2, interval=interval, limit=periods)

            returns1 = df1["close"].pct_change().dropna()
            returns2 = df2["close"].pct_change().dropna()

            common = returns1.index.intersection(returns2.index)
            if len(common) < 20:
                return 0.0

            corr = returns1.loc[common].corr(returns2.loc[common])
            self.correlation_cache[cache_key] = corr
            return corr

        except Exception:
            return 0.0

    def filter_correlated(self, opportunities: list[dict]) -> list[dict]:
        """
        Filter highly correlated coins.
        If both BTC and ETH gives BUY signal and >85% correlation,
        can only hold the higher confidence one.
        """
        if len(opportunities) <= 1:
            return opportunities

        tradeable = [o for o in opportunities if o.get("tradeable")]
        non_tradeable = [o for o in opportunities if not o.get("tradeable")]

        filtered = []
        excluded_symbols = set()

        for opp in tradeable:
            sym = opp["symbol"]
            if sym in excluded_symbols:
                continue

            # Scan if there is a highly correlating coin yet to be filtered    
            for other in tradeable:
                other_sym = other["symbol"]
                if other_sym == sym or other_sym in excluded_symbols:
                    continue

                # Filter only if signals have matching direction
                if opp["signal"] == other["signal"]:
                    corr = self.get_correlation(
                        self.client, sym, other_sym
                    )
                    if abs(corr) > config.MAX_CORRELATION:
                        # Filtering out the weaker one
                        if other["confidence"] < opp["confidence"]:
                            excluded_symbols.add(other_sym)
                        else:
                            excluded_symbols.add(sym)
                            break

            if sym not in excluded_symbols:
                filtered.append(opp)

        return filtered + non_tradeable

    def print_scan_results(self, opportunities: list[dict]):
        """Pretty-print scan results"""
        signal_icons = {1: "🟢 BUY ", -1: "🔴 SELL", 0: "⚪ HOLD"}
        align_icons = {
            "STRONG_ALIGNMENT": "✅",
            "PARTIAL_ALIGNMENT": "🔶",
            "DIVERGENCE": "⚠️",
            "NEUTRAL": "➖"
        }

        print(f"\n{'━' * 65}")
        print(f"{'Coin':<10} {'Jelzés':<10} {'Konf':>6} {'Trend':>7} "
              f"{'Align':<5} {'Ár':>12} {'Trade?'}")
        print(f"{'━' * 65}")

        for opp in opportunities:
            sym = opp["symbol"].replace("USDT", "")
            sig = signal_icons.get(opp["signal"], "?")
            conf = f"{opp['confidence']:.1%}"
            trend = f"{opp['weighted_trend']:+.2f}"
            align = align_icons.get(opp["alignment"], "?")
            price = f"${opp['price']:>10,.2f}"
            tradeable = "✅ YES" if opp.get("tradeable") else "❌ NO"

            print(f"{sym:<10} {sig}  {conf:>6} {trend:>7} "
                  f"  {align}   {price} {tradeable}")

        print(f"{'━' * 65}")

        # Summary
        tradeables = [o for o in opportunities if o.get("tradeable")]
        if tradeables:
            print(f"\n {len(tradeables)} tradeable signal(s):")
            for t in tradeables:
                details = t.get("timeframe_details", {})

                tf_parts = []
                for tf_name in config.TIMEFRAMES.keys():
                    tf_data = details.get(tf_name, {})
                    if "error" not in tf_data:
                        tf_interval = config.TIMEFRAMES[tf_name]
                        trend = tf_data.get("trend_score", 0)
                        tf_parts.append(f"{tf_interval}: {trend:+.2f}")

                print(f"   {t['symbol']}: {signal_icons[t['signal']]} "
                      f"konf={t['confidence']:.1%} "
                      f"[{' | '.join(tf_parts)}]")
        else:
            print("\n No tradeable signal(s).")

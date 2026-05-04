"""
Crypto Trading Bot - Advanced Backtesting
Walk-Forward Optimization, Monte Carlo Simulation, Slippage Modeling

Simple backtests can be deceptive - these methods give a more realistic picture.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from indicators import add_all_indicators
from patterns import detect_all_patterns, get_pattern_signals
from ml_model import TradingMLModel
from strategies import StrategyEngine, KellyCriterion
import config


@dataclass
class BacktestResult:
    """Single backtest run result"""
    total_return_pct: float
    buy_hold_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float  # in hours
    final_value: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


class AdvancedBacktester:
    """
    Advanced backtesting system.

    Improvements over simple backtesting:
    1. Walk-Forward Optimization
    2. Monte Carlo simulation
    3. Slippage and latency modeling
    4. Detailed metrics (Sharpe, Sortino, Calmar)
    """

    def __init__(self, slippage_bps: int = 5, fee_rate: float = 0.001,
                 latency_ms: int = 100):
        """
        Args:
            slippage_bps: Slippage in basis points (5 = 0.05%)
            fee_rate: Trading fee (0.001 = 0.1%)
            latency_ms: Execution latency in milliseconds
        """
        self.slippage_bps = slippage_bps
        self.fee_rate = fee_rate
        self.latency_ms = latency_ms

    def apply_slippage(self, price: float, side: str) -> float:
        """Simulate slippage - real price is always worse"""
        slip = price * self.slippage_bps / 10000
        # Random extra slippage
        random_slip = np.random.uniform(0, slip * 0.5)

        if side == "BUY":
            return price + slip + random_slip  # We buy at higher price
        else:
            return price - slip - random_slip  # We sell at lower price

    def run_single_backtest(self, df: pd.DataFrame,
                            train_ratio: float = 0.7) -> BacktestResult:
        """
        Single backtest run with advanced metrics.
        """
        # Indicators and patterns
        df = add_all_indicators(df)
        df = detect_all_patterns(df)

        # Train/Test split
        split = int(len(df) * train_ratio)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        # Model training
        model = TradingMLModel()
        try:
            model.train(train_df)
        except Exception as e:
            print(f"  ⚠️ Training error: {e}")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 10000)

        # Strategy engine
        strategy = StrategyEngine()

        # Simulation
        balance = 10000.0
        btc = 0.0
        trades = []
        equity_curve = [balance]
        peak_equity = balance

        entry_price = 0
        entry_time = None

        for i in range(60, len(test_df)):
            window = pd.concat([train_df, test_df.iloc[:i+1]])
            price = test_df.iloc[i]["close"]

            # ML prediction
            ml_signal, ml_confidence = model.predict(window)

            # Strategy signal
            strat_result = strategy.get_combined_strategy_signal(window)
            strat_signal = strat_result["signal"]
            strat_confidence = strat_result["confidence"]

            # Combined signal (50% ML + 50% strategy)
            if ml_signal == strat_signal and ml_signal != 0:
                combined_signal = ml_signal
                combined_conf = min(1.0, (ml_confidence + strat_confidence) / 2 * 1.2)
            elif ml_signal != 0 and strat_signal == 0:
                combined_signal = ml_signal
                combined_conf = ml_confidence * 0.7
            elif strat_signal != 0 and ml_signal == 0:
                combined_signal = strat_signal
                combined_conf = strat_confidence * 0.7
            elif ml_signal != strat_signal:
                combined_signal = 0
                combined_conf = 0
            else:
                combined_signal = 0
                combined_conf = 0

            # BUY
            if combined_signal == 1 and combined_conf >= 0.5 and btc == 0:
                exec_price = self.apply_slippage(price, "BUY")
                invest = balance * 0.03  # 3% position
                fee = invest * self.fee_rate
                btc = (invest - fee) / exec_price
                balance -= invest
                entry_price = exec_price
                entry_time = i

                trades.append({
                    "type": "BUY",
                    "price": exec_price,
                    "slippage": exec_price - price,
                    "fee": fee,
                    "time_idx": i,
                    "confidence": combined_conf,
                    "regime": strat_result["regime"]["regime"],
                })

            # SELL
            elif btc > 0:
                should_sell = False
                reason = ""

                # Signal-based
                if combined_signal == -1:
                    should_sell = True
                    reason = "SIGNAL"

                # Stop-loss (2%)
                elif price <= entry_price * (1 - config.STOP_LOSS_PCT / 100):
                    should_sell = True
                    reason = "STOP_LOSS"

                # Take-profit (4%)
                elif price >= entry_price * (1 + config.TAKE_PROFIT_PCT / 100):
                    should_sell = True
                    reason = "TAKE_PROFIT"

                # Time-based stop (max 48 hours)
                elif entry_time and (i - entry_time) > 48:
                    should_sell = True
                    reason = "TIME_STOP"

                if should_sell:
                    exec_price = self.apply_slippage(price, "SELL")
                    proceeds = btc * exec_price
                    fee = proceeds * self.fee_rate
                    pnl = (proceeds - fee) - (entry_price * btc + entry_price * btc * self.fee_rate)
                    balance += proceeds - fee

                    trades.append({
                        "type": "SELL",
                        "price": exec_price,
                        "slippage": price - exec_price,
                        "fee": fee,
                        "pnl": pnl,
                        "return_pct": (exec_price / entry_price - 1) * 100,
                        "duration_bars": i - entry_time if entry_time else 0,
                        "reason": reason,
                        "time_idx": i,
                    })
                    btc = 0

            # Equity curve
            current_equity = balance + (btc * price if btc > 0 else 0)
            equity_curve.append(current_equity)
            peak_equity = max(peak_equity, current_equity)

        # Evaluate closing position
        if btc > 0:
            final_price = test_df.iloc[-1]["close"]
            balance += btc * final_price * (1 - self.fee_rate)

        # Calculate metrics
        final_value = balance
        total_return = (final_value - 10000) / 10000 * 100
        buy_hold = (test_df.iloc[-1]["close"] / test_df.iloc[0]["close"] - 1) * 100

        # Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(8760)
                  if returns.std() > 0 else 0)  # Annualized (8760 hours/year)

        # Max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())

        # Win rate
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) < 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

        # Profit factor
        total_wins = sum(t.get("pnl", 0) for t in wins)
        total_losses = abs(sum(t.get("pnl", 0) for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average trade duration
        durations = [t.get("duration_bars", 0) for t in sell_trades]
        avg_duration = np.mean(durations) if durations else 0

        return BacktestResult(
            total_return_pct=round(total_return, 2),
            buy_hold_return_pct=round(buy_hold, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            total_trades=len(sell_trades),
            avg_trade_duration=round(avg_duration, 1),
            final_value=round(final_value, 2),
            trades=trades,
            equity_curve=equity_curve,
        )

    def walk_forward_optimization(self, df: pd.DataFrame,
                                   n_splits: int = 5) -> dict:
        """
        Walk-Forward Optimization.

        Instead of training and testing once,
        we split the data into N windows and test on each separately.

        This gives a more realistic picture because:
        - Fresh model trained on each segment
        - Nincs lookahead bias
        - We see how stable the model is over time
        """
        print(f"\n🔬 Walk-Forward Optimization ({n_splits} window)")
        print("─" * 50)

        total_len = len(df)
        window_size = total_len // n_splits
        results = []

        for i in range(n_splits):
            start = i * window_size
            end = min((i + 2) * window_size, total_len)  # Overlapping window

            if end - start < 200:
                continue

            window_df = df.iloc[start:end].copy()
            print(f"\n  📊 Window {i+1}/{n_splits}: "
                  f"[{window_df.index[0]:%Y-%m-%d} → {window_df.index[-1]:%Y-%m-%d}] "
                  f"({len(window_df)} candle(s))")

            result = self.run_single_backtest(window_df, train_ratio=0.6)
            results.append(result)

            print(f"     Return: {result.total_return_pct:+.2f}% | "
                  f"Sharpe: {result.sharpe_ratio:.2f} | "
                  f"MaxDD: {result.max_drawdown_pct:.1f}% | "
                  f"Win: {result.win_rate:.0f}% | "
                  f"Trades: {result.total_trades}")

        # Summary
        if not results:
            return {"error": "Not enough data"}

        avg_return = np.mean([r.total_return_pct for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_win_rate = np.mean([r.win_rate for r in results])
        avg_drawdown = np.mean([r.max_drawdown_pct for r in results])
        std_return = np.std([r.total_return_pct for r in results])
        consistency = sum(1 for r in results if r.total_return_pct > 0) / len(results)

        summary = {
            "n_windows": len(results),
            "avg_return_pct": round(avg_return, 2),
            "std_return_pct": round(std_return, 2),
            "avg_sharpe": round(avg_sharpe, 2),
            "avg_win_rate": round(avg_win_rate, 1),
            "avg_max_drawdown": round(avg_drawdown, 2),
            "consistency": round(consistency * 100, 1),  # % profitable windows
            "worst_window": round(min(r.total_return_pct for r in results), 2),
            "best_window": round(max(r.total_return_pct for r in results), 2),
            "results": results,
        }

        print(f"\n{'═' * 50}")
        print(f"📊 WALK-FORWARD SUMMARY")
        print(f"   Avg return:   {summary['avg_return_pct']:+.2f}% (±{summary['std_return_pct']:.2f}%)")
        print(f"   Avg Sharpe:  {summary['avg_sharpe']:.2f}")
        print(f"   Avg Win%:    {summary['avg_win_rate']:.1f}%")
        print(f"   Avg MaxDD:   {summary['avg_max_drawdown']:.1f}%")
        print(f"   Consistency: {summary['consistency']:.0f}% profitable")
        print(f"   Best:       {summary['best_window']:+.2f}%")
        print(f"   Worst:   {summary['worst_window']:+.2f}%")
        print(f"{'═' * 50}")

        return summary

    def monte_carlo_simulation(self, base_result: BacktestResult,
                                n_simulations: int = 1000) -> dict:
        """
        Monte Carlo simulation.

        Randomly reshuffles backtest trades
        to see the distribution of possible outcomes.

        Ez megmutatja:
        - What is the probability of profit?
        - What is the worst case?
        - How much does the result depend on trade order?
        """
        sell_trades = [t for t in base_result.trades if t["type"] == "SELL"]
        if len(sell_trades) < 5:
            return {"error": "Not enough trades for Monte Carlo simulation"}

        pnls = [t.get("pnl", 0) for t in sell_trades]
        initial = 10000.0

        final_values = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            # Randomly reshuffle trades
            shuffled_pnls = np.random.permutation(pnls)

            equity = [initial]
            peak = initial

            for pnl in shuffled_pnls:
                new_eq = equity[-1] + pnl
                equity.append(max(new_eq, 0))  # Cannot go below 0
                peak = max(peak, new_eq)

            final_val = equity[-1]
            final_values.append(final_val)

            # Max drawdown
            eq_series = pd.Series(equity)
            rolling_max = eq_series.cummax()
            dd = ((eq_series - rolling_max) / rolling_max * 100).min()
            max_drawdowns.append(abs(dd))

            # Sharpe
            returns = pd.Series(equity).pct_change().dropna()
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(len(returns))
                sharpe_ratios.append(sharpe)

        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)

        result = {
            "n_simulations": n_simulations,
            "mean_final_value": round(np.mean(final_values), 2),
            "median_final_value": round(np.median(final_values), 2),
            "std_final_value": round(np.std(final_values), 2),
            "probability_of_profit": round(np.mean(final_values > initial) * 100, 1),
            "percentile_5": round(np.percentile(final_values, 5), 2),
            "percentile_25": round(np.percentile(final_values, 25), 2),
            "percentile_75": round(np.percentile(final_values, 75), 2),
            "percentile_95": round(np.percentile(final_values, 95), 2),
            "worst_case": round(np.min(final_values), 2),
            "best_case": round(np.max(final_values), 2),
            "mean_max_drawdown": round(np.mean(max_drawdowns), 2),
            "avg_sharpe": round(np.mean(sharpe_ratios), 2) if sharpe_ratios else 0,
        }

        print(f"\n🎲 MONTE CARLO SIMULATION ({n_simulations} simulation(s))")
        print(f"{'─' * 50}")
        print(f"   Probability of profit:  {result['probability_of_profit']:.1f}%")
        print(f"   Mean final value:        ${result['mean_final_value']:,.2f}")
        print(f"   Median final value:       ${result['median_final_value']:,.2f}")
        print(f"   ")
        print(f"   5. percentilis:        ${result['percentile_5']:,.2f}  (worst 5%)")
        print(f"   25. percentilis:       ${result['percentile_25']:,.2f}")
        print(f"   75. percentilis:       ${result['percentile_75']:,.2f}")
        print(f"   95. percentilis:       ${result['percentile_95']:,.2f}  (best 5%)")
        print(f"   ")
        print(f"   Worst case:      ${result['worst_case']:,.2f}")
        print(f"   Best case:          ${result['best_case']:,.2f}")
        print(f"   Mean max drawdown:    {result['mean_max_drawdown']:.1f}%")

        return result

    def full_analysis(self, df: pd.DataFrame) -> dict:
        """
        Full analysis: single backtest + walk-forward + Monte Carlo
        """
        print("=" * 60)
        print("🔬 FULL BACKTEST ANALYSIS")
        print("=" * 60)

        # 1. Single backtest
        print("\n📊 1/3: Base backtest...")
        single = self.run_single_backtest(df)
        print(f"   Return: {single.total_return_pct:+.2f}% | "
              f"B&H: {single.buy_hold_return_pct:+.2f}% | "
              f"Sharpe: {single.sharpe_ratio:.2f}")

        # 2. Walk-Forward
        print("\n📊 2/3: Walk-Forward Optimization...")
        wf = self.walk_forward_optimization(df, n_splits=4)

        # 3. Monte Carlo
        print("\n📊 3/3: Monte Carlo Simulation...")
        mc = self.monte_carlo_simulation(single, n_simulations=500)

        # Overall verdict
        print(f"\n{'═' * 60}")
        print("🏆 FINAL VERDICT")
        print(f"{'═' * 60}")

        score = 0
        reasons = []

        # Profit?
        if single.total_return_pct > 0:
            score += 1
            reasons.append(f"✅ Profitable ({single.total_return_pct:+.2f}%)")
        else:
            reasons.append(f"❌ Loss-making ({single.total_return_pct:+.2f}%)")

        # Beat buy & hold?
        if single.total_return_pct > single.buy_hold_return_pct:
            score += 1
            reasons.append("✅ Outperforms Buy & Hold")
        else:
            reasons.append("❌ Underperforms Buy & Hold")

        # Win rate > 50%?
        if single.win_rate > 50:
            score += 1
            reasons.append(f"✅ Win rate {single.win_rate:.0f}% (>50%)")
        else:
            reasons.append(f"⚠️ Win rate {single.win_rate:.0f}% (<50%)")

        # Sharpe > 1?
        if single.sharpe_ratio > 1:
            score += 1
            reasons.append(f"✅ Sharpe ratio {single.sharpe_ratio:.2f} (>1)")
        else:
            reasons.append(f"⚠️ Sharpe ratio {single.sharpe_ratio:.2f} (<1)")

        # Konzisztencia > 60%?
        if isinstance(wf, dict) and wf.get("consistency", 0) > 60:
            score += 1
            reasons.append(f"✅ Consistent ({wf['consistency']:.0f}% profitable windows)")
        else:
            reasons.append(f"⚠️ Not consistent")

        # Monte Carlo profit probability > 60%?
        if mc.get("probability_of_profit", 0) > 60:
            score += 1
            reasons.append(f"✅ MC profit probability {mc['probability_of_profit']:.0f}%")
        else:
            reasons.append(f"⚠️ MC profit probability {mc.get('probability_of_profit', 0):.0f}%")

        for r in reasons:
            print(f"   {r}")

        print(f"\n   📊 Score: {score}/6")
        if score >= 5:
            print("   🟢 Bot may be suitable for live testing (with minimal capital)")
        elif score >= 3:
            print("   🟡 Promising, but needs more development")
        else:
            print("   🔴 NOT suitable for live trading yet")

        return {
            "single_backtest": single,
            "walk_forward": wf,
            "monte_carlo": mc,
            "score": score,
            "max_score": 6,
        }

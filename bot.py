"""
Crypto Trading Bot v2 - Multi-Coin, Multi-Timeframe
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import time
import json
from datetime import datetime

import config
from data_fetcher import get_client
from multi_analyzer import MultiCoinScanner
from risk_manager import RiskManager


class TradingBotV2:
    """Multi-coin, multi-timeframe trading bot."""

    def __init__(self):
        self.client = get_client()
        self.scanner = MultiCoinScanner(self.client)
        self.risk = RiskManager()
        self.cycle_count = 0
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        print("=" * 60)
        print("CRYPTO TRADING BOT v2")
        print(f"   Coinok: {len(config.TRADING_PAIRS)} db")
        print(f"   Timeframe-ek: {', '.join(config.TIMEFRAMES.values())}")
        print(f"   Mode: {'📝 PAPER' if config.PAPER_TRADING else '💰 LIVE'}")
        print("=" * 60)

    def train(self):
        return self.scanner.train_all_models()

    def scan_and_trade(self):
        self.cycle_count += 1
        print(f"\n{'═' * 60}")
        print(f"Scan #{self.cycle_count} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'═' * 60}")

        opportunities = self.scanner.scan()
        filtered = self.scanner.filter_correlated(opportunities)
        self.scanner.print_scan_results(filtered)

        self._check_open_positions()

        tradeables = [o for o in filtered if o.get("tradeable")]
        for opp in tradeables:
            self._try_open_trade(opp)

        self._log_cycle(filtered)

    def _check_open_positions(self):
        for trade in list(self.risk.open_trades):
            try:
                price = self.client.get_current_price(trade.symbol)
                exits = self.risk.check_exits(price)
                for exit_trade in exits:
                    print(f"   📤 {exit_trade.symbol} closed: {exit_trade.status} | "
                          f"PnL: ${exit_trade.pnl:+.2f}")
                    close_side = "SELL" if exit_trade.side == "BUY" else "BUY"
                    self.client.place_order(exit_trade.symbol, close_side, exit_trade.quantity)
            except Exception as e:
                print(f"  Position check error ({trade.symbol}): {e}")

    def _try_open_trade(self, opp: dict):
        symbol = opp["symbol"]
        signal = opp["signal"]
        confidence = opp["confidence"]
        price = opp["price"]

        side = "BUY" if signal == 1 else "SELL"

        # On spot market, only SELL if there is an open long position
        if side == "SELL":
            coin_longs = [t for t in self.risk.open_trades
                          if t.symbol == symbol and t.side == "BUY"]
            if not coin_longs:
                return

        # Already has an open position on this coin?
        coin_trades = [t for t in self.risk.open_trades if t.symbol == symbol]
        if len(coin_trades) >= config.MAX_TRADES_PER_COIN:
            print(f"  {symbol}: already has open position, skip")
            return

        portfolio_value = self._get_portfolio_value()
        can_trade, reason = self.risk.can_open_trade(portfolio_value)
        if not can_trade:
            print(f"  {symbol}: {reason}")
            return

        # Exposure check
        current_exposure = sum(t.entry_price * t.quantity for t in self.risk.open_trades)
        max_exposure = portfolio_value * (config.MAX_PORTFOLIO_EXPOSURE_PCT / 100)
        if current_exposure >= max_exposure:
            print(f"  {symbol}: Max exposure reached")
            return

        # Position size
        overrides = config.PAIR_OVERRIDES.get(symbol, {})
        max_pct = overrides.get("max_position_pct", config.MAX_POSITION_SIZE_PCT)
        max_usd = portfolio_value * (max_pct / 100)
        scale = 0.5 + (confidence * 0.5)
        position_usd = max_usd * scale
        quantity = round(position_usd / price, 6)

        if quantity <= 0 or position_usd < 5:
            return

        try:
            self.client.place_order(symbol, side, quantity)
            trade = self.risk.register_trade(
                symbol, side, price, quantity, atr=opp.get("atr")
            )
            print(f"\n   {side} {symbol}")
            print(f"    Quantity: {quantity} @ ${price:,.2f} (${position_usd:,.2f})")
            print(f"    Confidence: {confidence:.1%} | Alignment: {opp.get('alignment', '?')}")
            print(f"    SL: ${trade.stop_loss:,.2f} | TP: ${trade.take_profit:,.2f} | TS: ${trade.trailing_stop:,.2f}")
        except Exception as e:
            print(f"   {symbol} order error: {e}")

    def _get_portfolio_value(self) -> float:
        if hasattr(self.client, "get_portfolio_value"):
            return self.client.get_portfolio_value()
        return 10000.0

    def _log_cycle(self, opportunities: list):
        log_entry = {
            "cycle": self.cycle_count,
            "timestamp": datetime.now().isoformat(),
            "opportunities": [
                {
                    "symbol": o["symbol"], "signal": o["signal"],
                    "confidence": o["confidence"], "alignment": o.get("alignment"),
                    "price": o["price"], "tradeable": o.get("tradeable"),
                }
                for o in opportunities
            ],
            "open_trades": len(self.risk.open_trades),
            "stats": self.risk.get_statistics(),
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

    def run(self, cycles: int = None):
        self.train()
        print(f"\nBot started! Scan: {config.SCAN_INTERVAL_SECONDS}s")

        count = 0
        try:
            while True:
                self.scan_and_trade()
                count += 1
                if cycles and count >= cycles:
                    break
                print(f"\n    Next scan: {config.SCAN_INTERVAL_SECONDS}s ...")
                time.sleep(config.SCAN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n\nBot stopped (Ctrl+C)")
        finally:
            stats = self.risk.get_statistics()
            print("\n" + "=" * 60)
            print("FINAL STATISTICS:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            print("=" * 60)


if __name__ == "__main__":
    bot = TradingBotV2()
    bot.run()

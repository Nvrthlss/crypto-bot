"""
Crypto Trading Bot - Risk management
Stop-loss, position sizing, drawdown protection
"""

import config
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Single trade data"""
    symbol: str
    side: str           # "BUY" or "SELL"
    entry_price: float
    quantity: float
    timestamp: str
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED


class RiskManager:
    """
    Risk manager - prevents excessive losses.

    Functions:
    - Position sizing (Kelly criterion based)
    - Stop-loss / Take-profit setup
    - Trailing stop
    - Max daily loss limit
    - Max open positions limit
    """

    def __init__(self):
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.daily_pnl: float = 0.0
        self.daily_pnl_reset_date: str = datetime.now().strftime("%Y-%m-%d")
        self.peak_portfolio_value: float = 0.0

    def can_open_trade(self, portfolio_value: float) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Returns:
            (allowed, reason)
        """
        # Daily PnL reset
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.daily_pnl_reset_date:
            self.daily_pnl = 0.0
            self.daily_pnl_reset_date = today

        # Max nyitott pozíciók
        if len(self.open_trades) >= config.MAX_OPEN_TRADES:
            return False, f"Max open positions reached ({config.MAX_OPEN_TRADES})"

        # Max napi veszteség
        daily_loss_pct = abs(self.daily_pnl) / portfolio_value * 100
        if self.daily_pnl < 0 and daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss limit reached ({daily_loss_pct:.1f}%)"

        return True, "OK"

    def calculate_position_size(self, portfolio_value: float,
                                current_price: float,
                                confidence: float) -> float:
        """
        Position sizing based on confidence.

        Higher confidence = larger position (de sosem több mint MAX_POSITION_SIZE_PCT)
        """
        # Base position size
        max_usd = portfolio_value * (config.MAX_POSITION_SIZE_PCT / 100)

        # Confidence-based scaling (0.5x - 1.0x)
        scale = 0.5 + (confidence * 0.5)
        position_usd = max_usd * scale

        # BTC quantity
        quantity = position_usd / current_price

        return round(quantity, 6)

    def set_stop_levels(self, entry_price: float, side: str,
                        atr: float = None) -> dict:
        """
        Calculate stop-loss and take-profit levels.

        Uses ATR for dynamic levels if available.
        """
        if atr and atr > 0:
            # ATR-based dynamic levels (2x ATR stop, 3x ATR profit)
            if side == "BUY":
                stop_loss = entry_price - (2.0 * atr)
                take_profit = entry_price + (3.0 * atr)
                trailing_stop = entry_price - (1.5 * atr)
            else:
                stop_loss = entry_price + (2.0 * atr)
                take_profit = entry_price - (3.0 * atr)
                trailing_stop = entry_price + (1.5 * atr)
        else:
            # Fixed percentage levels
            if side == "BUY":
                stop_loss = entry_price * (1 - config.STOP_LOSS_PCT / 100)
                take_profit = entry_price * (1 + config.TAKE_PROFIT_PCT / 100)
                trailing_stop = entry_price * (1 - config.TRAILING_STOP_PCT / 100)
            else:
                stop_loss = entry_price * (1 + config.STOP_LOSS_PCT / 100)
                take_profit = entry_price * (1 - config.TAKE_PROFIT_PCT / 100)
                trailing_stop = entry_price * (1 + config.TRAILING_STOP_PCT / 100)

        return {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "trailing_stop": round(trailing_stop, 2)
        }

    def register_trade(self, symbol: str, side: str, entry_price: float,
                       quantity: float, atr: float = None) -> Trade:
        """Register new trade with stop levels"""
        levels = self.set_stop_levels(entry_price, side, atr)

        trade = Trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            timestamp=datetime.now().isoformat(),
            stop_loss=levels["stop_loss"],
            take_profit=levels["take_profit"],
            trailing_stop=levels["trailing_stop"]
        )

        self.open_trades.append(trade)
        return trade

    def check_exits(self, current_price: float) -> list[Trade]:
        """
        Check if positions should be closed.

        Returns:
            List of trades to close
        """
        trades_to_close = []

        for trade in self.open_trades:
            should_close = False
            reason = ""

            if trade.side == "BUY":
                # Stop-loss
                if current_price <= trade.stop_loss:
                    should_close = True
                    reason = "STOP_LOSS"

                # Take-profit
                elif current_price >= trade.take_profit:
                    should_close = True
                    reason = "TAKE_PROFIT"

                # Trailing stop update
                new_trailing = current_price * (1 - config.TRAILING_STOP_PCT / 100)
                if new_trailing > trade.trailing_stop:
                    trade.trailing_stop = new_trailing

                # Trailing stop activation
                if current_price <= trade.trailing_stop and current_price > trade.entry_price:
                    should_close = True
                    reason = "TRAILING_STOP"

                if should_close:
                    trade.pnl = (current_price - trade.entry_price) * trade.quantity
                    trade.status = reason

            elif trade.side == "SELL":
                if current_price >= trade.stop_loss:
                    should_close = True
                    reason = "STOP_LOSS"
                elif current_price <= trade.take_profit:
                    should_close = True
                    reason = "TAKE_PROFIT"

                new_trailing = current_price * (1 + config.TRAILING_STOP_PCT / 100)
                if new_trailing < trade.trailing_stop:
                    trade.trailing_stop = new_trailing

                if current_price >= trade.trailing_stop and current_price < trade.entry_price:
                    should_close = True
                    reason = "TRAILING_STOP"

                if should_close:
                    trade.pnl = (trade.entry_price - current_price) * trade.quantity

            if should_close:
                trades_to_close.append(trade)

        # Move trades
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            self.daily_pnl += trade.pnl

        return trades_to_close

    def get_statistics(self) -> dict:
        """Trading statistics"""
        if not self.closed_trades:
            return {"total_trades": 0, "message": "No closed trades yet"}

        pnls = [t.pnl for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            "total_trades": len(pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
            "total_pnl": sum(pnls),
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses else float("inf"),
            "max_win": max(pnls) if pnls else 0,
            "max_loss": min(pnls) if pnls else 0,
            "open_trades": len(self.open_trades),
            "daily_pnl": self.daily_pnl
        }

"""
Crypto Trading Bot - Binance Data Fetcher
Historical and real-time OHLCV data + Paper Trading client
"""

import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import urlencode
import config


class BinanceClient:
    """Binance API client - paper and live trading support"""

    def __init__(self):
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.base_url = config.BINANCE_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign_request(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def get_klines(self, symbol: str = None, interval: str = None,
                   limit: int = None, start_time: int = None) -> pd.DataFrame:
        symbol = symbol or config.TRADING_PAIR
        interval = interval or config.TIMEFRAME
        limit = limit or config.LOOKBACK_PERIODS.get(interval, 500)

        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time

        url = "https://api.binance.com/api/v3/klines"
        response = self.session.get(url, params=params)
        response.raise_for_status()

        df = pd.DataFrame(response.json(), columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        for col in ["open", "high", "low", "close", "volume",
                     "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trades"] = df["trades"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df.drop(columns=["ignore"], inplace=True)
        df.set_index("timestamp", inplace=True)
        return df

    def get_current_price(self, symbol: str = None) -> float:
        symbol = symbol or config.TRADING_PAIR
        url = "https://api.binance.com/api/v3/ticker/price"
        response = self.session.get(url, params={"symbol": symbol})
        response.raise_for_status()
        return float(response.json()["price"])

    def get_account_balance(self) -> dict:
        params = self._sign_request({})
        url = f"{self.base_url}/api/v3/account"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        balances = {}
        for asset in response.json().get("balances", []):
            free = float(asset["free"])
            locked = float(asset["locked"])
            if free > 0 or locked > 0:
                balances[asset["asset"]] = {
                    "free": free, "locked": locked, "total": free + locked
                }
        return balances

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET") -> dict:
        params = {
            "symbol": symbol, "side": side, "type": order_type,
            "quantity": f"{quantity:.8f}",
        }
        params = self._sign_request(params)
        url = f"{self.base_url}/api/v3/order"
        response = self.session.post(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_order_book(self, symbol: str = None, limit: int = 10) -> dict:
        symbol = symbol or config.TRADING_PAIR
        url = "https://api.binance.com/api/v3/depth"
        response = self.session.get(url, params={"symbol": symbol, "limit": limit})
        response.raise_for_status()
        return response.json()


class PaperTradingClient:
    """Paper trading - simulated trading with real market data"""

    def __init__(self, initial_balance: float = 25.0):
        self.balance = {"USDT": initial_balance}
        self.orders = []
        self.trades = []
        self.binance = BinanceClient()

    def get_klines(self, **kwargs) -> pd.DataFrame:
        return self.binance.get_klines(**kwargs)

    def get_current_price(self, symbol: str = None) -> float:
        return self.binance.get_current_price(symbol)

    def get_account_balance(self) -> dict:
        return {
            asset: {"free": amount, "locked": 0.0, "total": amount}
            for asset, amount in self.balance.items() if amount > 0
        }

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET") -> dict:
        price = self.get_current_price(symbol)
        fee_rate = 0.001
        # Coin name (pl. BTCUSDT -> BTC)
        coin = symbol.replace("USDT", "")

        if side == "BUY":
            cost = quantity * price * (1 + fee_rate)
            if cost > self.balance.get("USDT", 0):
                raise ValueError(
                    f"Not enough USDT! Required: {cost:.2f}, "
                    f"Available: {self.balance.get('USDT', 0):.2f}"
                )
            self.balance["USDT"] -= cost
            self.balance[coin] = self.balance.get(coin, 0) + quantity

        elif side == "SELL":
            if quantity > self.balance.get(coin, 0):
                raise ValueError(
                    f"Not enough {coin}! Required: {quantity:.8f}, "
                    f"Available: {self.balance.get(coin, 0):.8f}"
                )
            self.balance[coin] -= quantity
            if self.balance[coin] < 1e-10:
                del self.balance[coin]
            self.balance["USDT"] = self.balance.get("USDT", 0) + quantity * price * (1 - fee_rate)

        trade = {
            "symbol": symbol, "side": side, "quantity": quantity,
            "price": price, "fee": quantity * price * fee_rate,
            "timestamp": datetime.now().isoformat(), "type": "PAPER"
        }
        self.trades.append(trade)

        return {
            "orderId": len(self.trades), "symbol": symbol, "side": side,
            "origQty": str(quantity), "executedQty": str(quantity),
            "price": str(price), "status": "FILLED", "type": "PAPER_TRADE"
        }

    def get_portfolio_value(self) -> float:
        total = self.balance.get("USDT", 0)
        for asset, amount in self.balance.items():
            if asset != "USDT" and amount > 0:
                try:
                    price = self.get_current_price(f"{asset}USDT")
                    total += amount * price
                except Exception:
                    pass
        return total


def get_client():
    if config.PAPER_TRADING:
        print("PAPER TRADING mode active")
        return PaperTradingClient()
    else:
        print("LIVE TRADING mode! Real money!")
        return BinanceClient()

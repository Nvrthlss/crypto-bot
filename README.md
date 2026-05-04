# 🤖 Crypto Trading Bot

An ML-powered cryptocurrency trading bot that combines **technical analysis**, **candlestick pattern recognition**, and **ensemble machine learning** across multiple coins and timeframes.

## Features

- **Multi-Coin Scanning** — Monitors 27 coins across 6 sectors (Layer 1, DeFi, Gaming, AI, Meme, Layer 2)
- **Multi-Timeframe Analysis** — 5-level analysis (5m → 15m → 1h → 4h → 1d) with weighted signal aggregation
- **Ensemble ML** — Random Forest + Gradient Boosting + Logistic Regression with majority voting
- **14 Candlestick Patterns** — Doji, Hammer, Engulfing, Morning/Evening Star, Three White Soldiers, and more
- **10+ Technical Indicators** — RSI, MACD, Bollinger Bands, ADX, Ichimoku Cloud, Stochastic, ATR, OBV, VWAP
- **Risk Management** — Stop-loss, take-profit, trailing stop, position sizing, daily loss limits, correlation filtering
- **Paper Trading** — Full simulation with real market data before going live
- **Advanced Backtesting** — Walk-forward optimization, Monte Carlo simulation, slippage modeling
- **Market Regime Detection** — Automatically switches between trend-following, mean-reversion, and breakout strategies
- **Sentiment Analysis** — Fear & Greed Index, funding rates, long/short ratios (contrarian signals)
- **LSTM Deep Learning** — Optional bidirectional LSTM with attention mechanism (requires PyTorch)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Trading Bot                       │
├──────────┬──────────┬───────────┬───────────────────┤
│  Data    │ Analysis │    ML     │   Execution       │
│ Fetcher  │ Engine   │  Models   │   Engine          │
├──────────┼──────────┼───────────┼───────────────────┤
│ Binance  │ 10+ Ind. │ RF + GB   │ Risk Manager      │
│ API      │ 14 Patt. │ + LogReg  │ Stop-Loss/TP      │
│ OHLCV    │ 5 TFs    │ Ensemble  │ Position Sizing   │
│ Real-time│ Regime   │ LSTM*     │ Paper/Live        │
└──────────┴──────────┴───────────┴───────────────────┘
```

## Project Structure

```
crypto-bot/
├── bot.py                # Main trading engine
├── config.py             # Configuration (API keys, parameters)
├── data_fetcher.py       # Binance API client + Paper Trading
├── indicators.py         # Technical indicators (RSI, MACD, BB, ADX, Ichimoku...)
├── patterns.py           # Candlestick pattern recognition (14 patterns)
├── ml_model.py           # Ensemble ML (Random Forest + GB + LogReg)
├── multi_analyzer.py     # Multi-coin, multi-timeframe scanner
├── risk_manager.py       # Risk management (SL, TP, trailing stop, sizing)
├── strategies.py         # Strategy engine (trend, mean-reversion, breakout)
├── sentiment.py          # Market sentiment analysis
├── deep_learning.py      # LSTM + Attention model (optional, requires PyTorch)
├── advanced_backtest.py  # Walk-forward optimization + Monte Carlo
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

# Optional: for LSTM model
pip install torch
```

### 2. Configure
Edit `config.py` with your Binance API credentials:
```python
BINANCE_API_KEY = "your_key"
BINANCE_API_SECRET = "your_secret"
```

### 3. Run (Paper Trading)
```bash
python bot.py
```

The bot will:
1. Train ML models for each coin × timeframe combination
2. Scan all coins every 2 minutes
3. Execute paper trades based on ML signals + multi-timeframe alignment
4. Log everything to `logs/`

### 4. Go Live (after paper trading validation)
In `config.py`:
```python
PAPER_TRADING = False
BINANCE_BASE_URL = "https://api.binance.com"
```

## How It Works

### Signal Generation
1. **Data Collection** — Downloads OHLCV data from Binance for each coin across 5 timeframes
2. **Feature Engineering** — Computes 30+ features from technical indicators, price action, and candlestick patterns
3. **ML Prediction** — Ensemble model votes BUY/SELL/HOLD with confidence score
4. **Multi-Timeframe Alignment** — Higher timeframes confirm or weaken the signal:
   - All timeframes agree → **+25% confidence** (STRONG_ALIGNMENT)
   - Majority agrees → **+10% confidence** (PARTIAL_ALIGNMENT)
   - Majority disagrees → **-30% confidence** (DIVERGENCE)
5. **Risk Check** — Position sizing, exposure limits, daily loss limits

### Market Regime Detection
The bot automatically detects the current market regime and adjusts strategy:
- **Strong Trend** (ADX > 30) → Trend Following (EMA crossovers, momentum)
- **Ranging Market** (low ADX) → Mean Reversion (RSI extremes, Bollinger Bands)
- **Squeeze** (narrow Bollinger Bands) → Breakout (volume spike + direction)

### Risk Management
- **Position Sizing** — Scales with confidence (higher confidence = larger position)
- **Stop-Loss** — Fixed % or ATR-based dynamic stops
- **Take-Profit** — Fixed % or ATR-based targets
- **Trailing Stop** — Locks in profit as price moves favorably
- **Daily Loss Limit** — Stops trading if daily loss exceeds threshold
- **Correlation Filter** — Prevents overexposure to correlated coins

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRADING_PAIRS` | 27 coins | Coins to monitor |
| `TIMEFRAMES` | 5m→1d | Analysis timeframes |
| `ML_CONFIDENCE_THRESHOLD` | 0.50 | Min confidence to trade |
| `STOP_LOSS_PCT` | 2.0% | Stop-loss percentage |
| `TAKE_PROFIT_PCT` | 4.0% | Take-profit percentage |
| `MAX_OPEN_TRADES` | 2 | Max simultaneous positions |
| `SCAN_INTERVAL_SECONDS` | 120 | Scan frequency |

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — Ensemble ML models
- **pandas / numpy** — Data processing
- **Binance API** — Market data & order execution
- **PyTorch** (optional) — LSTM deep learning model

## Disclaimer

⚠️ This bot is for **educational purposes**. Cryptocurrency trading involves significant risk. Past performance does not guarantee future results. Always test thoroughly with paper trading before using real funds, and never invest more than you can afford to lose.

## License

MIT

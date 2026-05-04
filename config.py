"""
Crypto Trading Bot - Configuration
Multi-coin, Multi-timeframe (5 szint)
"""

# ============================================================
# BINANCE API
# ============================================================
BINANCE_API_KEY = "YOUR_API_KEY_HERE"
BINANCE_API_SECRET = "YOUR_API_SECRET_HERE"

BINANCE_BASE_URL = "https://testnet.binance.vision"
BINANCE_WS_URL = "wss://testnet.binance.vision/ws"

# Live URLs (ONLY after bot has proven itself in paper trading!)
# BINANCE_BASE_URL = "https://api.binance.com"
# BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# ============================================================
# TRADING PAIRS
# ============================================================
TRADING_PAIRS = [
    # Layer 1
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    # DeFi
    "UNIUSDT", "AAVEUSDT", "MKRUSDT",
    # Gaming / Metaverse
    "AXSUSDT", "MANAUSDT", "SANDUSDT",
    # AI
    "FETUSDT", "RENDERUSDT",
    # Infrastructure
    "LINKUSDT", "DOTUSDT",
    # Meme
    "PEPEUSDT", "SHIBUSDT", "DOGEUSDT", "WIFUSDT",
    # Layer 2
    "MATICUSDT", "ARBUSDT", "OPUSDT",
    # Trending
    "SUIUSDT", "NEARUSDT", "INJUSDT",
]

TRADING_PAIR = TRADING_PAIRS[0]

PAIR_OVERRIDES = {
    "BTCUSDT":  {"max_position_pct": 40.0, "min_confidence": 0.50},
    "ETHUSDT":  {"max_position_pct": 40.0, "min_confidence": 0.50},
    "DOGEUSDT": {"max_position_pct": 30.0, "min_confidence": 0.55},
    "SOLUSDT":  {"max_position_pct": 35.0, "min_confidence": 0.50},
}

# ============================================================
# MULTI-TIMEFRAME (5 szint)
# ============================================================
TIMEFRAMES = {
    "scalp":     "5m",
    "primary":   "15m",
    "secondary": "1h",
    "tertiary":  "4h",
    "macro":     "1d",
}

TIMEFRAME_WEIGHTS = {
    "scalp":     0.15,
    "primary":   0.30,
    "secondary": 0.25,
    "tertiary":  0.20,
    "macro":     0.10,
}

LOOKBACK_PERIODS = {
    "5m":  500,
    "15m": 500,
    "1h":  500,
    "4h":  200,
    "1d":  120,
}

TIMEFRAME = TIMEFRAMES["primary"]

# ============================================================
# RISK MANAGEMENT
# ============================================================
MAX_POSITION_SIZE_PCT = 40.0
STOP_LOSS_PCT = 2.0
TAKE_PROFIT_PCT = 4.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_OPEN_TRADES = 2
MAX_TRADES_PER_COIN = 1
TRAILING_STOP_PCT = 1.5
MAX_PORTFOLIO_EXPOSURE_PCT = 80

# ============================================================
# ML PARAMETERS
# ============================================================
ML_CONFIDENCE_THRESHOLD = 0.50
TRAIN_TEST_SPLIT = 0.8
RETRAIN_INTERVAL_HOURS = 24

# ============================================================
# CORRELATION FILTER
# ============================================================
MAX_CORRELATION = 0.85

# ============================================================
# MODE
# ============================================================
PAPER_TRADING = True

# ============================================================
# SCANNING
# ============================================================
SCAN_INTERVAL_SECONDS = 120

"""
Crypto Trading Bot - Machine Learning Ensemble
Random Forest + Gradient Boosting + Logistic Regression
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime


class TradingMLModel:
    """
    Ensemble ML modell: három modell szavazással dönt.
    Target: 1 = BUY, 0 = HOLD, -1 = SELL
    """

    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=20, random_state=42, n_jobs=1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5,
            learning_rate=0.05, min_samples_leaf=20, random_state=42
        )
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)

        self.scaler = StandardScaler()
        self.is_trained = False
        self._hold_only = False
        self.feature_columns = []
        self.last_train_time = None
        self.train_metrics = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        if "rsi" in df.columns:
            features["rsi"] = df["rsi"]
            features["rsi_oversold"] = (df["rsi"] < 30).astype(int)
            features["rsi_overbought"] = (df["rsi"] > 70).astype(int)
            features["rsi_momentum"] = df["rsi"] - df["rsi"].shift(3)

        if "macd_line" in df.columns:
            features["macd_line"] = df["macd_line"]
            features["macd_histogram"] = df["macd_histogram"]
            features["macd_crossover"] = (
                (df["macd_line"] > df["macd_signal"]) &
                (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
            ).astype(int)
            features["macd_crossunder"] = (
                (df["macd_line"] < df["macd_signal"]) &
                (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
            ).astype(int)

        if "bb_pct" in df.columns:
            features["bb_pct"] = df["bb_pct"]
            features["bb_width"] = df["bb_width"]
            features["bb_squeeze"] = (
                df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)
            ).astype(int)

        if "ema_9" in df.columns and "ema_21" in df.columns:
            features["ema_9_21_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)
            features["ema_distance"] = (df["ema_9"] - df["ema_21"]) / df["close"]

        if "ema_50" in df.columns and "ema_200" in df.columns:
            features["golden_cross"] = (df["ema_50"] > df["ema_200"]).astype(int)
            features["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        if "adx" in df.columns:
            features["adx"] = df["adx"]
            features["strong_trend"] = (df["adx"] > 25).astype(int)
            features["trend_direction"] = np.where(
                df.get("plus_di", 0) > df.get("minus_di", 0), 1, -1
            )

        if "stoch_k" in df.columns:
            features["stoch_k"] = df["stoch_k"]
            features["stoch_crossover"] = (
                (df["stoch_k"] > df["stoch_d"]) &
                (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
            ).astype(int)

        if "atr_pct" in df.columns:
            features["atr_pct"] = df["atr_pct"]
            features["volatility_high"] = (
                df["atr_pct"] > df["atr_pct"].rolling(50).quantile(0.8)
            ).astype(int)

        if "volume" in df.columns:
            features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
            features["volume_spike"] = (features["volume_ratio"] > 2.0).astype(int)

        if "obv" in df.columns and "obv_ema" in df.columns:
            features["obv_trend"] = (df["obv"] > df["obv_ema"]).astype(int)

        features["return_1"] = df["close"].pct_change(1)
        features["return_3"] = df["close"].pct_change(3)
        features["return_5"] = df["close"].pct_change(5)
        features["return_10"] = df["close"].pct_change(10)

        features["higher_highs"] = (
            (df["high"] > df["high"].shift(1)) &
            (df["high"].shift(1) > df["high"].shift(2))
        ).astype(int)
        features["lower_lows"] = (
            (df["low"] < df["low"].shift(1)) &
            (df["low"].shift(1) < df["low"].shift(2))
        ).astype(int)

        for col in [c for c in df.columns if c.startswith("pat_")]:
            features[col] = df[col]

        if "ichimoku_tenkan" in df.columns:
            features["ichi_tk_cross"] = (
                df["ichimoku_tenkan"] > df["ichimoku_kijun"]
            ).astype(int)
            features["price_above_cloud"] = (
                df["close"] > df[["ichimoku_senkou_a", "ichimoku_senkou_b"]].max(axis=1)
            ).astype(int)

        return features

    def create_labels(self, df: pd.DataFrame, forward_periods: int = 5,
                      threshold_pct: float = 1.0) -> pd.Series:
        future_return = df["close"].shift(-forward_periods) / df["close"] - 1
        future_return_pct = future_return * 100
        labels = pd.Series(0, index=df.index)
        labels[future_return_pct > threshold_pct] = 1
        labels[future_return_pct < -threshold_pct] = -1
        return labels

    def train(self, df: pd.DataFrame, forward_periods: int = 5,
              threshold_pct: float = None) -> dict:
        features = self.prepare_features(df)

        # Auto-threshold from volatility
        if threshold_pct is None:
            avg_move = df["close"].pct_change(forward_periods).abs().median() * 100
            threshold_pct = max(0.1, avg_move * 0.5)

        # Reduce threshold if not enough classes
        for attempt in [threshold_pct, threshold_pct * 0.7,
                        threshold_pct * 0.4, threshold_pct * 0.2]:
            labels = self.create_labels(df, forward_periods, attempt)
            valid_mask = features.notna().all(axis=1) & labels.notna()
            if len(labels[valid_mask].unique()) >= 2:
                threshold_pct = attempt
                break

        labels = self.create_labels(df, forward_periods, threshold_pct)
        valid_mask = features.notna().all(axis=1) & labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]

        if len(features) < 100:
            raise ValueError(f"Not enough data: {len(features)} sor")

        if len(labels.unique()) < 2:
            print(f"  ⚠️ Only 1 class — HOLD model")
            self.is_trained = True
            self._hold_only = True
            self.feature_columns = features.columns.tolist()
            self.train_metrics = {
                "ensemble_accuracy": 0.0, "rf_accuracy": 0.0,
                "gb_accuracy": 0.0, "lr_accuracy": 0.0,
                "train_samples": len(features), "test_samples": 0,
                "label_distribution": {"buy": 0, "hold": len(features), "sell": 0},
                "top_features": {}
            }
            return self.train_metrics

        self._hold_only = False
        self.feature_columns = features.columns.tolist()

        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.rf_model.fit(X_train_scaled, y_train)
        self.gb_model.fit(X_train_scaled, y_train)
        self.lr_model.fit(X_train_scaled, y_train)

        rf_pred = self.rf_model.predict(X_test_scaled)
        gb_pred = self.gb_model.predict(X_test_scaled)
        lr_pred = self.lr_model.predict(X_test_scaled)
        ensemble_pred = np.sign(rf_pred + gb_pred + lr_pred)

        self.train_metrics = {
            "rf_accuracy": accuracy_score(y_test, rf_pred),
            "gb_accuracy": accuracy_score(y_test, gb_pred),
            "lr_accuracy": accuracy_score(y_test, lr_pred),
            "ensemble_accuracy": accuracy_score(y_test, ensemble_pred),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "label_distribution": {
                "buy": int((y_train == 1).sum()),
                "hold": int((y_train == 0).sum()),
                "sell": int((y_train == -1).sum())
            }
        }

        importance = pd.Series(
            self.rf_model.feature_importances_, index=self.feature_columns
        ).sort_values(ascending=False)
        self.train_metrics["top_features"] = importance.head(10).to_dict()

        self.is_trained = True
        self.last_train_time = datetime.now()

        print(f"    ✅ Ensemble: {self.train_metrics['ensemble_accuracy']:.1%} "
              f"(RF:{self.train_metrics['rf_accuracy']:.1%} "
              f"GB:{self.train_metrics['gb_accuracy']:.1%} "
              f"LR:{self.train_metrics['lr_accuracy']:.1%})")

        return self.train_metrics

    def predict(self, df: pd.DataFrame) -> tuple:
        if not self.is_trained:
            raise ValueError("Model is not trained!")

        if self._hold_only:
            return 0, 0.0

        features = self.prepare_features(df)
        latest = features.iloc[[-1]]

        if latest.isna().any(axis=1).iloc[0]:
            return 0, 0.0

        latest_scaled = self.scaler.transform(latest[self.feature_columns])

        rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
        gb_proba = self.gb_model.predict_proba(latest_scaled)[0]
        lr_proba = self.lr_model.predict_proba(latest_scaled)[0]

        classes = self.rf_model.classes_
        avg_proba = (rf_proba + gb_proba + lr_proba) / 3

        predicted_class = classes[np.argmax(avg_proba)]
        confidence = np.max(avg_proba)

        return int(predicted_class), float(confidence)

    def save(self, path: str = "model"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.rf_model, f"{path}/rf_model.pkl")
        joblib.dump(self.gb_model, f"{path}/gb_model.pkl")
        joblib.dump(self.lr_model, f"{path}/lr_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.feature_columns, f"{path}/features.pkl")

    def load(self, path: str = "model"):
        self.rf_model = joblib.load(f"{path}/rf_model.pkl")
        self.gb_model = joblib.load(f"{path}/gb_model.pkl")
        self.lr_model = joblib.load(f"{path}/lr_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.feature_columns = joblib.load(f"{path}/features.pkl")
        self.is_trained = True
        self._hold_only = False

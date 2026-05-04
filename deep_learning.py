"""
Crypto Trading Bot - Deep Learning modul (LSTM + Attention)
Advanced time series prediction alongside traditional ML models.

Why is LSTM better than Random Forest for time series?
- Remembers long-term patterns (memory cells)
- Naturally handles sequential data
- Attention highlights important time steps

Note: PyTorch-based implementation.
Runs on CPU too if no GPU available, just trains slower.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# PyTorch imports - graceful fallback if not installed
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. LSTM model not available.")
    print("   Install: pip install torch")


class TimeSeriesDataset:
    """PyTorch Dataset for time series data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.X = []
        self.y = []

        for i in range(sequence_length, len(features)):
            self.X.append(features[i - sequence_length:i])
            self.y.append(labels[i])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

        # Label mapping: {-1, 0, 1} -> {0, 1, 2} (PyTorch CE loss)
        self.y = self.y + 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if TORCH_AVAILABLE:
            return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]
        return self.X[idx], self.y[idx]


if TORCH_AVAILABLE:
    class AttentionLayer(nn.Module):
        """
        Attention mechanizmus az LSTM-hez.
        Learns which parts of the sequence are most important.
        """

        def __init__(self, hidden_size: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

        def forward(self, lstm_output):
            # lstm_output: (batch, seq_len, hidden_size)
            attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            context = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_size)
            return context, attention_weights

    class LSTMModel(nn.Module):
        """
        LSTM + Attention model for trading signals.

        Architecture:
        Input -> LSTM (2 layers, bidirectional) -> Attention -> FC -> Softmax
        """

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.3,
                     num_classes: int = 3):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )

            self.attention = AttentionLayer(hidden_size * 2)  # *2 bidirectional

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            context, attention_weights = self.attention(lstm_out)
            output = self.classifier(context)
            return output, attention_weights


class DeepLearningModel:
    """
    LSTM + Attention wrapper for the trading bot.
    Same interface as TradingMLModel.
    """

    def __init__(self, sequence_length: int = 60, hidden_size: int = 128,
                 num_layers: int = 2, learning_rate: float = 0.001,
                 epochs: int = 50, batch_size: int = 32):

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed!")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.feature_columns = []
        self.train_metrics = {}

        print(f"   🖥️ Device: {self.device}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering - same as TradingMLModel"""
        features = pd.DataFrame(index=df.index)

        # Technical indicators
        for col in ["rsi", "macd_line", "macd_histogram", "bb_pct", "bb_width",
                     "adx", "stoch_k", "atr_pct"]:
            if col in df.columns:
                features[col] = df[col]

        # EMA features
        if "ema_9" in df.columns and "ema_21" in df.columns:
            features["ema_9_21_ratio"] = df["ema_9"] / df["ema_21"]
        if "ema_50" in df.columns and "ema_200" in df.columns:
            features["ema_50_200_ratio"] = df["ema_50"] / df["ema_200"]
            features["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        # Price features
        features["return_1"] = df["close"].pct_change(1)
        features["return_3"] = df["close"].pct_change(3)
        features["return_5"] = df["close"].pct_change(5)
        features["return_10"] = df["close"].pct_change(10)
        features["volatility_5"] = df["close"].pct_change().rolling(5).std()
        features["volatility_20"] = df["close"].pct_change().rolling(20).std()

        # Volume
        if "volume" in df.columns:
            features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
            features["volume_change"] = df["volume"].pct_change()

        # OBV
        if "obv" in df.columns and "obv_ema" in df.columns:
            features["obv_ratio"] = df["obv"] / df["obv_ema"]

        # Candlestick patternek
        pattern_cols = [c for c in df.columns if c.startswith("pat_")]
        for col in pattern_cols:
            features[col] = df[col]

        return features

    def create_labels(self, df: pd.DataFrame, forward_periods: int = 5,
                      threshold_pct: float = 1.0) -> pd.Series:
        """Target: 1=BUY, 0=HOLD, -1=SELL"""
        future_return = df["close"].shift(-forward_periods) / df["close"] - 1
        future_return_pct = future_return * 100

        labels = pd.Series(0, index=df.index)
        labels[future_return_pct > threshold_pct] = 1
        labels[future_return_pct < -threshold_pct] = -1
        return labels

    def train(self, df: pd.DataFrame, forward_periods: int = 5,
              threshold_pct: float = 1.0) -> dict:
        """Train LSTM model"""
        features = self.prepare_features(df)
        labels = self.create_labels(df, forward_periods, threshold_pct)

        # Remove NaN
        valid = features.notna().all(axis=1) & labels.notna()
        features = features[valid]
        labels = labels[valid]

        self.feature_columns = features.columns.tolist()
        n_features = len(self.feature_columns)

        # Train/test split (time series!)
        split = int(len(features) * 0.8)
        X_train_raw = features.iloc[:split].values
        X_test_raw = features.iloc[split:].values
        y_train_raw = labels.iloc[:split].values
        y_test_raw = labels.iloc[split:].values

        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # Create dataset
        train_ds = TimeSeriesDataset(X_train_scaled, y_train_raw, self.sequence_length)
        test_ds = TimeSeriesDataset(X_test_scaled, y_test_raw, self.sequence_length)

        if len(train_ds) < 50:
            raise ValueError(f"Not enough sequences for training: {len(train_ds)}")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        # Create model
        self.model = LSTMModel(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Class weights (handle imbalanced data)
        class_counts = np.bincount(train_ds.y, minlength=3)
        if class_counts.min() > 0:
            class_weights = 1.0 / class_counts.astype(float)
            class_weights = class_weights / class_weights.sum() * 3
            weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        else:
            weights_tensor = torch.ones(3).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        print(f"   🧠 LSTM training ({self.epochs} epoch, {len(train_ds)} sequence(s))...")

        best_test_acc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output, _ = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            train_acc = correct / total

            # Teszt
            self.model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    output, _ = self.model(X_batch)
                    _, predicted = torch.max(output, 1)
                    test_total += y_batch.size(0)
                    test_correct += (predicted == y_batch).sum().item()

            test_acc = test_correct / test_total if test_total > 0 else 0

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}/{self.epochs}: "
                      f"loss={train_loss/len(train_loader):.4f} "
                      f"train_acc={train_acc:.1%} test_acc={test_acc:.1%}")

            # Early stopping
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"      ⏹️ Early stopping (epoch {epoch+1})")
                    break

        # Reload best model
        if best_test_acc > 0:
            self.model.load_state_dict(best_state)

        self.is_trained = True
        self.train_metrics = {
            "lstm_test_accuracy": best_test_acc,
            "train_sequences": len(train_ds),
            "test_sequences": len(test_ds),
            "n_features": n_features,
            "sequence_length": self.sequence_length,
        }

        print(f"   ✅ LSTM best test accuracy: {best_test_acc:.1%}")
        return self.train_metrics

    def predict(self, df: pd.DataFrame) -> tuple:
        """
        Prediction: signal + confidence

        Returns:
            (signal, confidence): signal ∈ {-1, 0, 1}, confidence ∈ [0, 1]
        """
        if not self.is_trained or self.model is None:
            return 0, 0.0

        features = self.prepare_features(df)

        if len(features) < self.sequence_length:
            return 0, 0.0

        # Utolsó sequence(s)
        latest = features.iloc[-self.sequence_length:]

        if latest.isna().any().any():
            return 0, 0.0

        scaled = self.scaler.transform(latest[self.feature_columns].values)
        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output, attention_weights = self.model(X)
            probas = torch.softmax(output, dim=1).cpu().numpy()[0]

        # Classes: 0=SELL, 1=HOLD, 2=BUY (mert labels+1 volt)
        predicted_class = int(np.argmax(probas))
        confidence = float(probas[predicted_class])

        # Convert back: {0,1,2} -> {-1,0,1}
        signal = predicted_class - 1

        return signal, confidence

    def get_attention_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get attention weights - melyik időpillanatok
        voltak a legfontosabbak a döntéshez.
        """
        if not self.is_trained or self.model is None:
            return np.array([])

        features = self.prepare_features(df)
        latest = features.iloc[-self.sequence_length:]

        if latest.isna().any().any():
            return np.array([])

        scaled = self.scaler.transform(latest[self.feature_columns].values)
        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(X)

        return attention_weights.cpu().numpy()[0].flatten()

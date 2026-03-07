"""
Training script for the LSTM price predictor.

Usage
-----
    python training/train_price_model.py \
        --data-dir data/ohlcv \
        --epochs 50 \
        --batch-size 64 \
        --seq-len 30 \
        --checkpoint checkpoints/price_predictor.pt

Data format
-----------
CSV files in *data-dir*, one per token pair, with columns:
  timestamp, open, high, low, close, volume, gas_price_gwei,
  block_time_ms, pending_tx_count

Labels are generated automatically:
  next_close > close * (1 + THRESHOLD) → "up"   (2)
  next_close < close * (1 - THRESHOLD) → "down" (0)
  otherwise                              → "flat" (1)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_price_model")

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRICE_THRESHOLD = 0.002  # 0.2 % move to count as directional


def load_csv_dataset(data_dir: str, seq_len: int, input_size: int):
    """Load all CSV files from *data_dir* and return (X, y) arrays."""
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "gas_price_gwei", "block_time_ms", "pending_tx_count",
    ]

    all_X, all_y = [], []
    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s — using synthetic data for demo", data_dir)
        return _synthetic_dataset(seq_len, input_size, n_samples=2000)

    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed — using synthetic data")
        return _synthetic_dataset(seq_len, input_size, n_samples=2000)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        # Fill missing feature columns with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        closes = df["close"].values
        features = df[feature_cols].values.astype(np.float32)

        # Normalise each feature column to zero-mean unit-variance
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        # Pad to input_size
        if features.shape[1] < input_size:
            pad = np.zeros((features.shape[0], input_size - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)

        for i in range(len(features) - seq_len - 1):
            seq = features[i: i + seq_len]
            next_close = closes[i + seq_len]
            curr_close = closes[i + seq_len - 1]
            if curr_close == 0:
                continue
            change = (next_close - curr_close) / curr_close
            label = 1  # flat
            if change > PRICE_THRESHOLD:
                label = 2  # up
            elif change < -PRICE_THRESHOLD:
                label = 0  # down
            all_X.append(seq)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    return X, y


def _synthetic_dataset(seq_len: int, input_size: int, n_samples: int = 2000):
    """Generate synthetic random-walk data for quick demo training."""
    np.random.seed(42)
    X = np.random.randn(n_samples, seq_len, input_size).astype(np.float32)
    y = np.random.randint(0, 3, size=n_samples).astype(np.int64)
    return X, y


def train(args):
    X, y = load_csv_dataset(args.data_dir, args.seq_len, args.input_size)
    logger.info("Dataset: X=%s y=%s", X.shape, y.shape)

    from models.price_predictor import PricePredictor

    predictor = PricePredictor(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        device=args.device,
    )

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Mini-batch training
        perm = np.random.permutation(len(X_train))
        losses = []
        for start in range(0, len(X_train), args.batch_size):
            idx_batch = perm[start: start + args.batch_size]
            loss = predictor.train_step(X_train[idx_batch], y_train[idx_batch], lr=args.lr)
            losses.append(loss)

        train_loss = float(np.mean(losses))

        # Validation
        val_losses = []
        for start in range(0, len(X_val), args.batch_size):
            val_loss_batch = predictor.validate(
                X_val[start: start + args.batch_size],
                y_val[start: start + args.batch_size],
            )
            val_losses.append(val_loss_batch)
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            predictor.save(args.checkpoint)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ohlcv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="checkpoints/price_predictor.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)

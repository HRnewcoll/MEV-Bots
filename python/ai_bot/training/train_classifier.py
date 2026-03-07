"""
Training script for the MEV Opportunity Classifier.

Usage
-----
    python training/train_classifier.py \
        --data-file data/mempool_labels.csv \
        --epochs 30 \
        --checkpoint checkpoints/opp_classifier.pkl

Data format
-----------
CSV with columns:
  value_eth, gas_price_gwei, input_length, selector_is_swap,
  selector_is_approve, nonce_norm, gas_limit_norm, mempool_age_ms_norm,
  [... 24 more feature columns ...],
  label   (0=no_mev, 1=sandwich, 2=arb, 3=liquidation)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_classifier")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_dataset(data_file: str, feature_dim: int):
    """Load labelled mempool data from a CSV file."""
    try:
        import pandas as pd
        df = pd.read_csv(data_file)
        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        # Pad or trim to feature_dim
        if X.shape[1] < feature_dim:
            pad = np.zeros((len(X), feature_dim - X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)
        else:
            X = X[:, :feature_dim]
        return X, y
    except (ImportError, FileNotFoundError):
        logger.warning("Using synthetic dataset for demo")
        np.random.seed(0)
        X = np.random.randn(1000, feature_dim).astype(np.float32)
        y = np.random.randint(0, 4, size=1000).astype(np.int64)
        return X, y


def train(args):
    X, y = load_dataset(args.data_file, args.feature_dim)
    logger.info("Dataset: X=%s y=%s", X.shape, y.shape)

    from models.opportunity_classifier import OpportunityClassifier

    clf = OpportunityClassifier(feature_dim=args.feature_dim, device=args.device)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))

    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(split)
        losses = []
        for start in range(0, split, args.batch_size):
            batch_idx = perm[start: start + args.batch_size]
            loss = clf.train_batch(X[batch_idx], y[batch_idx], lr=args.lr)
            losses.append(loss)
        logger.info("Epoch %d/%d | loss=%.4f", epoch, args.epochs, float(np.mean(losses)))

    clf.save(args.checkpoint)
    logger.info("Classifier saved to %s", args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="data/mempool_labels.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="checkpoints/opp_classifier.pkl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)

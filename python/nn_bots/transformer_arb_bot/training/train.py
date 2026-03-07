"""
Supervised training loop for the Transformer Arbitrage Model.

Generates synthetic training data from randomised reserve snapshots, trains
the model to predict which (buy_dex, sell_dex) pairs are profitable, and
saves the best checkpoint.

Usage
-----
    python training/train.py [--epochs 50] [--lr 1e-4] [--batch 32]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import TransformerArbModel, _uniswap_v2_out

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_transformer")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _generate_batch(
    batch_size: int,
    num_dexs: int,
    trade_amount_eth: float = 0.1,
) -> tuple:
    """
    Generate a random batch of (feature_matrices, labels).

    A sample is "profitable" if performing the two-leg swap across the pair
    returns more than the gas cost.
    """
    trade_wei = int(trade_amount_eth * 1e18)
    gas_cost_eth = 0.003  # ≈ 350k gas × 10 Gwei

    feature_matrices = []
    label_matrices   = []

    for _ in range(batch_size):
        # Random reserves for each DEX (large numbers, ~10-500 ETH per side)
        reserves_in  = np.random.randint(int(10e18), int(500e18), size=num_dexs)
        reserves_out = np.random.randint(int(10e18), int(500e18), size=num_dexs)
        fees = np.random.choice([10, 25, 30, 100], size=num_dexs)
        gas_gwei = np.random.uniform(5, 80)

        snapshots = [
            {
                "reserve_in":  int(reserves_in[i]),
                "reserve_out": int(reserves_out[i]),
                "fee_bps":     int(fees[i]),
            }
            for i in range(num_dexs)
        ]

        matrix = TransformerArbModel.build_feature_matrix(snapshots, gas_price_gwei=gas_gwei,
                                                           trade_amount_wei=trade_wei)
        labels = np.zeros((num_dexs, num_dexs), dtype=np.float32)

        for i in range(num_dexs):
            for j in range(num_dexs):
                if i == j:
                    continue
                out_buy  = _uniswap_v2_out(trade_wei, int(reserves_in[i]), int(reserves_out[i]))
                out_sell = _uniswap_v2_out(out_buy,   int(reserves_out[j]), int(reserves_in[j]))
                profit = (out_sell - trade_wei) / 1e18
                labels[i, j] = 1.0 if profit > gas_cost_eth else 0.0

        feature_matrices.append(matrix)
        label_matrices.append(labels)

    return np.array(feature_matrices), np.array(label_matrices)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 50,
    batch_size: int = 32,
    num_dexs: int = 4,
    lr: float = 1e-4,
    checkpoint: str = "checkpoints/transformer_arb.pt",
) -> None:
    model = TransformerArbModel()

    best_val_loss = float("inf")
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training
        x_train, y_train = _generate_batch(batch_size * 4, num_dexs)
        train_loss = model.train_step(x_train, y_train, lr=lr)

        # Validation
        x_val, y_val = _generate_batch(batch_size, num_dexs)
        val_loss, val_acc = model.validate(x_val, y_val)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
            epoch, epochs, train_loss, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(checkpoint)
            logger.info("  ✓ New best model saved (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete — best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TransformerArbModel")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--num-dexs",  type=int,   default=4)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="checkpoints/transformer_arb.pt")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        num_dexs=args.num_dexs,
        lr=args.lr,
        checkpoint=args.checkpoint,
    )

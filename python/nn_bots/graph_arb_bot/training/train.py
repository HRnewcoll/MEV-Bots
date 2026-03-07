"""
Supervised training loop for the GNN Arbitrage Model.

Generates synthetic token graphs with random reserve configurations,
labels each directed edge as "part of a profitable cycle" or not,
and trains the GNN to classify edges correctly.

Usage
-----
    python training/train.py [--epochs 50] [--lr 1e-4] [--num-tokens 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import GNNArbModel, TokenGraph, _v2_out_with_fee

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gnn")

# Realistic EVM token addresses (used only as keys)
_MOCK_TOKENS = [
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
    "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
    "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
    "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI
    "0x514910771AF9Ca656af840dff83E8264EcF986CA",  # LINK
]
_DEX_NAMES = ["uniswap_v2", "sushiswap", "curve"]
_FEES = [30, 30, 4]


def _random_graph(num_tokens: int = 4, num_dexs: int = 2) -> TokenGraph:
    """Build a random TokenGraph for training."""
    tokens = _MOCK_TOKENS[:num_tokens]
    graph = TokenGraph()

    pool_count = 0
    total_pools = num_dexs * (num_tokens * (num_tokens - 1) // 2)

    for dex_i in range(num_dexs):
        fee = _FEES[dex_i % len(_FEES)]
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                # Random reserves 10–500 ETH equivalent
                r_a = np.random.randint(int(10e18), int(500e18))
                r_b = np.random.randint(int(10e18), int(500e18))
                graph.add_pool(tokens[i], tokens[j], r_a, r_b, fee,
                               _DEX_NAMES[dex_i % len(_DEX_NAMES)], pool_count, total_pools)
                pool_count += 1

    return graph


def _compute_labels(
    graph: TokenGraph,
    trade_wei: int = int(0.1e18),
    gas_cost_eth: float = 0.003,
) -> np.ndarray:
    """
    Label each directed edge 1.0 if it is the forward leg of a profitable 2-hop
    cycle, 0.0 otherwise.
    """
    _, edge_src, edge_dst, edge_feat = graph.to_tensors()
    n_edges = len(edge_src)
    labels = np.zeros(n_edges, dtype=np.float32)

    # Build a map from (src, dst) → list of edge indices
    edge_map: dict = {}
    for i, (s, d) in enumerate(zip(edge_src, edge_dst)):
        key = (int(s), int(d))
        edge_map.setdefault(key, []).append(i)

    for fwd_i, (src, dst) in enumerate(zip(edge_src, edge_dst)):
        bwd_key = (int(dst), int(src))
        if bwd_key not in edge_map:
            continue
        for bwd_i in edge_map[bwd_key]:
            # Decode reserves from features
            ff = edge_feat[fwd_i]
            bf = edge_feat[bwd_i]
            r_in_f  = int(10 ** float(ff[0]) * 1e18)
            r_out_f = int(10 ** float(ff[1]) * 1e18)
            fee_f   = int(round(float(ff[2]) * 100))
            r_in_b  = int(10 ** float(bf[0]) * 1e18)
            r_out_b = int(10 ** float(bf[1]) * 1e18)
            fee_b   = int(round(float(bf[2]) * 100))

            out_fwd = _v2_out_with_fee(trade_wei, r_in_f, r_out_f, fee_f)
            out_bwd = _v2_out_with_fee(out_fwd, r_in_b, r_out_b, fee_b)
            profit  = (out_bwd - trade_wei) / 1e18 - gas_cost_eth
            if profit > 0:
                labels[fwd_i] = 1.0
                break

    return labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 50,
    num_tokens: int = 4,
    num_dexs: int = 2,
    lr: float = 1e-4,
    checkpoint: str = "checkpoints/gnn_arb.pt",
    samples_per_epoch: int = 20,
) -> None:
    model = GNNArbModel()
    best_val_loss = float("inf")
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_losses = []
        for _ in range(samples_per_epoch):
            graph = _random_graph(num_tokens, num_dexs)
            node_feat, edge_src, edge_dst, edge_feat = graph.to_tensors()
            if len(edge_src) == 0:
                continue
            labels = _compute_labels(graph)
            loss = model.train_step(node_feat, edge_src, edge_dst, edge_feat, labels, lr=lr)
            train_losses.append(loss)

        # Validation (separate random graphs)
        val_losses, val_accs = [], []
        for _ in range(max(samples_per_epoch // 4, 1)):
            graph = _random_graph(num_tokens, num_dexs)
            node_feat, edge_src, edge_dst, edge_feat = graph.to_tensors()
            if len(edge_src) == 0:
                continue
            labels = _compute_labels(graph)
            v_loss, v_acc = model.validate(node_feat, edge_src, edge_dst, edge_feat, labels)
            val_losses.append(v_loss)
            val_accs.append(v_acc)

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val   = float(np.mean(val_losses))   if val_losses   else 0.0
        avg_acc   = float(np.mean(val_accs))     if val_accs     else 0.0

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
            epoch, epochs, avg_train, avg_val, avg_acc,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save(checkpoint)
            logger.info("  ✓ New best saved (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete — best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNNArbModel")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--num-tokens",      type=int,   default=4)
    parser.add_argument("--num-dexs",        type=int,   default=2)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--checkpoint",                  default="checkpoints/gnn_arb.pt")
    parser.add_argument("--samples-per-epoch", type=int, default=20)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        num_tokens=args.num_tokens,
        num_dexs=args.num_dexs,
        lr=args.lr,
        checkpoint=args.checkpoint,
        samples_per_epoch=args.samples_per_epoch,
    )

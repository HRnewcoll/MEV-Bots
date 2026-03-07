"""
Transformer-based Arbitrage Bot — neural network model.

Architecture
------------
We treat each DEX that holds a token pair as one "position" in a sequence.
A Transformer encoder applies multi-head self-attention over that sequence so
the model learns cross-DEX dependencies (e.g., "when Uniswap reserve drops,
SushiSwap is likely to have a better price").

Input
-----
Shape: (batch, num_dexs, DEX_FEATURE_DIM)

Per-DEX feature vector (DEX_FEATURE_DIM = 10):
  0   log_reserve_in        log10(reserve_in  / 1e18 + 1e-9)
  1   log_reserve_out       log10(reserve_out / 1e18 + 1e-9)
  2   reserve_ratio         log10(reserve_in / (reserve_out + 1e-30))
  3   fee_bps_norm          fee_bps / 100
  4   dex_id_norm           dex index / num_dexs  (positional hint)
  5   gas_price_norm        gas_price_gwei / 200
  6   liquidity_depth       log10((reserve_in + reserve_out) / 1e18 + 1e-9)
  7   price_impact_1pct     price impact of a 1 % of reserve_in trade
  8   amount_in_norm        trade_amount / 1e18
  9   (reserved / future)   0.0

Output
------
Shape: (batch, num_dexs, num_dexs)

Logits[i][j] = expected log-profit for buying on DEX-i and selling on DEX-j.
The diagonal is masked out (can't arb against yourself).

Training
--------
See training/train.py.  Labels: binary profitable/not-profitable for each
(buy_dex, sell_dex) pair, generated from simulated reserve data.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — TransformerArbModel will use heuristic baseline")

DEX_FEATURE_DIM = 10


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Standard sinusoidal positional encoding for sequences of DEXs."""

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        # shape: (1, max_len, d_model) — broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder model
# ---------------------------------------------------------------------------

class _TransformerEncoder(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Transformer encoder that produces a (batch, num_dexs, d_model) representation,
    followed by a pairwise profit-score head.
    """

    def __init__(
        self,
        feature_dim: int = DEX_FEATURE_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # Pairwise score: [ctx_i || ctx_j] → scalar profit logit
        self.pair_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, num_dexs, feature_dim)
        h = self.input_proj(x)           # (batch, num_dexs, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)          # (batch, num_dexs, d_model)

        # Pairwise scores: for each (i, j) pair compute profit logit
        n = h.size(1)
        # Expand to (batch, n, n, d_model)
        hi = h.unsqueeze(2).expand(-1, -1, n, -1)  # buy-DEX context
        hj = h.unsqueeze(1).expand(-1, n, -1, -1)  # sell-DEX context
        pair = torch.cat([hi, hj], dim=-1)          # (batch, n, n, 2*d_model)
        scores = self.pair_head(pair).squeeze(-1)    # (batch, n, n)

        # Mask diagonal (can't arb i→i)
        mask = torch.eye(n, device=x.device, dtype=torch.bool)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TransformerArbModel:
    """
    Wraps the Transformer encoder with convenience methods for inference and training.

    Falls back to a heuristic (largest reserve spread) when PyTorch is unavailable.
    """

    def __init__(
        self,
        feature_dim: int = DEX_FEATURE_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.device = device

        if _TORCH_AVAILABLE:
            self.model: Optional[_TransformerEncoder] = _TransformerEncoder(
                feature_dim=feature_dim,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.model.to(device)
            self.model.eval()
        else:
            self.model = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def extract_dex_features(
        dex_info: Dict,
        dex_index: int,
        num_dexs: int,
        gas_price_gwei: float = 30.0,
        trade_amount_wei: int = int(0.1 * 1e18),
    ) -> np.ndarray:
        """
        Convert a DEX reserve dict into a fixed-length feature vector.

        ``dex_info`` must contain:
          - ``reserve_in``  (int, wei)
          - ``reserve_out`` (int, wei)
          - ``fee_bps``     (int, e.g. 30)
        """
        reserve_in  = max(int(dex_info.get("reserve_in",  1)), 1)
        reserve_out = max(int(dex_info.get("reserve_out", 1)), 1)
        fee_bps     = float(dex_info.get("fee_bps", 30))

        price_impact = _compute_price_impact(reserve_in, trade_amount_wei)

        # Uniswap V2 output for 1% of reserve_in
        test_in = max(reserve_in // 100, 1)
        out_1pct = _uniswap_v2_out(test_in, reserve_in, reserve_out)
        pi_1pct  = 1.0 - (out_1pct / test_in) * (reserve_in / reserve_out) if out_1pct > 0 else 1.0

        vec = np.zeros(DEX_FEATURE_DIM, dtype=np.float32)
        vec[0] = math.log10(reserve_in  / 1e18 + 1e-9)
        vec[1] = math.log10(reserve_out / 1e18 + 1e-9)
        vec[2] = math.log10(reserve_in  / (reserve_out + 1e-30))
        vec[3] = fee_bps / 100.0
        vec[4] = dex_index / max(num_dexs - 1, 1)
        vec[5] = gas_price_gwei / 200.0
        vec[6] = math.log10((reserve_in + reserve_out) / 1e18 + 1e-9)
        vec[7] = min(float(pi_1pct), 1.0)
        vec[8] = trade_amount_wei / 1e18
        vec[9] = 0.0  # reserved
        return vec

    @staticmethod
    def build_feature_matrix(
        dex_snapshots: List[Dict],
        gas_price_gwei: float = 30.0,
        trade_amount_wei: int = int(0.1 * 1e18),
    ) -> np.ndarray:
        """
        Build a (num_dexs, DEX_FEATURE_DIM) matrix from a list of DEX snapshots.

        Each snapshot dict must contain: reserve_in, reserve_out, fee_bps.
        Returns shape (num_dexs, DEX_FEATURE_DIM).
        """
        n = len(dex_snapshots)
        matrix = np.zeros((n, DEX_FEATURE_DIM), dtype=np.float32)
        for i, snap in enumerate(dex_snapshots):
            matrix[i] = TransformerArbModel.extract_dex_features(
                snap, i, n, gas_price_gwei, trade_amount_wei
            )
        return matrix

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score_pairs(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Return a (num_dexs, num_dexs) matrix of profit logits.

        Positive logit[i][j] → model thinks buying on DEX-i and selling
        on DEX-j is profitable.

        Falls back to a heuristic (price-ratio spread) when PyTorch is absent.
        """
        n = len(feature_matrix)

        if _TORCH_AVAILABLE and self.model is not None:
            import torch
            x = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores = self.model(x).squeeze(0).cpu().numpy()
            return scores
        else:
            # Heuristic: score = log(reserve_out_i / reserve_in_i) - log(reserve_out_j / reserve_in_j)
            # proxy for price difference
            scores = np.full((n, n), float("-inf"))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ri = feature_matrix[i, 2]  # reserve_ratio i
                        rj = feature_matrix[j, 2]  # reserve_ratio j
                        scores[i, j] = rj - ri     # sell where price is high
            return scores

    def best_pair(
        self, dex_snapshots: List[Dict], gas_price_gwei: float = 30.0
    ) -> Optional[Tuple[int, int, float]]:
        """
        Return (buy_dex_index, sell_dex_index, score) for the highest-scoring pair.
        Returns None if fewer than 2 DEXs are available.
        """
        if len(dex_snapshots) < 2:
            return None
        matrix = self.build_feature_matrix(dex_snapshots, gas_price_gwei)
        scores = self.score_pairs(matrix)

        finite = np.isfinite(scores)
        if not np.any(finite):
            return None

        masked = np.where(finite, scores, -1e30)
        idx = int(np.argmax(masked))
        i, j = divmod(idx, len(dex_snapshots))
        return i, j, float(scores[i, j])

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(
        self,
        feature_matrices: np.ndarray,
        labels: np.ndarray,
        lr: float = 1e-4,
    ) -> float:
        """
        One gradient-descent step on a batch.

        Args:
            feature_matrices: (batch, num_dexs, DEX_FEATURE_DIM)
            labels:           (batch, num_dexs, num_dexs) — 1.0 for profitable pairs, 0.0 otherwise
        Returns:
            binary-cross-entropy loss
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return 0.0

        import torch
        import torch.nn as nn

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        x = torch.tensor(feature_matrices, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        optimizer.zero_grad()
        scores = self.model(x)

        # Mask diagonals
        n = scores.size(1)
        mask = torch.eye(n, device=self.device, dtype=torch.bool).unsqueeze(0)
        valid_scores = scores[~mask.expand_as(scores)]
        valid_labels = y[~mask.expand_as(y)]

        loss = F.binary_cross_entropy_with_logits(valid_scores, valid_labels)
        loss.backward()
        optimizer.step()
        self.model.eval()
        return float(loss.item())

    def validate(
        self,
        feature_matrices: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute BCE loss and accuracy on a validation batch.

        Returns: (loss, accuracy)
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return 0.0, 0.0

        import torch

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(feature_matrices, dtype=torch.float32).to(self.device)
            y = torch.tensor(labels, dtype=torch.float32).to(self.device)
            scores = self.model(x)

            n = scores.size(1)
            mask = torch.eye(n, device=self.device, dtype=torch.bool).unsqueeze(0)
            valid_scores = scores[~mask.expand_as(scores)]
            valid_labels = y[~mask.expand_as(y)]

            loss = F.binary_cross_entropy_with_logits(valid_scores, valid_labels)
            preds = (valid_scores > 0).float()
            acc   = (preds == valid_labels).float().mean()
        return float(loss.item()), float(acc.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if not _TORCH_AVAILABLE or self.model is None:
            return
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("TransformerArbModel saved to %s", path)

    def load(self, path: str) -> None:
        if not _TORCH_AVAILABLE or self.model is None:
            return
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("TransformerArbModel loaded from %s", path)


# ---------------------------------------------------------------------------
# Pure math helpers (no PyTorch required)
# ---------------------------------------------------------------------------

def _uniswap_v2_out(amount_in: int, reserve_in: int, reserve_out: int) -> int:
    """Uniswap V2 constant-product formula (0.3 % fee)."""
    if reserve_in <= 0 or reserve_out <= 0 or amount_in <= 0:
        return 0
    amount_in_fee = amount_in * 997
    return (amount_in_fee * reserve_out) // (reserve_in * 1000 + amount_in_fee)


def _compute_price_impact(reserve_in: int, amount_in: int) -> float:
    """Fraction by which the effective price is worse than the mid-price."""
    if reserve_in <= 0 or amount_in <= 0:
        return 0.0
    # mid price: reserve_out / reserve_in (cancel out)
    # effective: (amount_in * 997) / (reserve_in * 1000 + amount_in * 997)
    effective_fraction = (amount_in * 997) / (reserve_in * 1000 + amount_in * 997)
    mid_fraction = amount_in / (reserve_in + amount_in)
    if mid_fraction <= 0:
        return 0.0
    return max(0.0, 1.0 - effective_fraction / mid_fraction)

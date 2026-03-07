"""
Graph Neural Network (GNN) for multi-hop arbitrage path scoring.

Architecture
------------
We model the DEX ecosystem as a directed weighted graph:

  Nodes  → tokens  (WETH, USDC, DAI, WBTC, …)
  Edges  → liquidity pools  (each pool has two directed edges: token0→token1
           and token1→token0 with different effective prices)

The GNN applies multiple rounds of message-passing to propagate liquidity and
price information across the graph.  After K rounds the node embeddings capture
"how easy is it to reach any other token from this one at a good price".

A path-scoring head takes pairs of (source_node_emb, dest_node_emb) and
predicts expected profit for the hop.

For multi-hop paths (length 2 or 3) we score the product of edge profits along
each candidate path using the node embeddings as residual shortcuts.

This is architecturally different from both the LSTM predictor (no temporal
dimension here — we reason over graph structure) and the Transformer bot
(which attends over DEX sequences for a single pair, not the full token graph).

Design goals
  * No external graph library required — pure PyTorch + NumPy
  * Graceful degradation (Bellman-Ford heuristic) when PyTorch is absent
  * All shapes are fully dynamic: add/remove tokens/pools at runtime

Key types
---------
TokenGraph:  Pure-Python / NumPy data structure that holds the graph.
GNNArbModel: Wraps the torch module + inference / training helpers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — GNNArbModel will use Bellman-Ford heuristic")

# Per-edge feature dimension
EDGE_FEATURE_DIM = 8
# Per-node feature dimension (initialised from edge aggregation)
NODE_FEATURE_DIM = 8


# ---------------------------------------------------------------------------
# Token Graph — pure Python/NumPy
# ---------------------------------------------------------------------------

class TokenGraph:
    """
    Directed weighted graph of tokens and liquidity pools.

    Edges carry on-chain features; nodes start with aggregated summaries
    of their adjacent edges.
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, int] = {}          # address → node_id
        self._token_names: List[str] = []          # node_id → address
        # Each edge: (src_id, dst_id, features_array)
        self._edges: List[Tuple[int, int, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _token_id(self, address: str) -> int:
        addr = address.lower()
        if addr not in self._tokens:
            self._tokens[addr] = len(self._token_names)
            self._token_names.append(addr)
        return self._tokens[addr]

    def add_pool(
        self,
        token_a: str,
        token_b: str,
        reserve_a: int,
        reserve_b: int,
        fee_bps: int,
        dex_name: str = "",
        pool_index: int = 0,
        total_pools: int = 1,
    ) -> None:
        """
        Add both directed edges (a→b and b→a) for a liquidity pool.
        """
        a_id = self._token_id(token_a)
        b_id = self._token_id(token_b)

        f_ab = _pool_edge_features(reserve_a, reserve_b, fee_bps, pool_index, total_pools)
        f_ba = _pool_edge_features(reserve_b, reserve_a, fee_bps, pool_index, total_pools)

        self._edges.append((a_id, b_id, f_ab))
        self._edges.append((b_id, a_id, f_ba))

    @property
    def num_nodes(self) -> int:
        return len(self._token_names)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------
    # Tensor conversion helpers
    # ------------------------------------------------------------------

    def to_tensors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert graph to arrays ready for the GNN.

        Returns:
            node_features:   (num_nodes, NODE_FEATURE_DIM)  — aggregated from edges
            edge_index_src:  (num_edges,) int               — source node ids
            edge_index_dst:  (num_edges,) int               — dest node ids
            edge_features:   (num_edges, EDGE_FEATURE_DIM)
        """
        n = self.num_nodes
        e = self.num_edges
        if n == 0 or e == 0:
            return (
                np.zeros((0, NODE_FEATURE_DIM), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32),
            )

        edge_src = np.zeros(e, dtype=np.int64)
        edge_dst = np.zeros(e, dtype=np.int64)
        edge_feat = np.zeros((e, EDGE_FEATURE_DIM), dtype=np.float32)

        for idx, (src, dst, feat) in enumerate(self._edges):
            edge_src[idx]   = src
            edge_dst[idx]   = dst
            edge_feat[idx]  = feat

        # Node features = mean of outgoing edge features (simple aggregation)
        node_feat = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
        counts    = np.zeros(n, dtype=np.float32)
        for idx, (src, dst, feat) in enumerate(self._edges):
            node_feat[src] += feat
            counts[src]    += 1
        counts = np.maximum(counts, 1)
        node_feat /= counts[:, None]

        return node_feat, edge_src, edge_dst, edge_feat

    # ------------------------------------------------------------------
    # Bellman-Ford heuristic (no PyTorch)
    # ------------------------------------------------------------------

    def best_single_hop_heuristic(
        self, amount_in_wei: int = int(0.1 * 1e18)
    ) -> Optional[Tuple[str, str, float]]:
        """
        Return (token_a_addr, token_b_addr, estimated_net_profit_eth) for
        the most profitable single-hop cycle: A→B on one pool, B→A on another.
        """
        # Collect best output for each (token_a, token_b) ordered pair
        best_rate: Dict[Tuple[int, int], Tuple[float, Tuple[int, int, np.ndarray]]] = {}
        for edge in self._edges:
            src, dst, feat = edge
            r_in  = feat[0]  # log_reserve_in encoded; decode back
            r_out = feat[1]
            # Decode: feat[0] = log10(reserve_in/1e18 + 1e-9) → reserve_in ≈ 10^feat[0] * 1e18
            res_in  = 10 ** float(feat[0]) * 1e18
            res_out = 10 ** float(feat[1]) * 1e18
            fee_bps = int(round(float(feat[2]) * 100))
            out = _v2_out_with_fee(amount_in_wei, int(res_in), int(res_out), fee_bps)
            rate = out / amount_in_wei if amount_in_wei > 0 else 0
            key = (src, dst)
            if key not in best_rate or rate > best_rate[key][0]:
                best_rate[key] = (rate, edge)

        best_profit = 0.0
        best_path: Optional[Tuple[str, str, float]] = None
        n = self.num_nodes
        GAS_ETH = 0.003

        for src in range(n):
            for mid in range(n):
                if src == mid:
                    continue
                for dst_back in range(n):
                    if dst_back != src:
                        continue
                    k_fwd = (src, mid)
                    k_bwd = (mid, src)
                    if k_fwd not in best_rate or k_bwd not in best_rate:
                        continue
                    out_fwd = best_rate[k_fwd][0] * amount_in_wei
                    out_bwd = best_rate[k_bwd][0] * out_fwd
                    profit = (out_bwd - amount_in_wei) / 1e18 - GAS_ETH
                    if profit > best_profit:
                        best_profit = profit
                        ta = self._token_names[src]
                        tb = self._token_names[mid]
                        best_path = (ta, tb, profit)

        return best_path


# ---------------------------------------------------------------------------
# GNN torch module
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class _MessagePassingLayer(nn.Module):
        """
        One round of message passing:
          1. Compute edge messages from (src_emb, dst_emb, edge_feat)
          2. Aggregate incoming messages at each node (mean pooling)
          3. Update node embedding via GRU-style gate
        """

        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
            super().__init__()
            # Message network: [src_emb || dst_emb || edge_feat] → message
            self.msg_net = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, node_dim),
            )
            # Node update: GRU-like gate
            self.update_gate = nn.GRUCell(node_dim, node_dim)

        def forward(
            self,
            node_emb:   "torch.Tensor",    # (N, node_dim)
            edge_src:   "torch.Tensor",    # (E,) int
            edge_dst:   "torch.Tensor",    # (E,) int
            edge_feat:  "torch.Tensor",    # (E, edge_dim)
        ) -> "torch.Tensor":               # (N, node_dim)
            N = node_emb.size(0)

            src_emb  = node_emb[edge_src]           # (E, node_dim)
            dst_emb  = node_emb[edge_dst]           # (E, node_dim)
            msg_input = torch.cat([src_emb, dst_emb, edge_feat], dim=-1)
            messages  = self.msg_net(msg_input)      # (E, node_dim)

            # Mean aggregate incoming messages
            agg = torch.zeros(N, node_emb.size(1), device=node_emb.device)
            agg.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(messages), messages)
            counts = torch.zeros(N, 1, device=node_emb.device)
            counts.scatter_add_(0, edge_dst.unsqueeze(1), torch.ones(len(edge_dst), 1, device=node_emb.device))
            counts = counts.clamp(min=1)
            agg = agg / counts

            # GRU update
            new_emb = self.update_gate(agg, node_emb)
            return new_emb

    class _GNNModule(nn.Module):
        """
        Full GNN: K message-passing rounds + edge profit scoring head.
        """

        def __init__(
            self,
            node_feature_dim: int = NODE_FEATURE_DIM,
            edge_feature_dim: int = EDGE_FEATURE_DIM,
            hidden_dim: int = 64,
            num_layers: int = 3,
        ) -> None:
            super().__init__()
            self.node_emb_proj = nn.Linear(node_feature_dim, hidden_dim)
            self.edge_emb_proj = nn.Linear(edge_feature_dim, edge_feature_dim)
            self.layers = nn.ModuleList([
                _MessagePassingLayer(hidden_dim, edge_feature_dim, hidden_dim)
                for _ in range(num_layers)
            ])
            # Edge profit head: [src_final || dst_final || edge_feat] → profit_logit
            self.profit_head = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            node_feat:  "torch.Tensor",   # (N, node_feature_dim)
            edge_src:   "torch.Tensor",   # (E,) int
            edge_dst:   "torch.Tensor",   # (E,) int
            edge_feat:  "torch.Tensor",   # (E, edge_feature_dim)
        ) -> "torch.Tensor":              # (E,) profit logits per directed edge
            node_emb = self.node_emb_proj(node_feat)     # (N, hidden)
            edge_emb = self.edge_emb_proj(edge_feat)     # (E, edge_dim)

            for layer in self.layers:
                node_emb = layer(node_emb, edge_src, edge_dst, edge_emb)

            # Score each directed edge
            src_emb = node_emb[edge_src]
            dst_emb = node_emb[edge_dst]
            edge_input = torch.cat([src_emb, dst_emb, edge_emb], dim=-1)
            logits = self.profit_head(edge_input).squeeze(-1)  # (E,)
            return logits

else:
    _MessagePassingLayer = None  # type: ignore
    _GNNModule = None            # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GNNArbModel:
    """
    Graph Neural Network for multi-hop arbitrage opportunity scoring.

    Wraps _GNNModule with feature engineering, training helpers, and
    save/load.  Falls back to TokenGraph.best_single_hop_heuristic()
    when PyTorch is unavailable.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        device: str = "cpu",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        if _TORCH_AVAILABLE:
            self.model: Optional[_GNNModule] = _GNNModule(
                node_feature_dim=NODE_FEATURE_DIM,
                edge_feature_dim=EDGE_FEATURE_DIM,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
            self.model.to(device)
            self.model.eval()
        else:
            self.model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score_edges(self, graph: TokenGraph) -> Optional[np.ndarray]:
        """
        Score all directed edges in *graph*.

        Returns an (num_edges,) numpy array of profit logits, or None
        if the graph is empty.
        """
        node_feat, edge_src, edge_dst, edge_feat = graph.to_tensors()
        if len(edge_src) == 0:
            return None

        if _TORCH_AVAILABLE and self.model is not None:
            import torch
            nf = torch.tensor(node_feat, dtype=torch.float32).to(self.device)
            es = torch.tensor(edge_src,  dtype=torch.long).to(self.device)
            ed = torch.tensor(edge_dst,  dtype=torch.long).to(self.device)
            ef = torch.tensor(edge_feat, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(nf, es, ed, ef)
            return logits.cpu().numpy()

        # Heuristic: use reserve ratio from edge features
        scores = np.zeros(len(edge_src), dtype=np.float32)
        for i, feat in enumerate(edge_feat):
            # feat[0] = log_reserve_in, feat[1] = log_reserve_out
            scores[i] = float(feat[1] - feat[0])  # higher → better for buyer
        return scores

    def best_cycle(
        self,
        graph: TokenGraph,
        amount_in_wei: int = int(0.1 * 1e18),
        gas_cost_eth: float = 0.003,
    ) -> Optional[Dict]:
        """
        Find the most profitable 2-hop cycle (A → B → A) according to the
        GNN edge scores.

        Returns a dict with path info or None.
        """
        logits = self.score_edges(graph)
        if logits is None:
            return None

        _, edge_src, edge_dst, edge_feat = graph.to_tensors()
        n = graph.num_nodes
        token_names = graph._token_names

        # Build index: (src, dst) → (edge_index, score)
        edge_map: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for i, (s, d) in enumerate(zip(edge_src, edge_dst)):
            key = (int(s), int(d))
            score = float(logits[i])
            if key not in edge_map or score > edge_map[key][1]:
                edge_map[key] = (i, score)

        best_profit = 0.0
        best_result: Optional[Dict] = None

        for src in range(n):
            for mid in range(n):
                if src == mid:
                    continue
                k_fwd = (src, mid)
                k_bwd = (mid, src)
                if k_fwd not in edge_map or k_bwd not in edge_map:
                    continue

                fwd_idx, fwd_score = edge_map[k_fwd]
                bwd_idx, bwd_score = edge_map[k_bwd]

                # Arithmetic verification
                fwd_feat = edge_feat[fwd_idx]
                bwd_feat = edge_feat[bwd_idx]
                r_in_fwd  = int(10 ** float(fwd_feat[0]) * 1e18)
                r_out_fwd = int(10 ** float(fwd_feat[1]) * 1e18)
                fee_fwd   = int(round(float(fwd_feat[2]) * 100))
                r_in_bwd  = int(10 ** float(bwd_feat[0]) * 1e18)
                r_out_bwd = int(10 ** float(bwd_feat[1]) * 1e18)
                fee_bwd   = int(round(float(bwd_feat[2]) * 100))

                out_fwd = _v2_out_with_fee(amount_in_wei, r_in_fwd, r_out_fwd, fee_fwd)
                out_bwd = _v2_out_with_fee(out_fwd,       r_in_bwd, r_out_bwd, fee_bwd)
                profit  = (out_bwd - amount_in_wei) / 1e18 - gas_cost_eth

                if profit > best_profit:
                    best_profit = profit
                    best_result = {
                        "token_start": token_names[src],
                        "token_mid":   token_names[mid],
                        "fwd_edge_idx": fwd_idx,
                        "bwd_edge_idx": bwd_idx,
                        "fwd_score": fwd_score,
                        "bwd_score": bwd_score,
                        "net_profit_eth": profit,
                        "out_fwd": out_fwd,
                        "out_bwd": out_bwd,
                    }

        return best_result

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(
        self,
        node_feat:  np.ndarray,
        edge_src:   np.ndarray,
        edge_dst:   np.ndarray,
        edge_feat:  np.ndarray,
        labels:     np.ndarray,
        lr: float = 1e-4,
    ) -> float:
        """
        One gradient step on a single graph sample.

        labels: (num_edges,) — 1.0 if edge is part of a profitable cycle.
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return 0.0

        import torch

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        nf = torch.tensor(node_feat, dtype=torch.float32).to(self.device)
        es = torch.tensor(edge_src,  dtype=torch.long).to(self.device)
        ed = torch.tensor(edge_dst,  dtype=torch.long).to(self.device)
        ef = torch.tensor(edge_feat, dtype=torch.float32).to(self.device)
        y  = torch.tensor(labels,    dtype=torch.float32).to(self.device)

        optimizer.zero_grad()
        logits = self.model(nf, es, ed, ef)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        self.model.eval()
        return float(loss.item())

    def validate(
        self,
        node_feat:  np.ndarray,
        edge_src:   np.ndarray,
        edge_dst:   np.ndarray,
        edge_feat:  np.ndarray,
        labels:     np.ndarray,
    ) -> Tuple[float, float]:
        """Compute (loss, accuracy) on a single graph sample."""
        if not _TORCH_AVAILABLE or self.model is None:
            return 0.0, 0.0

        import torch

        self.model.eval()
        with torch.no_grad():
            nf = torch.tensor(node_feat, dtype=torch.float32).to(self.device)
            es = torch.tensor(edge_src,  dtype=torch.long).to(self.device)
            ed = torch.tensor(edge_dst,  dtype=torch.long).to(self.device)
            ef = torch.tensor(edge_feat, dtype=torch.float32).to(self.device)
            y  = torch.tensor(labels,    dtype=torch.float32).to(self.device)
            logits = self.model(nf, es, ed, ef)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            preds = (logits > 0).float()
            acc   = (preds == y).float().mean()
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
        logger.info("GNNArbModel saved to %s", path)

    def load(self, path: str) -> None:
        if not _TORCH_AVAILABLE or self.model is None:
            return
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("GNNArbModel loaded from %s", path)


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def _pool_edge_features(
    reserve_in: int,
    reserve_out: int,
    fee_bps: int,
    pool_index: int = 0,
    total_pools: int = 1,
) -> np.ndarray:
    """
    Build an EDGE_FEATURE_DIM feature vector for a directed pool edge.

    Features:
      0  log10(reserve_in / 1e18 + 1e-9)
      1  log10(reserve_out / 1e18 + 1e-9)
      2  fee_bps / 100
      3  log10(reserve_in / (reserve_out + 1e-30))   — price direction
      4  log10((reserve_in + reserve_out) / 1e18 + 1e-9)  — liquidity depth
      5  pool_index / max(total_pools - 1, 1)         — pool identity
      6  (reserved)
      7  (reserved)
    """
    r_in  = max(int(reserve_in),  1)
    r_out = max(int(reserve_out), 1)
    feat = np.zeros(EDGE_FEATURE_DIM, dtype=np.float32)
    feat[0] = math.log10(r_in  / 1e18 + 1e-9)
    feat[1] = math.log10(r_out / 1e18 + 1e-9)
    feat[2] = fee_bps / 100.0
    feat[3] = math.log10(r_in / (r_out + 1e-30))
    feat[4] = math.log10((r_in + r_out) / 1e18 + 1e-9)
    feat[5] = pool_index / max(total_pools - 1, 1)
    feat[6] = 0.0
    feat[7] = 0.0
    return feat


def _v2_out_with_fee(amount_in: int, reserve_in: int, reserve_out: int, fee_bps: int = 30) -> int:
    """Generalised Uniswap V2 output formula for an arbitrary fee (in basis points)."""
    if reserve_in <= 0 or reserve_out <= 0 or amount_in <= 0:
        return 0
    k = 10_000 - fee_bps           # e.g. 9970 for 0.30 % fee
    fee_denom = 10_000
    num = amount_in * k * reserve_out
    den = reserve_in * fee_denom + amount_in * k
    return num // den

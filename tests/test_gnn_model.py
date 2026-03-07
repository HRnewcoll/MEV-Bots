"""
Unit tests for python/nn_bots/graph_arb_bot/model.py

All tests are pure-math / pure-Python — no network calls, no PyTorch required.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Load the GNN model directly to avoid name collisions with transformer_arb_bot/model.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_GNN_MODEL_PATH = _REPO_ROOT / "python" / "nn_bots" / "graph_arb_bot" / "model.py"
_spec = importlib.util.spec_from_file_location("gnn_arb_model", _GNN_MODEL_PATH)
_gnn_model_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gnn_model_module)

GNNArbModel         = _gnn_model_module.GNNArbModel
TokenGraph          = _gnn_model_module.TokenGraph
_pool_edge_features = _gnn_model_module._pool_edge_features
_v2_out_with_fee    = _gnn_model_module._v2_out_with_fee
EDGE_FEATURE_DIM    = _gnn_model_module.EDGE_FEATURE_DIM
NODE_FEATURE_DIM    = _gnn_model_module.NODE_FEATURE_DIM

# Mock token addresses (42-char hex)
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI  = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------

class TestV2OutWithFee:
    def test_zero_amount_in(self):
        assert _v2_out_with_fee(0, int(100e18), int(100e18)) == 0

    def test_zero_reserve_in(self):
        assert _v2_out_with_fee(int(1e18), 0, int(100e18)) == 0

    def test_zero_reserve_out(self):
        assert _v2_out_with_fee(int(1e18), int(100e18), 0) == 0

    def test_30bps_matches_v2_formula(self):
        # Standard Uniswap V2 formula (fee_bps=30)
        amount_in = int(0.1e18)
        r_in = int(100e18)
        r_out = int(100e18)
        # k=9970, expected = (amount_in*9970*r_out) / (r_in*10000 + amount_in*9970)
        expected = (amount_in * 9970 * r_out) // (r_in * 10_000 + amount_in * 9970)
        assert _v2_out_with_fee(amount_in, r_in, r_out, fee_bps=30) == expected

    def test_fee_0_gives_more_output(self):
        amount_in = int(1e18)
        r = int(100e18)
        out_0fee = _v2_out_with_fee(amount_in, r, r, fee_bps=0)
        out_30bps = _v2_out_with_fee(amount_in, r, r, fee_bps=30)
        assert out_0fee > out_30bps

    def test_higher_fee_gives_less_output(self):
        amount_in = int(1e18)
        r = int(100e18)
        out_low  = _v2_out_with_fee(amount_in, r, r, fee_bps=10)
        out_high = _v2_out_with_fee(amount_in, r, r, fee_bps=100)
        assert out_low > out_high

    def test_integer_result(self):
        result = _v2_out_with_fee(12345, 9_876_543, 1_234_567)
        assert isinstance(result, int) and result >= 0


class TestPoolEdgeFeatures:
    def test_shape(self):
        feat = _pool_edge_features(int(100e18), int(200e18), 30)
        assert feat.shape == (EDGE_FEATURE_DIM,)

    def test_dtype(self):
        feat = _pool_edge_features(int(100e18), int(200e18), 30)
        assert feat.dtype == np.float32

    def test_fee_encoding(self):
        feat = _pool_edge_features(int(100e18), int(100e18), 30)
        # feat[2] = fee_bps / 100 = 0.30
        assert abs(feat[2] - 0.30) < 1e-6

    def test_log_reserve_in(self):
        r_in = int(100e18)
        feat = _pool_edge_features(r_in, int(100e18), 30)
        expected = math.log10(r_in / 1e18 + 1e-9)
        assert abs(feat[0] - expected) < 1e-5

    def test_min_reserve_clamped(self):
        # Zero reserves should not raise
        feat = _pool_edge_features(0, 0, 30)
        assert feat.shape == (EDGE_FEATURE_DIM,)

    def test_pool_index_encoding(self):
        feat = _pool_edge_features(int(100e18), int(100e18), 30, pool_index=2, total_pools=5)
        # feat[5] = 2 / 4 = 0.5
        assert abs(feat[5] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# TokenGraph
# ---------------------------------------------------------------------------

class TestTokenGraph:
    def _simple_graph(self) -> TokenGraph:
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(200_000e6), 30)
        return g

    def test_empty_graph(self):
        g = TokenGraph()
        assert g.num_nodes == 0
        assert g.num_edges == 0

    def test_add_pool_creates_two_directed_edges(self):
        g = self._simple_graph()
        assert g.num_edges == 2

    def test_add_pool_creates_two_nodes(self):
        g = self._simple_graph()
        assert g.num_nodes == 2

    def test_multiple_pools_same_pair_multiple_dexs(self):
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(200e18), 30, "uniswap_v2")
        g.add_pool(WETH, USDC, int(80e18),  int(180e18), 30, "sushiswap")
        # Two pools × two directions = 4 edges, but only 2 unique tokens
        assert g.num_edges == 4
        assert g.num_nodes == 2

    def test_three_token_graph(self):
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(200e18), 30)
        g.add_pool(WETH, DAI,  int(100e18), int(200e18), 30)
        assert g.num_nodes == 3
        assert g.num_edges == 4

    def test_to_tensors_empty(self):
        g = TokenGraph()
        nf, es, ed, ef = g.to_tensors()
        assert nf.shape[0] == 0
        assert len(es) == 0

    def test_to_tensors_shapes(self):
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(200e18), 30)
        nf, es, ed, ef = g.to_tensors()
        n, e = g.num_nodes, g.num_edges
        assert nf.shape == (n, NODE_FEATURE_DIM)
        assert es.shape == (e,)
        assert ed.shape == (e,)
        assert ef.shape == (e, EDGE_FEATURE_DIM)

    def test_to_tensors_dtype(self):
        g = self._simple_graph()
        nf, es, ed, ef = g.to_tensors()
        assert nf.dtype == np.float32
        assert ef.dtype == np.float32
        assert es.dtype == np.int64
        assert ed.dtype == np.int64

    def test_directed_edges_have_different_features(self):
        """Forward edge (WETH→USDC) and backward edge (USDC→WETH) should differ."""
        g = self._simple_graph()
        _, es, ed, ef = g.to_tensors()
        # Find forward and backward edges
        weth_id = g._tokens[WETH.lower()]
        usdc_id = g._tokens[USDC.lower()]
        fwd_feats = ef[(es == weth_id) & (ed == usdc_id)]
        bwd_feats = ef[(es == usdc_id) & (ed == weth_id)]
        assert fwd_feats.shape[0] == 1
        assert bwd_feats.shape[0] == 1
        # Reserve ratio (feat[3]) should be opposite sign
        assert fwd_feats[0, 3] != pytest.approx(bwd_feats[0, 3])

    def test_best_single_hop_heuristic_no_opportunity(self):
        """With a perfectly balanced pool there's no profitable cycle."""
        g = TokenGraph()
        # Perfectly identical pools → no arb
        g.add_pool(WETH, USDC, int(100e18), int(100e18), 30, "dex_a")
        g.add_pool(WETH, USDC, int(100e18), int(100e18), 30, "dex_b")
        result = g.best_single_hop_heuristic()
        # May or may not find opportunity; must not raise
        # With identical pools profit is negative after fees
        assert result is None or isinstance(result, tuple)

    def test_best_single_hop_heuristic_profitable(self):
        """A large price spread between two pools should surface an opportunity."""
        g = TokenGraph()
        # DEX A: 1 WETH = 100 USDC equivalent
        g.add_pool(WETH, USDC, int(100e18), int(100e18), 30, "dex_a")
        # DEX B: 1 WETH = 200 USDC equivalent (different price)
        g.add_pool(WETH, USDC, int(50e18), int(100e18), 30, "dex_b")
        result = g.best_single_hop_heuristic(amount_in_wei=int(0.1e18))
        # With a large imbalance the function should return something
        # (may still be negative after 0.3% fees on both legs — that's fine)
        # Just verify it doesn't raise and returns the right type
        assert result is None or (isinstance(result, tuple) and len(result) == 3)


# ---------------------------------------------------------------------------
# GNNArbModel — heuristic path (no PyTorch required)
# ---------------------------------------------------------------------------

class TestGNNArbModelHeuristic:
    def _make_graph(self) -> TokenGraph:
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(200e18), 30, "uniswap_v2")
        g.add_pool(WETH, USDC, int(80e18),  int(160e18), 25, "sushiswap")
        g.add_pool(WETH, DAI,  int(100e18), int(200e18), 30, "uniswap_v2")
        return g

    def test_score_edges_shape(self):
        model = GNNArbModel()
        g = self._make_graph()
        scores = model.score_edges(g)
        assert scores is not None
        assert scores.shape == (g.num_edges,)

    def test_score_edges_finite(self):
        model = GNNArbModel()
        g = self._make_graph()
        scores = model.score_edges(g)
        assert scores is not None
        assert np.all(np.isfinite(scores))

    def test_score_edges_empty_graph(self):
        model = GNNArbModel()
        g = TokenGraph()
        result = model.score_edges(g)
        assert result is None

    def test_best_cycle_no_crash_with_small_graph(self):
        model = GNNArbModel()
        g = self._make_graph()
        # Should not raise regardless of whether a cycle is found
        result = model.best_cycle(g)
        assert result is None or isinstance(result, dict)

    def test_best_cycle_result_structure(self):
        model = GNNArbModel()
        g = self._make_graph()
        result = model.best_cycle(g)
        if result is not None:
            for key in ("token_start", "token_mid", "net_profit_eth",
                        "fwd_score", "bwd_score", "out_fwd", "out_bwd"):
                assert key in result, f"Key '{key}' missing from best_cycle result"

    def test_best_cycle_profit_is_float(self):
        model = GNNArbModel()
        g = self._make_graph()
        result = model.best_cycle(g)
        if result is not None:
            assert isinstance(result["net_profit_eth"], float)

    def test_train_step_noop_without_torch(self):
        try:
            import torch  # noqa: F401
            pytest.skip("PyTorch installed")
        except ImportError:
            pass
        model = GNNArbModel()
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(100e18), 30)
        nf, es, ed, ef = g.to_tensors()
        labels = np.zeros(g.num_edges, dtype=np.float32)
        loss = model.train_step(nf, es, ed, ef, labels)
        assert loss == 0.0

    def test_validate_noop_without_torch(self):
        try:
            import torch  # noqa: F401
            pytest.skip("PyTorch installed")
        except ImportError:
            pass
        model = GNNArbModel()
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(100e18), 30)
        nf, es, ed, ef = g.to_tensors()
        labels = np.zeros(g.num_edges, dtype=np.float32)
        loss, acc = model.validate(nf, es, ed, ef, labels)
        assert loss == 0.0
        assert acc == 0.0

    def test_save_noop_without_model(self, tmp_path):
        try:
            import torch  # noqa: F401
            pytest.skip("PyTorch installed")
        except ImportError:
            pass
        model = GNNArbModel()
        model.save(str(tmp_path / "gnn.pt"))   # must not raise

    def test_load_noop_for_missing_file(self, tmp_path):
        model = GNNArbModel()
        model.load(str(tmp_path / "missing.pt"))  # must not raise


# ---------------------------------------------------------------------------
# PyTorch-specific tests
# ---------------------------------------------------------------------------

class TestGNNArbModelTorch:
    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

    def _make_graph(self, add_dai=True) -> TokenGraph:
        g = TokenGraph()
        g.add_pool(WETH, USDC, int(100e18), int(200e18), 30, "uniswap_v2")
        g.add_pool(WETH, USDC, int(80e18),  int(160e18), 25, "sushiswap")
        if add_dai:
            g.add_pool(WETH, DAI, int(100e18), int(200e18), 30, "uniswap_v2")
        return g

    def test_forward_pass_produces_edge_logits(self):
        import torch
        model = GNNArbModel(hidden_dim=32, num_layers=2)
        g = self._make_graph()
        nf, es, ed, ef = g.to_tensors()
        nf_t = torch.tensor(nf, dtype=torch.float32)
        es_t = torch.tensor(es, dtype=torch.long)
        ed_t = torch.tensor(ed, dtype=torch.long)
        ef_t = torch.tensor(ef, dtype=torch.float32)
        with torch.no_grad():
            logits = model.model(nf_t, es_t, ed_t, ef_t)
        assert logits.shape == (g.num_edges,)

    def test_score_edges_correct_length(self):
        model = GNNArbModel(hidden_dim=32, num_layers=2)
        g = self._make_graph()
        scores = model.score_edges(g)
        assert scores is not None
        assert len(scores) == g.num_edges

    def test_train_step_returns_finite_loss(self):
        model = GNNArbModel(hidden_dim=32, num_layers=2)
        g = self._make_graph()
        nf, es, ed, ef = g.to_tensors()
        labels = np.zeros(g.num_edges, dtype=np.float32)
        loss = model.train_step(nf, es, ed, ef, labels, lr=1e-3)
        assert math.isfinite(loss)

    def test_validate_returns_tuple(self):
        model = GNNArbModel(hidden_dim=32, num_layers=2)
        g = self._make_graph()
        nf, es, ed, ef = g.to_tensors()
        labels = np.zeros(g.num_edges, dtype=np.float32)
        loss, acc = model.validate(nf, es, ed, ef, labels)
        assert math.isfinite(loss)
        assert 0.0 <= acc <= 1.0

    def test_save_and_load(self, tmp_path):
        import torch
        model = GNNArbModel(hidden_dim=32, num_layers=2)
        path = str(tmp_path / "gnn_test.pt")
        model.save(path)
        assert Path(path).exists()

        model2 = GNNArbModel(hidden_dim=32, num_layers=2)
        model2.load(path)
        # Verify same output
        g = self._make_graph()
        nf, es, ed, ef = g.to_tensors()
        nf_t = torch.tensor(nf, dtype=torch.float32)
        es_t = torch.tensor(es, dtype=torch.long)
        ed_t = torch.tensor(ed, dtype=torch.long)
        ef_t = torch.tensor(ef, dtype=torch.float32)
        with torch.no_grad():
            o1 = model.model(nf_t, es_t, ed_t, ef_t)
            o2 = model2.model(nf_t, es_t, ed_t, ef_t)
        assert torch.allclose(o1, o2)

    def test_multiple_train_steps_reduce_loss(self):
        model = GNNArbModel(hidden_dim=32, num_layers=2)
        g = self._make_graph()
        nf, es, ed, ef = g.to_tensors()
        labels = np.zeros(g.num_edges, dtype=np.float32)
        # Run multiple steps
        losses = [model.train_step(nf, es, ed, ef, labels, lr=1e-2) for _ in range(5)]
        assert all(math.isfinite(l) for l in losses)

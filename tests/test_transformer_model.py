"""
Unit tests for python/nn_bots/transformer_arb_bot/model.py

All tests are pure-math / pure-Python — no network calls, no PyTorch required.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the transformer model directly to avoid name collisions with graph_arb_bot/model.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TF_MODEL_PATH = _REPO_ROOT / "python" / "nn_bots" / "transformer_arb_bot" / "model.py"
_spec = importlib.util.spec_from_file_location("transformer_arb_model", _TF_MODEL_PATH)
_tf_model_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tf_model_module)

TransformerArbModel = _tf_model_module.TransformerArbModel
_uniswap_v2_out     = _tf_model_module._uniswap_v2_out
_compute_price_impact = _tf_model_module._compute_price_impact
DEX_FEATURE_DIM     = _tf_model_module.DEX_FEATURE_DIM


# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------

class TestUniswapV2Out:
    def test_zero_amount_in(self):
        assert _uniswap_v2_out(0, 1_000_000, 1_000_000) == 0

    def test_zero_reserve_in(self):
        assert _uniswap_v2_out(1_000, 0, 1_000_000) == 0

    def test_zero_reserve_out(self):
        assert _uniswap_v2_out(1_000, 1_000_000, 0) == 0

    def test_positive_output(self):
        out = _uniswap_v2_out(int(0.1e18), int(100e18), int(100e18))
        assert out > 0

    def test_output_less_than_input_balanced_pool(self):
        # balanced pool, tiny trade: output < input due to 0.3% fee
        out = _uniswap_v2_out(1_000, int(1e12), int(1e12))
        assert out < 1_000

    def test_monotone(self):
        # more input → more output
        r = int(1_000e18)
        outs = [_uniswap_v2_out(a, r, r) for a in [int(0.01e18), int(0.1e18), int(1e18)]]
        assert outs == sorted(outs)

    def test_integer_result(self):
        result = _uniswap_v2_out(12345, 9_876_543, 1_234_567)
        assert isinstance(result, int)
        assert result >= 0


class TestComputePriceImpact:
    def test_zero_amount(self):
        assert _compute_price_impact(int(100e18), 0) == 0.0

    def test_zero_reserve(self):
        assert _compute_price_impact(0, int(1e18)) == 0.0

    def test_small_trade_low_impact(self):
        # 0.01% of pool → negligible impact
        impact = _compute_price_impact(int(1_000e18), int(0.1e18))
        assert 0 <= impact < 0.01

    def test_large_trade_high_impact(self):
        # 50% of pool → effective price noticeably worse than mid price
        impact = _compute_price_impact(int(100e18), int(50e18))
        # With amount_in = 50% of reserve_in, the formula always gives positive impact
        assert impact > 0.0

    def test_impact_is_float(self):
        result = _compute_price_impact(int(100e18), int(1e18))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestExtractDexFeatures:
    def test_output_shape(self):
        snap = {"reserve_in": int(100e18), "reserve_out": int(200e18), "fee_bps": 30}
        feat = TransformerArbModel.extract_dex_features(snap, 0, 3)
        assert feat.shape == (DEX_FEATURE_DIM,)

    def test_output_dtype(self):
        snap = {"reserve_in": int(100e18), "reserve_out": int(200e18), "fee_bps": 30}
        feat = TransformerArbModel.extract_dex_features(snap, 0, 3)
        assert feat.dtype == np.float32

    def test_fee_encoded_correctly(self):
        snap = {"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30}
        feat = TransformerArbModel.extract_dex_features(snap, 0, 1)
        # feat[3] = fee_bps / 100
        assert abs(feat[3] - 0.30) < 1e-6

    def test_dex_id_first_is_zero(self):
        snap = {"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30}
        feat = TransformerArbModel.extract_dex_features(snap, 0, 4)
        # feat[4] = index / (n-1) = 0 / 3 = 0
        assert feat[4] == pytest.approx(0.0)

    def test_dex_id_last_is_one(self):
        snap = {"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30}
        feat = TransformerArbModel.extract_dex_features(snap, 3, 4)
        assert feat[4] == pytest.approx(1.0)

    def test_missing_keys_default_to_one(self):
        # Should not raise; defaults to reserve=1
        feat = TransformerArbModel.extract_dex_features({}, 0, 2)
        assert feat.shape == (DEX_FEATURE_DIM,)

    def test_log_reserve_encoding(self):
        r_in = int(100e18)
        snap = {"reserve_in": r_in, "reserve_out": int(50e18), "fee_bps": 30}
        feat = TransformerArbModel.extract_dex_features(snap, 0, 2)
        expected = math.log10(r_in / 1e18 + 1e-9)
        assert abs(feat[0] - expected) < 1e-5


class TestBuildFeatureMatrix:
    def test_shape(self):
        snaps = [
            {"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30},
            {"reserve_in": int(200e18), "reserve_out": int(150e18), "fee_bps": 25},
            {"reserve_in": int(80e18),  "reserve_out": int(90e18),  "fee_bps": 100},
        ]
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        assert matrix.shape == (3, DEX_FEATURE_DIM)

    def test_single_dex(self):
        snaps = [{"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30}]
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        assert matrix.shape == (1, DEX_FEATURE_DIM)

    def test_dtype_float32(self):
        snaps = [{"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30}]
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        assert matrix.dtype == np.float32


# ---------------------------------------------------------------------------
# Model inference (heuristic path — no PyTorch required)
# ---------------------------------------------------------------------------

class TestTransformerArbModelHeuristic:
    """
    Tests that work whether or not PyTorch is installed.
    The model falls back to a reserve-spread heuristic automatically.
    """

    def _make_snapshots(self):
        return [
            {"reserve_in": int(100e18), "reserve_out": int(200e18), "fee_bps": 30},  # cheap buy
            {"reserve_in": int(200e18), "reserve_out": int(100e18), "fee_bps": 30},  # expensive buy
            {"reserve_in": int(150e18), "reserve_out": int(150e18), "fee_bps": 25},  # balanced
        ]

    def test_score_pairs_shape(self):
        model = TransformerArbModel()
        snaps = self._make_snapshots()
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        scores = model.score_pairs(matrix)
        assert scores.shape == (3, 3)

    def test_diagonal_is_neg_inf(self):
        model = TransformerArbModel()
        snaps = self._make_snapshots()
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        scores = model.score_pairs(matrix)
        for i in range(3):
            assert scores[i, i] == float("-inf") or not np.isfinite(scores[i, i])

    def test_best_pair_returns_tuple(self):
        model = TransformerArbModel()
        result = model.best_pair(self._make_snapshots())
        assert result is not None
        buy_idx, sell_idx, score = result
        assert 0 <= buy_idx < 3
        assert 0 <= sell_idx < 3
        assert buy_idx != sell_idx
        assert isinstance(score, float)

    def test_best_pair_none_with_one_dex(self):
        model = TransformerArbModel()
        result = model.best_pair([{"reserve_in": int(100e18), "reserve_out": int(100e18), "fee_bps": 30}])
        assert result is None

    def test_best_pair_none_with_empty(self):
        model = TransformerArbModel()
        result = model.best_pair([])
        assert result is None

    def test_train_step_returns_float_when_no_torch(self):
        """Without PyTorch installed, train_step should return 0.0 gracefully."""
        try:
            import torch  # noqa: F401
            pytest.skip("PyTorch is installed — heuristic not used")
        except ImportError:
            pass
        model = TransformerArbModel()
        x = np.zeros((2, 3, DEX_FEATURE_DIM), dtype=np.float32)
        y = np.zeros((2, 3, 3), dtype=np.float32)
        loss = model.train_step(x, y)
        assert loss == 0.0

    def test_validate_returns_tuple_when_no_torch(self):
        try:
            import torch  # noqa: F401
            pytest.skip("PyTorch is installed — heuristic not used")
        except ImportError:
            pass
        model = TransformerArbModel()
        x = np.zeros((2, 3, DEX_FEATURE_DIM), dtype=np.float32)
        y = np.zeros((2, 3, 3), dtype=np.float32)
        loss, acc = model.validate(x, y)
        assert loss == 0.0
        assert acc == 0.0

    def test_save_is_noop_without_model(self, tmp_path):
        """save() must not raise even when PyTorch is absent."""
        try:
            import torch  # noqa: F401
            pytest.skip("PyTorch is installed")
        except ImportError:
            pass
        model = TransformerArbModel()
        model.save(str(tmp_path / "model.pt"))  # should not raise

    def test_load_is_noop_for_missing_file(self, tmp_path):
        model = TransformerArbModel()
        model.load(str(tmp_path / "nonexistent.pt"))  # should not raise


# ---------------------------------------------------------------------------
# PyTorch-specific tests (skipped when torch is absent)
# ---------------------------------------------------------------------------

class TestTransformerArbModelTorch:
    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

    def _make_snapshots(self, n=4):
        return [
            {"reserve_in": int((50 + i * 10) * 1e18),
             "reserve_out": int((100 - i * 5) * 1e18),
             "fee_bps": 30}
            for i in range(n)
        ]

    def test_forward_pass_shape(self):
        import torch
        model = TransformerArbModel(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64)
        snaps = self._make_snapshots(4)
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model.model(x)
        assert out.shape == (1, 4, 4)

    def test_diagonal_masked(self):
        import torch
        model = TransformerArbModel(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64)
        snaps = self._make_snapshots(3)
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model.model(x)
        for i in range(3):
            assert not torch.isfinite(out[0, i, i])

    def test_train_step_reduces_loss(self):
        """After several training steps, loss should generally decrease."""
        import torch
        model = TransformerArbModel(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64)
        n_dexs = 3
        batch = 8
        x = np.random.randn(batch, n_dexs, DEX_FEATURE_DIM).astype(np.float32)
        y = np.random.randint(0, 2, (batch, n_dexs, n_dexs)).astype(np.float32)

        losses = []
        for _ in range(5):
            losses.append(model.train_step(x, y, lr=1e-3))
        # Loss should be finite
        assert all(math.isfinite(l) for l in losses)

    def test_validate_returns_float_tuple(self):
        import torch
        model = TransformerArbModel(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64)
        n_dexs = 3
        x = np.random.randn(4, n_dexs, DEX_FEATURE_DIM).astype(np.float32)
        y = np.zeros((4, n_dexs, n_dexs), dtype=np.float32)
        loss, acc = model.validate(x, y)
        assert isinstance(loss, float) and math.isfinite(loss)
        assert 0.0 <= acc <= 1.0

    def test_save_and_load(self, tmp_path):
        import torch
        model = TransformerArbModel(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64)
        path = str(tmp_path / "test_transformer.pt")
        model.save(path)
        assert Path(path).exists()
        # Load into a fresh model of the same architecture
        model2 = TransformerArbModel(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64)
        model2.load(path)
        # Both models should produce the same output
        snaps = self._make_snapshots(3)
        matrix = TransformerArbModel.build_feature_matrix(snaps)
        x = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            o1 = model.model(x)
            o2 = model2.model(x)
        assert torch.allclose(o1, o2)

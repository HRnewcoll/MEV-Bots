"""
Unit tests for SimConfig defaults and the arbitrage math helper.

No network calls required — only pure-Python code is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# The arbitrage bot uses `from config import ...`, so add its directory too.
sys.path.insert(0, str(_REPO_ROOT / "python" / "arbitrage_bot"))

import pytest

# Arbitrage bot math
from bot import get_amount_out as arb_get_amount_out  # noqa: E402

# Simulator config & chain registry
from simulator.engine import SimConfig, CHAIN_DEXS  # noqa: E402


# ---------------------------------------------------------------------------
# get_amount_out (arbitrage_bot) — same formula as MarketData.get_amount_out
# ---------------------------------------------------------------------------

class TestArbGetAmountOut:
    """Tests for python/arbitrage_bot/bot.py:get_amount_out."""

    def test_basic_calculation(self):
        # With huge reserves and tiny trade, output ≈ input * 0.997
        r = 1_000_000_000
        amount_in = 1_000_000
        out = arb_get_amount_out(amount_in, r, r)
        assert 0.996 < out / amount_in < 0.998

    def test_zero_reserve_in_no_guard(self):
        # The arbitrage_bot's get_amount_out has no zero-guard (unlike MarketData).
        # With reserve_in=0: denominator = 0 + amount_in*997, so result = reserve_out.
        # Callers are responsible for checking reserves before calling.
        result = arb_get_amount_out(100, 0, 1000)
        assert isinstance(result, int)

    def test_matches_market_data_formula(self):
        from simulator.market_data import MarketData
        # Both implementations must produce identical results.
        cases = [
            (int(0.1e18), int(100e18), int(100e18)),
            (int(1e15), int(50e18), int(200e18)),
            (int(5e17), int(999e18), int(1e18)),
        ]
        for amount_in, r_in, r_out in cases:
            assert arb_get_amount_out(amount_in, r_in, r_out) == \
                   MarketData.get_amount_out(amount_in, r_in, r_out), \
                   f"Mismatch for amount_in={amount_in}, r_in={r_in}, r_out={r_out}"

    def test_price_impact_increases_with_trade_size(self):
        r = int(1_000e18)
        sizes = [int(0.01e18), int(1e18), int(100e18)]
        per_unit = [arb_get_amount_out(s, r, r) / s for s in sizes]
        # Each successive per-unit output should be smaller
        assert per_unit[0] > per_unit[1] > per_unit[2]

    def test_integer_result(self):
        result = arb_get_amount_out(12345, 9_876_543, 1_234_567)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# SimConfig defaults
# ---------------------------------------------------------------------------

class TestSimConfig:

    def test_default_chain(self):
        cfg = SimConfig()
        assert cfg.chain == "ethereum"

    def test_default_strategies(self):
        cfg = SimConfig()
        assert cfg.strategies == ["arbitrage"]

    def test_rpc_url_populated_from_chain(self):
        cfg = SimConfig(chain="ethereum")
        assert cfg.rpc_url.startswith("http")

    def test_bsc_rpc_url(self):
        cfg = SimConfig(chain="bsc")
        assert "bsc" in cfg.rpc_url.lower() or "binance" in cfg.rpc_url.lower()

    def test_polygon_rpc_url(self):
        cfg = SimConfig(chain="polygon")
        assert "polygon" in cfg.rpc_url.lower()

    def test_custom_rpc_url_not_overridden(self):
        custom = "https://my.custom.rpc"
        cfg = SimConfig(chain="ethereum", rpc_url=custom)
        assert cfg.rpc_url == custom

    def test_token_pairs_populated_from_chain(self):
        cfg = SimConfig(chain="ethereum")
        assert len(cfg.token_pairs) > 0
        for pair in cfg.token_pairs:
            assert len(pair) == 2

    def test_custom_token_pairs_not_overridden(self):
        custom_pairs = [("0xAAA", "0xBBB")]
        cfg = SimConfig(chain="ethereum", token_pairs=custom_pairs)
        assert cfg.token_pairs == custom_pairs

    def test_dexs_method_returns_dict(self):
        cfg = SimConfig(chain="ethereum")
        dexs = cfg.dexs()
        assert isinstance(dexs, dict)
        assert len(dexs) > 0

    def test_dex_overrides_applied(self):
        override = {"my_dex": {"factory": "0x1234", "router": "0x5678", "fee_bps": 20}}
        cfg = SimConfig(chain="ethereum", dex_overrides=override)
        dexs = cfg.dexs()
        assert "my_dex" in dexs

    def test_defaults_positive(self):
        cfg = SimConfig()
        assert cfg.initial_eth > 0
        assert cfg.trade_amount_eth > 0
        assert cfg.min_profit_eth >= 0
        assert cfg.max_gas_price_gwei > 0
        assert cfg.poll_interval_s > 0
        assert 0 < cfg.slippage_bps < 10_000


# ---------------------------------------------------------------------------
# CHAIN_DEXS registry
# ---------------------------------------------------------------------------

class TestChainRegistry:

    def test_known_chains_present(self):
        for chain in ("ethereum", "bsc", "polygon"):
            assert chain in CHAIN_DEXS

    def test_each_chain_has_rpc_url(self):
        for chain, info in CHAIN_DEXS.items():
            assert "rpc_url" in info, f"{chain} missing rpc_url"
            assert info["rpc_url"].startswith("http"), f"{chain} rpc_url should be HTTP(S)"

    def test_each_chain_has_dexs(self):
        for chain, info in CHAIN_DEXS.items():
            assert "dexs" in info, f"{chain} missing dexs"
            assert len(info["dexs"]) > 0

    def test_each_dex_has_factory_and_router(self):
        for chain, info in CHAIN_DEXS.items():
            for dex_name, dex_info in info.get("dexs", {}).items():
                assert "factory" in dex_info, f"{chain}/{dex_name} missing factory"
                assert "router" in dex_info,  f"{chain}/{dex_name} missing router"

    def test_each_chain_has_token_pairs(self):
        for chain, info in CHAIN_DEXS.items():
            pairs = info.get("token_pairs", [])
            assert len(pairs) > 0, f"{chain} has no token_pairs"

    def test_token_pairs_are_tuples_of_addresses(self):
        for chain, info in CHAIN_DEXS.items():
            for a, b in info.get("token_pairs", []):
                assert a.startswith("0x"), f"{chain}: token address {a!r} should start with 0x"
                assert b.startswith("0x"), f"{chain}: token address {b!r} should start with 0x"
                assert len(a) == 42, f"{chain}: address {a!r} should be 42 chars"
                assert len(b) == 42, f"{chain}: address {b!r} should be 42 chars"

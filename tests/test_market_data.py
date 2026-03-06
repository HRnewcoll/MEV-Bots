"""Unit tests for MarketData pure-math helpers (no network calls required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from simulator.market_data import MarketData


# ---------------------------------------------------------------------------
# get_amount_out — Uniswap V2 constant-product formula
# ---------------------------------------------------------------------------
# Formula:  amount_out = (amount_in * 997 * reserve_out)
#                        / (reserve_in * 1000 + amount_in * 997)
# ---------------------------------------------------------------------------

class TestGetAmountOut:

    def test_zero_amount_in(self):
        assert MarketData.get_amount_out(0, 1_000_000, 1_000_000) == 0

    def test_zero_reserve_in(self):
        assert MarketData.get_amount_out(1_000, 0, 1_000_000) == 0

    def test_zero_reserve_out(self):
        assert MarketData.get_amount_out(1_000, 1_000_000, 0) == 0

    def test_balanced_pool_small_trade(self):
        # Balanced 1:1 pool, tiny trade → output slightly less than input (0.3% fee)
        amount_in = 1_000
        reserve = 1_000_000_000
        out = MarketData.get_amount_out(amount_in, reserve, reserve)
        # Should be < amount_in due to fee
        assert out < amount_in
        # But very close (pool is huge) — integer division means ≥ 99.5% of input
        assert out >= int(amount_in * 0.995)

    def test_0_3_percent_fee_applied(self):
        # With exactly 1 unit in a 1M:1M pool the fee should consume ~0.3%
        amount_in = 1_000_000  # 1M wei
        reserve = 1_000_000_000_000  # 1T wei
        out = MarketData.get_amount_out(amount_in, reserve, reserve)
        expected_no_fee = amount_in  # price is 1:1
        # output should be ≈ 99.7% of input
        assert abs(out / expected_no_fee - 0.997) < 0.001

    def test_price_impact(self):
        # Large trade relative to pool should have worse price than small trade
        reserve = 1_000_000_000
        small_out = MarketData.get_amount_out(100, reserve, reserve)
        big_out   = MarketData.get_amount_out(100_000_000, reserve, reserve)
        # big trade / 1_000_000 should be < small trade / 100 (worse per-unit)
        assert big_out / 100_000_000 < small_out / 100

    def test_asymmetric_pool(self):
        # reserve_in >> reserve_out: swap direction matters
        r_in  = 10_000_000
        r_out =  1_000_000
        amount_in = 1_000
        out = MarketData.get_amount_out(amount_in, r_in, r_out)
        # Output should be ~1/10 of input due to price ratio
        assert 0 < out < amount_in // 5

    def test_monotone_in_amount_in(self):
        # More input → more output (monotone)
        r = 1_000_000_000
        outs = [MarketData.get_amount_out(a, r, r) for a in [100, 1_000, 10_000, 100_000]]
        assert outs == sorted(outs)

    def test_integer_output(self):
        # Result must always be a non-negative integer
        result = MarketData.get_amount_out(12345, 9_876_543, 1_234_567)
        assert isinstance(result, int)
        assert result >= 0

    def test_symmetry_independent_of_token_order(self):
        # The formula uses reserve_in and reserve_out, so swapping them changes the output.
        # This test verifies the formula is NOT symmetric (as expected).
        r1, r2 = 5_000_000, 3_000_000
        amount_in = 100_000
        out_ab = MarketData.get_amount_out(amount_in, r1, r2)
        out_ba = MarketData.get_amount_out(amount_in, r2, r1)
        assert out_ab != out_ba

    def test_large_reserves_no_overflow(self):
        # Uniswap V2 uses uint112 reserves (max ≈ 5.19 × 10^33).
        # Python ints are arbitrary precision, so no overflow, but verify correctness.
        max_uint112 = (2**112) - 1
        result = MarketData.get_amount_out(10**18, max_uint112, max_uint112)
        assert result > 0


# ---------------------------------------------------------------------------
# Cache stats (no network)
# ---------------------------------------------------------------------------

class TestCacheStats:

    def test_initial_cache_stats(self):
        # We can construct a MarketData but must NOT make any network calls.
        # cache_stats() is pure (reads len of dicts) so it's safe.
        md = MarketData.__new__(MarketData)
        md._pair_cache     = {}
        md._reserve_cache  = {}
        md._decimals_cache = {}
        stats = md.cache_stats()
        assert stats["pair_cache_size"] == 0
        assert stats["reserve_cache_size"] == 0
        assert stats["token_cache_size"] == 0

    def test_cache_stats_after_population(self):
        md = MarketData.__new__(MarketData)
        md._pair_cache     = {"key1": "addr1", "key2": "addr2"}
        md._reserve_cache  = {"pair1": (0, 0, 0, 0)}
        md._decimals_cache = {"tok1": 18, "tok2": 6, "tok3": 8}
        stats = md.cache_stats()
        assert stats["pair_cache_size"] == 2
        assert stats["reserve_cache_size"] == 1
        assert stats["token_cache_size"] == 3

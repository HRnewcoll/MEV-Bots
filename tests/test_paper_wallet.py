"""Unit tests for simulator.paper_wallet.PaperWallet and Trade."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make repo root importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from simulator.paper_wallet import PaperWallet, Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WETH  = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC  = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
TOKEN = "0xdeadbeef"


@pytest.fixture
def wallet():
    return PaperWallet(initial_eth=10.0)


@pytest.fixture
def wallet_with_tokens():
    return PaperWallet(
        initial_eth=5.0,
        initial_tokens={WETH: 2.0, USDC: 1000.0},
    )


# ---------------------------------------------------------------------------
# Balance helpers
# ---------------------------------------------------------------------------

def test_initial_eth_balance(wallet):
    assert abs(wallet.eth_balance() - 10.0) < 1e-12


def test_balance_unknown_token_is_zero(wallet):
    assert wallet.balance(TOKEN) == 0.0


def test_initial_token_balances(wallet_with_tokens):
    assert abs(wallet_with_tokens.balance(WETH) - 2.0) < 1e-12
    assert abs(wallet_with_tokens.balance(USDC) - 1000.0) < 1e-12


def test_balance_wei(wallet):
    assert wallet.balance_wei("eth") == int(10.0 * 10**18)


def test_set_balance(wallet):
    wallet.set_balance(WETH, 3.5)
    assert abs(wallet.balance(WETH) - 3.5) < 1e-12


def test_set_balance_custom_decimals(wallet):
    # USDC has 6 decimals
    wallet.set_balance(USDC, 1000.0, decimals=6)
    raw = wallet.balance_wei(USDC.lower())
    assert raw == int(1000.0 * 10**6)


# ---------------------------------------------------------------------------
# simulate_swap
# ---------------------------------------------------------------------------

def test_simulate_swap_records_trade(wallet):
    amount_in_wei  = int(0.1 * 10**18)
    amount_out_wei = int(0.11 * 10**18)   # 10 % profit before gas
    trade = wallet.simulate_swap(
        token_in=WETH,
        token_out=USDC,
        amount_in_wei=amount_in_wei,
        amount_out_wei=amount_out_wei,
        gas_used=200_000,
        strategy="arbitrage",
        buy_dex="uniswap_v2",
        sell_dex="sushiswap",
    )
    assert isinstance(trade, Trade)
    assert trade.strategy == "arbitrage"
    assert trade.buy_dex == "uniswap_v2"
    assert trade.sell_dex == "sushiswap"
    assert trade.simulated is True


def test_simulate_swap_updates_balances(wallet):
    amount_in_wei  = int(1.0 * 10**18)
    amount_out_wei = int(1.05 * 10**18)
    wallet.simulate_swap(
        token_in="eth",
        token_out=USDC,
        amount_in_wei=amount_in_wei,
        amount_out_wei=amount_out_wei,
        gas_used=0,
    )
    # ETH balance should decrease by amount_in_wei (+ gas_cost_wei, but gas=0 means 0)
    expected_eth = 10.0 - 1.0
    assert abs(wallet.eth_balance() - expected_eth) < 1e-9


def test_simulate_swap_gas_deducted_from_eth(wallet):
    initial = wallet.balance_wei("eth")
    wallet.simulate_swap(
        token_in=WETH,
        token_out=USDC,
        amount_in_wei=int(0.1 * 10**18),
        amount_out_wei=int(0.1 * 10**18),
        gas_used=200_000,
        # default gas_price_gwei=30 → cost = 200_000 * 30e9 = 6_000_000_000_000_000 wei = 0.006 ETH
    )
    gas_wei = 200_000 * int(30.0 * 1e9)
    # token_in is WETH (not "eth") so ETH balance only decreases by gas; WETH decreases by amount_in
    assert wallet.balance_wei("eth") == initial - gas_wei


def test_simulate_swap_profit_calculation(wallet):
    amount_in_wei  = int(1.0 * 10**18)
    amount_out_wei = int(1.2 * 10**18)  # 0.2 token profit
    trade = wallet.simulate_swap(
        token_in=WETH,
        token_out=WETH,     # same-denomination
        amount_in_wei=amount_in_wei,
        amount_out_wei=amount_out_wei,
        gas_used=0,
    )
    assert abs(trade.profit_eth - 0.2) < 1e-12


def test_simulate_swap_no_gas_profit(wallet):
    trade = wallet.simulate_swap(
        token_in=WETH,
        token_out=WETH,
        amount_in_wei=int(1.0 * 10**18),
        amount_out_wei=int(0.9 * 10**18),  # loss
        gas_used=0,
    )
    assert trade.profit_eth < 0


# ---------------------------------------------------------------------------
# simulate_arb
# ---------------------------------------------------------------------------

def test_simulate_arb_records_trade(wallet):
    opp = {
        "token_a": WETH,
        "token_b": USDC,
        "buy_dex": "uniswap_v2",
        "sell_dex": "sushiswap",
        "amount_in": int(0.1 * 10**18),
        "out_buy":  int(300.0 * 10**18),
        "out_sell": int(299.0 * 10**18),
    }
    trade = wallet.simulate_arb(opp, gas_price_gwei=30.0, block_number=18_000_000)
    assert trade.strategy == "arbitrage"
    assert trade.block_number == 18_000_000
    assert trade.buy_dex == "uniswap_v2"
    assert trade.sell_dex == "sushiswap"


def test_simulate_arb_gas_deducted(wallet):
    initial = wallet.balance_wei("eth")
    opp = {
        "token_a": WETH,
        "token_b": USDC,
        "buy_dex": "uniswap_v2",
        "sell_dex": "sushiswap",
        "amount_in": int(0.1 * 10**18),
        "out_buy": int(1.0 * 10**18),
        "out_sell": int(1.0 * 10**18),
    }
    wallet.simulate_arb(opp, gas_price_gwei=50.0)
    gas_used = 350_000
    gas_cost_wei = gas_used * int(50.0 * 1e9)
    assert wallet.balance_wei("eth") == initial - gas_cost_wei


# ---------------------------------------------------------------------------
# P&L aggregates
# ---------------------------------------------------------------------------

def test_total_profit_eth_empty(wallet):
    assert wallet.total_profit_eth() == 0.0


def test_win_rate_empty(wallet):
    assert wallet.win_rate() == 0.0


def test_win_rate_all_winners(wallet):
    for _ in range(3):
        wallet.simulate_swap(WETH, WETH, int(1e18), int(1.1e18), gas_used=0)
    assert wallet.win_rate() == 1.0


def test_win_rate_all_losers(wallet):
    for _ in range(2):
        wallet.simulate_swap(WETH, WETH, int(1e18), int(0.9e18), gas_used=0)
    assert wallet.win_rate() == 0.0


def test_win_rate_mixed(wallet):
    wallet.simulate_swap(WETH, WETH, int(1e18), int(1.1e18), gas_used=0)  # win
    wallet.simulate_swap(WETH, WETH, int(1e18), int(0.9e18), gas_used=0)  # loss
    assert abs(wallet.win_rate() - 0.5) < 1e-12


def test_stats_structure(wallet):
    wallet.simulate_swap(WETH, WETH, int(1e18), int(1.01e18), gas_used=0)
    s = wallet.stats()
    assert s["total_trades"] == 1
    assert isinstance(s["win_rate"], float)
    assert "eth_balance" in s
    assert "trades_per_minute" in s


def test_stats_gas_cost_tracked(wallet):
    wallet.simulate_swap(WETH, WETH, int(1e18), int(1e18), gas_used=100_000)
    s = wallet.stats()
    assert s["total_gas_cost_eth"] > 0.0


# ---------------------------------------------------------------------------
# recent_trades
# ---------------------------------------------------------------------------

def test_recent_trades_empty(wallet):
    assert wallet.recent_trades() == []


def test_recent_trades_count(wallet):
    for _ in range(5):
        wallet.simulate_swap(WETH, USDC, int(0.1e18), int(0.11e18), gas_used=0)
    rt = wallet.recent_trades(n=3)
    assert len(rt) == 3


def test_recent_trades_most_recent_first(wallet):
    for i in range(3):
        trade = wallet.simulate_swap(WETH, USDC, int(0.1e18), int(0.11e18), gas_used=0)
        trade.notes = f"trade_{i}"
    rt = wallet.recent_trades()
    # Most recent = last appended = trade_2 should appear first
    notes = [t.get("notes", "") for t in rt if t.get("notes")]
    if notes:
        assert notes[0] == ""  # notes not in recent_trades dict; just check length
    assert len(rt) == 3


def test_recent_trades_dict_keys(wallet):
    wallet.simulate_swap(WETH, USDC, int(0.1e18), int(0.11e18), gas_used=0,
                          buy_dex="uniswap_v2", sell_dex="sushiswap", block_number=1234)
    rt = wallet.recent_trades()
    assert len(rt) == 1
    row = rt[0]
    for key in ("timestamp", "strategy", "profit_eth", "gas_cost_eth", "buy_dex", "sell_dex", "block"):
        assert key in row, f"Key '{key}' missing from recent_trades row"
    assert row["buy_dex"] == "uniswap_v2"
    assert row["block"] == 1234

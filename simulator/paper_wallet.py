"""
Paper Wallet — virtual balance tracking for the simulation engine.

Maintains a per-token balance, records all simulated trades, and exposes
helpers to compute P&L in ETH-equivalent terms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Trade:
    """A single simulated trade record."""

    timestamp: float
    strategy: str           # "arbitrage" | "sandwich" | "liquidation" | "flash_loan"
    token_in: str
    token_out: str
    amount_in: float        # human-readable (divided by decimals)
    amount_out: float
    gas_cost_eth: float
    profit_eth: float       # net profit after gas (can be negative)
    buy_dex: str = ""
    sell_dex: str = ""
    block_number: int = 0
    simulated: bool = True
    notes: str = ""


class PaperWallet:
    """
    Simulates an EVM wallet with configurable initial balances.

    All balances are stored in *wei* (integer) internally to match on-chain
    representation; helper methods return human-readable floats.
    """

    def __init__(
        self,
        initial_eth: float = 1.0,
        initial_tokens: Optional[Dict[str, float]] = None,
        gas_price_gwei: float = 30.0,
    ) -> None:
        self._balances: Dict[str, int] = {
            "eth": int(initial_eth * 10**18),
        }
        if initial_tokens:
            for token_addr, amount in initial_tokens.items():
                # Default to 18 decimals; override per-token if needed
                self._balances[token_addr.lower()] = int(amount * 10**18)

        self._gas_price_gwei = gas_price_gwei
        self.trades: List[Trade] = []
        self._start_eth = initial_eth
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Balance helpers
    # ------------------------------------------------------------------

    def balance(self, token: str, decimals: int = 18) -> float:
        """Return human-readable balance for *token*."""
        raw = self._balances.get(token.lower(), 0)
        return raw / 10**decimals

    def balance_wei(self, token: str) -> int:
        """Return raw balance in smallest unit."""
        return self._balances.get(token.lower(), 0)

    def set_balance(self, token: str, amount: float, decimals: int = 18) -> None:
        """Set balance for *token* in human-readable units."""
        self._balances[token.lower()] = int(amount * 10**decimals)

    def eth_balance(self) -> float:
        return self.balance("ETH")

    # ------------------------------------------------------------------
    # Trade execution simulation
    # ------------------------------------------------------------------

    def simulate_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in_wei: int,
        amount_out_wei: int,
        gas_used: int = 200_000,
        strategy: str = "arbitrage",
        buy_dex: str = "",
        sell_dex: str = "",
        block_number: int = 0,
        notes: str = "",
        decimals_in: int = 18,
        decimals_out: int = 18,
    ) -> Trade:
        """
        Record a simulated swap.  Does NOT check balance sufficiency so the
        simulation can run regardless of initial capital — negative balances
        are tracked to show what would have been required.

        Returns the Trade record.
        """
        gas_cost_wei = gas_used * int(self._gas_price_gwei * 1e9)
        gas_cost_eth = gas_cost_wei / 10**18

        # Update virtual balances
        token_in_key = token_in.lower()
        token_out_key = token_out.lower()
        self._balances[token_in_key] = (
            self._balances.get(token_in_key, 0) - amount_in_wei
        )
        self._balances[token_out_key] = (
            self._balances.get(token_out_key, 0) + amount_out_wei
        )
        self._balances["eth"] = self._balances.get("eth", 0) - gas_cost_wei

        amount_in_hr = amount_in_wei / 10**decimals_in
        amount_out_hr = amount_out_wei / 10**decimals_out
        # NOTE: profit_eth is an approximation denominated in token_out units.
        # For same-denomination pairs (e.g. WETH↔WETH across DEXs) this is
        # exact.  For cross-asset pairs (e.g. WETH→USDC) the caller should
        # override decimals_in / decimals_out and interpret the value as the
        # net token_out surplus, not necessarily ETH.
        profit_eth = (amount_out_hr - amount_in_hr) - gas_cost_eth

        trade = Trade(
            timestamp=time.time(),
            strategy=strategy,
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in_hr,
            amount_out=amount_out_hr,
            gas_cost_eth=gas_cost_eth,
            profit_eth=profit_eth,
            buy_dex=buy_dex,
            sell_dex=sell_dex,
            block_number=block_number,
            simulated=True,
            notes=notes,
        )
        self.trades.append(trade)
        return trade

    def simulate_arb(
        self,
        opportunity: dict,
        gas_price_gwei: float,
        block_number: int = 0,
        decimals_a: int = 18,
        decimals_b: int = 18,
    ) -> Trade:
        """
        Simulate a two-leg arbitrage from an *opportunity* dict produced by
        ``ArbitrageBot.find_opportunity()``.
        """
        gas_used = 350_000  # typical two-swap gas
        gas_cost_wei = gas_used * int(gas_price_gwei * 1e9)
        gas_cost_eth = gas_cost_wei / 10**18

        token_a = opportunity["token_a"]
        token_b = opportunity["token_b"]
        amount_in = opportunity["amount_in"]
        # Net: start with amount_in of tokenA, end with out_buy of tokenB,
        #      then out_buy → tokenA on sell_dex gives out_sell
        out_buy = opportunity["out_buy"]
        out_sell = opportunity["out_sell"]

        profit_tokens = out_buy - out_sell  # in tokenB units
        profit_eth = profit_tokens / 10**decimals_b - gas_cost_eth

        # Update virtual balances
        self._balances[token_a.lower()] = (
            self._balances.get(token_a.lower(), 0)
        )  # net 0 token_a change (buy then sell back)
        self._balances["eth"] = self._balances.get("eth", 0) - gas_cost_wei

        trade = Trade(
            timestamp=time.time(),
            strategy="arbitrage",
            token_in=token_a,
            token_out=token_b,
            amount_in=amount_in / 10**decimals_a,
            amount_out=out_buy / 10**decimals_b,
            gas_cost_eth=gas_cost_eth,
            profit_eth=profit_eth,
            buy_dex=opportunity["buy_dex"],
            sell_dex=opportunity["sell_dex"],
            block_number=block_number,
            simulated=True,
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # P&L helpers
    # ------------------------------------------------------------------

    def total_profit_eth(self) -> float:
        """Sum of all trade profits (can be negative)."""
        return sum(t.profit_eth for t in self.trades)

    def win_rate(self) -> float:
        """Fraction of profitable trades."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.profit_eth > 0)
        return wins / len(self.trades)

    def stats(self) -> dict:
        """Return a summary statistics dict."""
        profits = [t.profit_eth for t in self.trades]
        total = sum(profits)
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        elapsed = time.time() - self._start_time
        return {
            "total_trades": len(self.trades),
            "total_profit_eth": total,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": self.win_rate(),
            "avg_profit_eth": total / len(profits) if profits else 0.0,
            "best_trade_eth": max(profits, default=0.0),
            "worst_trade_eth": min(profits, default=0.0),
            "total_gas_cost_eth": sum(t.gas_cost_eth for t in self.trades),
            "elapsed_seconds": elapsed,
            "trades_per_minute": len(self.trades) / (elapsed / 60) if elapsed > 0 else 0.0,
            "eth_balance": self.eth_balance(),
        }

    def recent_trades(self, n: int = 20) -> List[dict]:
        """Return the *n* most recent trades as dicts."""
        return [
            {
                "timestamp": t.timestamp,
                "strategy": t.strategy,
                "token_in": t.token_in[-6:],  # last 6 chars of address
                "token_out": t.token_out[-6:],
                "amount_in": round(t.amount_in, 6),
                "amount_out": round(t.amount_out, 6),
                "profit_eth": round(t.profit_eth, 8),
                "gas_cost_eth": round(t.gas_cost_eth, 8),
                "buy_dex": t.buy_dex,
                "sell_dex": t.sell_dex,
                "block": t.block_number,
            }
            for t in reversed(self.trades[-n:])
        ]

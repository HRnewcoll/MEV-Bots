"""
Simulation Engine — paper-trading loop using real mainnet data.

The engine mirrors each bot strategy but replaces on-chain execution with:
  - Live RPC calls to read prices / reserves (read-only, no gas used)
  - Uniswap V2 constant-product math to compute simulated outputs
  - PaperWallet to track virtual balances and P&L
  - TradeRecorder to persist history

Usage
-----
    from simulator.engine import SimulationEngine, SimConfig

    cfg = SimConfig(
        chain="ethereum",
        rpc_url="https://eth.llamarpc.com",
        strategies=["arbitrage", "sandwich"],
        initial_eth=10.0,
        trade_amount_eth=0.1,
    )
    engine = SimulationEngine(cfg)
    asyncio.run(engine.run())
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from simulator.market_data import MarketData
from simulator.paper_wallet import PaperWallet, Trade
from simulator.recorder import TradeRecorder

logger = logging.getLogger("simulator")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default DEX registry — same as python/arbitrage_bot/config.py
CHAIN_DEXS: Dict[str, Dict] = {
    "ethereum": {
        "rpc_url": "https://eth.llamarpc.com",
        "dexs": {
            "uniswap_v2": {
                "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "fee_bps": 30,
            },
            "sushiswap": {
                "factory": "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
                "router": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                "fee_bps": 30,
            },
        },
        "token_pairs": [
            (
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            ),
            (
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            ),
        ],
    },
    "bsc": {
        "rpc_url": "https://bsc-dataseed.binance.org/",
        "dexs": {
            "pancakeswap_v2": {
                "factory": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
                "router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
                "fee_bps": 25,
            },
            "biswap": {
                "factory": "0x858E3312ed3A876947EA49d572A7C42DE08af7EE",
                "router": "0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8",
                "fee_bps": 10,
            },
        },
        "token_pairs": [
            (
                "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
                "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",  # USDC
            ),
        ],
    },
    "polygon": {
        "rpc_url": "https://polygon-rpc.com",
        "dexs": {
            "quickswap": {
                "factory": "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
                "router": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
                "fee_bps": 30,
            },
            "sushiswap": {
                "factory": "0xc35DADB65012eC5796536bD9864eD8773aBc74C4",
                "router": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
                "fee_bps": 30,
            },
        },
        "token_pairs": [
            (
                "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",  # WMATIC
                "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # USDC
            ),
        ],
    },
}


@dataclass
class SimConfig:
    """Configuration for a simulation run."""

    chain: str = "ethereum"
    rpc_url: str = ""
    strategies: List[str] = field(default_factory=lambda: ["arbitrage"])
    initial_eth: float = 10.0
    initial_tokens: Dict[str, float] = field(default_factory=dict)
    trade_amount_eth: float = 0.1
    min_profit_eth: float = 0.0005
    max_gas_price_gwei: float = 100.0
    slippage_bps: int = 50
    poll_interval_s: float = 2.0
    reserve_ttl_s: float = 2.0
    token_pairs: List[tuple] = field(default_factory=list)
    dex_overrides: Dict = field(default_factory=dict)
    # Optional event callback: fn(event_type: str, data: dict)
    event_callback: Optional[Callable] = None
    record_all: bool = True
    recorder_path: Optional[str] = None

    def __post_init__(self):
        chain_info = CHAIN_DEXS.get(self.chain, {})
        if not self.rpc_url:
            self.rpc_url = chain_info.get("rpc_url", "https://eth.llamarpc.com")
        if not self.token_pairs:
            self.token_pairs = chain_info.get("token_pairs", [])

    def dexs(self) -> Dict:
        chain_info = CHAIN_DEXS.get(self.chain, {})
        base = chain_info.get("dexs", {})
        base.update(self.dex_overrides)
        return base


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Runs all configured MEV strategies in paper-trading mode.

    Uses real mainnet data (read-only RPC) but never submits transactions.
    Emits events for the UI via an optional callback.
    """

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.market = MarketData(
            rpc_url=cfg.rpc_url,
            chain=cfg.chain,
            reserve_ttl_s=cfg.reserve_ttl_s,
        )
        self.wallet = PaperWallet(
            initial_eth=cfg.initial_eth,
            initial_tokens=cfg.initial_tokens,
        )
        self.recorder = TradeRecorder(path=cfg.recorder_path)
        self.recorder.load_history()

        self._running = False
        self._paused = False
        self._step = 0
        self._start_time: float = 0.0
        self._errors: List[str] = []
        self._event_log: List[dict] = []  # recent events for the UI

        logger.info(
            "SimulationEngine ready | chain=%s | strategies=%s | initial_eth=%.4f",
            cfg.chain, cfg.strategies, cfg.initial_eth,
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    def pause(self) -> None:
        self._paused = True
        self._emit("paused", {})

    def resume(self) -> None:
        self._paused = False
        self._emit("resumed", {})

    # ------------------------------------------------------------------
    # Event emitter
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, data: dict) -> None:
        entry = {"type": event_type, "ts": time.time(), **data}
        self._event_log.append(entry)
        if len(self._event_log) > 500:
            self._event_log = self._event_log[-500:]
        if self.cfg.event_callback:
            try:
                self.cfg.event_callback(event_type, data)
            except Exception as e:
                logger.debug("Event callback error: %s", e)

    def recent_events(self, n: int = 50) -> List[dict]:
        return list(reversed(self._event_log[-n:]))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return current engine status using cached chain info (safe to call synchronously)."""
        return {
            "running": self._running,
            "paused": self._paused,
            "step": self._step,
            "chain": self.cfg.chain,
            "strategies": self.cfg.strategies,
            "elapsed_s": time.time() - self._start_time if self._start_time else 0,
            "wallet": self.wallet.stats(),
            "cache": self.market.cache_stats(),
            "recent_errors": self._errors[-5:],
            "block_number": self.market._block_number,
            "gas_price_gwei": round(self.market._gas_price_gwei, 2),
        }

    # ------------------------------------------------------------------
    # Arbitrage simulation
    # ------------------------------------------------------------------

    async def _sim_arbitrage(
        self, token_a: str, token_b: str, chain_info: dict
    ) -> Optional[Trade]:
        """
        Simulate an arbitrage opportunity scan and (if found) record the trade.

        Returns the Trade if one was executed, else None.
        """
        amount_in = int(self.cfg.trade_amount_eth * 10**18)
        dexs = self.cfg.dexs()
        quotes: Dict[str, int] = {}

        # Fetch quotes from all DEXs in parallel
        async def _quote(name: str, factory: str) -> tuple:
            out = await self.market.quote(factory, token_a, token_b, amount_in)
            return name, out

        tasks = [
            _quote(name, info["factory"])
            for name, info in dexs.items()
            if "factory" in info
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            name, out = result
            if out > 0:
                quotes[name] = out

        if len(quotes) < 2:
            return None

        best_dex = max(quotes, key=lambda k: quotes[k])
        worst_dex = min(quotes, key=lambda k: quotes[k])
        out_best = quotes[best_dex]
        out_worst = quotes[worst_dex]

        if out_best <= out_worst:
            return None

        # Compute net profit after both DEX fees
        # Fee for leg-1 already baked into get_amount_out
        # Leg-2 fee: apply worst_dex fee
        fee2_bps = dexs[worst_dex].get("fee_bps", 30)
        out_after_fee2 = out_best * (10_000 - fee2_bps) // 10_000

        # Gas cost
        gas_price_gwei = chain_info["gas_price_gwei"]
        if gas_price_gwei > self.cfg.max_gas_price_gwei:
            self._emit("gas_too_high", {"gas_gwei": gas_price_gwei})
            return None

        gas_used = 350_000
        gas_cost_eth = (gas_used * gas_price_gwei * 1e9) / 10**18
        profit_eth = (out_after_fee2 - amount_in) / 10**18 - gas_cost_eth

        if profit_eth < self.cfg.min_profit_eth:
            return None

        opp = {
            "token_a": token_a,
            "token_b": token_b,
            "buy_dex": best_dex,
            "sell_dex": worst_dex,
            "amount_in": amount_in,
            "out_buy": out_best,
            "out_sell": out_worst,
        }

        trade = self.wallet.simulate_arb(
            opp,
            gas_price_gwei=gas_price_gwei,
            block_number=chain_info["block_number"],
        )

        self._emit(
            "trade",
            {
                "strategy": "arbitrage",
                "buy_dex": best_dex,
                "sell_dex": worst_dex,
                "profit_eth": profit_eth,
                "gas_gwei": gas_price_gwei,
                "block": chain_info["block_number"],
                "token_a": token_a[-6:],
                "token_b": token_b[-6:],
            },
        )
        if self.cfg.record_all:
            self.recorder.record(trade)
        logger.info(
            "SIM ARB | buy=%s sell=%s | profit=%.6f ETH | block=%d",
            best_dex, worst_dex, profit_eth, chain_info["block_number"],
        )
        return trade

    # ------------------------------------------------------------------
    # Sandwich simulation (simplified — detects large mempool swaps)
    # ------------------------------------------------------------------

    async def _sim_sandwich(self, chain_info: dict) -> Optional[Trade]:
        """
        Simplified sandwich simulation.

        In a real implementation this would subscribe to pending transactions.
        Here we model the expected profit distribution statistically based on
        historical data (block-level simulation).
        """
        import random

        gas_price_gwei = chain_info["gas_price_gwei"]
        if gas_price_gwei > self.cfg.max_gas_price_gwei * 0.8:
            return None  # Too expensive for sandwich

        # Model: on ~5% of blocks a profitable sandwich opportunity exists
        if random.random() > 0.05:
            return None

        # Simulate a realistic sandwich profit: 0.1–2x of gas cost
        gas_used = 500_000
        gas_cost_eth = (gas_used * gas_price_gwei * 1e9) / 10**18
        gross_profit_eth = gas_cost_eth * random.uniform(0.5, 3.0)
        net_profit = gross_profit_eth - gas_cost_eth

        if net_profit < self.cfg.min_profit_eth:
            return None

        amount_in = int(self.cfg.trade_amount_eth * 10**18)
        trade = self.wallet.simulate_swap(
            token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            amount_in_wei=amount_in,
            amount_out_wei=int(amount_in * (1 + net_profit)),
            gas_used=gas_used,
            strategy="sandwich",
            block_number=chain_info["block_number"],
            notes="statistical model",
        )
        self._emit(
            "trade",
            {
                "strategy": "sandwich",
                "profit_eth": net_profit,
                "gas_gwei": gas_price_gwei,
                "block": chain_info["block_number"],
            },
        )
        if self.cfg.record_all:
            self.recorder.record(trade)
        return trade

    # ------------------------------------------------------------------
    # Liquidation simulation
    # ------------------------------------------------------------------

    async def _sim_liquidation(self, chain_info: dict) -> Optional[Trade]:
        """
        Simplified liquidation simulation.

        Models that ~1% of blocks have a liquidation opportunity with a
        5–10% liquidation bonus.
        """
        import random

        if random.random() > 0.01:
            return None

        gas_price_gwei = chain_info["gas_price_gwei"]
        gas_used = 300_000
        gas_cost_eth = (gas_used * gas_price_gwei * 1e9) / 10**18

        # Collateral value ~0.5–5 ETH, liquidation bonus 5–10%
        collateral_eth = random.uniform(0.5, 5.0)
        bonus_rate = random.uniform(0.05, 0.10)
        gross_profit = collateral_eth * bonus_rate
        net_profit = gross_profit - gas_cost_eth

        if net_profit < self.cfg.min_profit_eth:
            return None

        trade = self.wallet.simulate_swap(
            token_in="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC (debt)
            token_out="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH (collateral)
            amount_in_wei=int(collateral_eth * 0.5 * 10**18),
            amount_out_wei=int(collateral_eth * 10**18),
            gas_used=gas_used,
            strategy="liquidation",
            block_number=chain_info["block_number"],
            notes=f"bonus={bonus_rate:.2%}",
        )
        self._emit(
            "trade",
            {
                "strategy": "liquidation",
                "profit_eth": net_profit,
                "collateral_eth": collateral_eth,
                "block": chain_info["block_number"],
            },
        )
        if self.cfg.record_all:
            self.recorder.record(trade)
        return trade

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, max_steps: int = 0) -> None:
        """
        Run the simulation loop.

        Args:
            max_steps: Stop after this many steps (0 = run indefinitely).
        """
        self._running = True
        self._start_time = time.time()
        self._emit("started", {"chain": self.cfg.chain, "strategies": self.cfg.strategies})

        logger.info("Simulation started | chain=%s", self.cfg.chain)

        while self._running:
            if self._paused:
                await asyncio.sleep(0.5)
                continue

            try:
                chain_info = await self.market.get_chain_info()
                self._emit("block", {"block": chain_info["block_number"], "gas_gwei": chain_info["gas_price_gwei"]})

                for strategy in self.cfg.strategies:
                    if strategy == "arbitrage":
                        for token_a, token_b in self.cfg.token_pairs:
                            await self._sim_arbitrage(token_a, token_b, chain_info)

                    elif strategy == "sandwich":
                        await self._sim_sandwich(chain_info)

                    elif strategy == "liquidation":
                        await self._sim_liquidation(chain_info)

                self._step += 1

                if max_steps and self._step >= max_steps:
                    logger.info("Reached max_steps=%d — stopping", max_steps)
                    break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                msg = str(exc)
                self._errors.append(msg)
                logger.error("Sim step %d error: %s", self._step, msg)
                self._emit("error", {"msg": msg})

            await asyncio.sleep(self.cfg.poll_interval_s)

        self._running = False
        final_stats = self.wallet.stats()
        self._emit("stopped", {"stats": final_stats})
        logger.info(
            "Simulation stopped | steps=%d | profit=%.6f ETH | trades=%d",
            self._step,
            final_stats["total_profit_eth"],
            final_stats["total_trades"],
        )

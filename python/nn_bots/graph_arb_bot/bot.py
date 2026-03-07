"""
GNN Arbitrage Bot — async EVM bot.

Strategy
--------
This bot builds a live token-pool graph from on-chain reserve data and
passes it through a Graph Neural Network to score every directed edge
(liquidity pool in one direction).  The two highest-scoring complementary
edges that form a profitable cycle are then executed as a two-leg arb.

Why a graph approach?
- Detects multi-hop opportunities that pure pair-scanning misses
  (e.g. WETH→USDC via pool A, USDC→WETH via pool B on a different DEX)
- Handles any number of tokens/pools without changing model architecture
- GNN embeddings propagate global liquidity context: a dry WETH pool
  affects the scores of all adjacent token pairs
"""

import asyncio
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

from config import GNNBotConfig, CHAINS
from model import GNNArbModel, TokenGraph, _v2_out_with_fee

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graph_arb_bot")

# Minimal ABIs
_FACTORY_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "tokenA", "type": "address"}, {"name": "tokenB", "type": "address"}],
        "name": "getPair",
        "outputs": [{"name": "pair", "type": "address"}],
        "type": "function",
    }
]
_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "_reserve0", "type": "uint112"},
            {"name": "_reserve1", "type": "uint112"},
            {"name": "_blockTimestampLast", "type": "uint32"},
        ],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function",
    },
]
_ROUTER_ABI = [
    {
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
        "type": "function",
    }
]
_ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]
_NULL_ADDR = "0x0000000000000000000000000000000000000000"


class GraphArbBot:
    """
    Graph Neural Network–driven arbitrage bot.

    Builds a fresh TokenGraph each scan cycle from live on-chain data,
    runs the GNN scorer, and executes any profitable 2-hop cycle.
    """

    def __init__(self, cfg: GNNBotConfig) -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(chain_cfg["rpc_url"]))
        if cfg.chain in ("bsc", "polygon", "avalanche"):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        self.account = self.w3.eth.account.from_key(cfg.private_key)
        self.chain_cfg = chain_cfg
        self.dexs = chain_cfg["dexs"]

        # GNN model
        self.model = GNNArbModel(
            hidden_dim=cfg.gnn_hidden_dim,
            num_layers=cfg.gnn_num_layers,
        )
        if Path(cfg.model_checkpoint).exists():
            self.model.load(cfg.model_checkpoint)

        # Cache for pool addresses (permanent — addresses never change)
        self._pair_cache: Dict[str, Optional[str]] = {}

        logger.info(
            "GraphArbBot ready | chain=%s | dexs=%s | wallet=%s",
            cfg.chain,
            list(self.dexs.keys()),
            self.account.address,
        )

    # ------------------------------------------------------------------
    # On-chain data fetching
    # ------------------------------------------------------------------

    async def _get_pair_address(
        self, factory: str, token_a: str, token_b: str
    ) -> Optional[str]:
        key = f"{factory.lower()}:{min(token_a, token_b).lower()}:{max(token_a, token_b).lower()}"
        if key in self._pair_cache:
            return self._pair_cache[key]
        try:
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(factory), abi=_FACTORY_ABI
            )
            pair = await contract.functions.getPair(
                self.w3.to_checksum_address(token_a),
                self.w3.to_checksum_address(token_b),
            ).call()
            result = pair if pair != _NULL_ADDR else None
        except Exception:
            result = None
        self._pair_cache[key] = result
        return result

    async def _get_reserves(
        self, pair_addr: str, token_a: str
    ) -> Tuple[int, int, str, str]:
        """Return (reserve_a, reserve_b, token0, token1)."""
        pair = self.w3.eth.contract(
            address=self.w3.to_checksum_address(pair_addr), abi=_PAIR_ABI
        )
        reserves, t0, t1 = await asyncio.gather(
            pair.functions.getReserves().call(),
            pair.functions.token0().call(),
            pair.functions.token1().call(),
        )
        r0, r1 = reserves[0], reserves[1]
        if t0.lower() == token_a.lower():
            return r0, r1, t0, t1
        return r1, r0, t1, t0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    async def build_graph(self) -> Tuple[TokenGraph, float]:
        """
        Fetch reserves for every (pair, DEX) combination and build a TokenGraph.

        Returns (graph, gas_price_gwei).
        """
        pairs = self.cfg.token_pairs
        if not pairs:
            # Fall back to default pairs from chain config
            pairs = [
                (p[0], p[1])
                for p in self.chain_cfg.get("token_pairs", [])
            ]

        gas_price = await self.w3.eth.gas_price
        gas_gwei = gas_price / 1e9

        graph = TokenGraph()
        total_pools = len(self.dexs) * len(pairs)

        pool_index = 0
        tasks = []

        async def _add_pool(dex_name: str, factory: str, fee_bps: int,
                            ta: str, tb: str, pidx: int) -> None:
            pair_addr = await self._get_pair_address(factory, ta, tb)
            if not pair_addr:
                return
            try:
                ra, rb, t0, t1 = await self._get_reserves(pair_addr, ta)
                graph.add_pool(ta, tb, ra, rb, fee_bps, dex_name, pidx, total_pools)
            except Exception as exc:
                logger.debug("Reserve fetch failed %s/%s on %s: %s", ta[-6:], tb[-6:], dex_name, exc)

        for dex_name, dex_info in self.dexs.items():
            factory = dex_info.get("factory")
            fee_bps = dex_info.get("fee_bps", 30)
            if not factory:
                continue
            for ta, tb in pairs:
                tasks.append(_add_pool(dex_name, factory, fee_bps, ta, tb, pool_index))
                pool_index += 1

        await asyncio.gather(*tasks, return_exceptions=True)
        return graph, gas_gwei

    # ------------------------------------------------------------------
    # Opportunity detection
    # ------------------------------------------------------------------

    async def find_opportunity(self) -> Optional[Dict]:
        """
        Build the graph, run GNN, find best profitable cycle.
        """
        graph, gas_gwei = await self.build_graph()
        if graph.num_edges < 2:
            return None

        trade_wei = int(self.cfg.trade_amount_eth * 1e18)
        gas_used_est = 400_000
        gas_cost_eth = (gas_used_est * gas_gwei * 1e9) / 1e18

        if gas_gwei > self.cfg.max_gas_price_gwei:
            self._emit_log("Gas too high: %.1f Gwei", gas_gwei)
            return None

        cycle = self.model.best_cycle(graph, amount_in_wei=trade_wei, gas_cost_eth=gas_cost_eth)
        if cycle is None:
            return None
        if cycle["net_profit_eth"] < self.cfg.min_profit_eth:
            return None

        cycle["trade_amount_wei"] = trade_wei
        cycle["gas_cost_eth"] = gas_cost_eth
        cycle["gas_gwei"] = gas_gwei
        return cycle

    @staticmethod
    def _emit_log(msg: str, *args) -> None:
        logger.debug(msg, *args)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _approve(self, token: str, spender: str, amount: int) -> None:
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=_ERC20_ABI
        )
        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        gas_price = await self.w3.eth.gas_price
        tx = await contract.functions.approve(spender, amount).build_transaction(
            {"from": self.account.address, "nonce": nonce, "gasPrice": gas_price}
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    async def execute(self, cycle: Dict) -> Optional[str]:
        """
        Execute the two swaps that form the arbitrage cycle.

        Leg 1: token_start → token_mid  (forward edge pool)
        Leg 2: token_mid   → token_start (backward edge pool)
        """
        token_start = cycle["token_start"]
        token_mid   = cycle["token_mid"]
        trade_wei   = cycle["trade_amount_wei"]
        slippage    = Decimal(1) - Decimal(self.cfg.slippage_bps) / Decimal(10_000)
        min_out_1   = int(cycle["out_fwd"] * slippage)

        gas_price = await self.w3.eth.gas_price
        if gas_price / 1e9 > self.cfg.max_gas_price_gwei:
            logger.warning("Gas price spiked — skipping execution")
            return None

        # We need to know which router to use for each edge.
        # The cycle only records edge indices; resolve the DEX name from edge features.
        # For now we use the first DEX that has a valid router — in production
        # you'd persist the dex_name in the edge features or lookup table.
        all_routers = [v["router"] for v in self.dexs.values() if "router" in v]
        if len(all_routers) < 1:
            logger.error("No routers configured")
            return None
        router1_addr = all_routers[0]
        router2_addr = all_routers[1] if len(all_routers) > 1 else all_routers[0]

        dl = int(time.time()) + 60

        # Leg 1
        await self._approve(token_start, router1_addr, trade_wei)
        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        r1 = self.w3.eth.contract(
            address=self.w3.to_checksum_address(router1_addr), abi=_ROUTER_ABI
        )
        tx = await r1.functions.swapExactTokensForTokens(
            trade_wei, min_out_1,
            [self.w3.to_checksum_address(token_start),
             self.w3.to_checksum_address(token_mid)],
            self.account.address, dl,
        ).build_transaction(
            {"from": self.account.address, "nonce": nonce, "gasPrice": gas_price}
        )
        signed = self.account.sign_transaction(tx)
        hash1 = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt1 = await self.w3.eth.wait_for_transaction_receipt(hash1, timeout=120)
        if receipt1["status"] != 1:
            logger.error("Leg 1 failed")
            return None

        # Leg 2: use full balance of token_mid
        mid_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token_mid), abi=_ERC20_ABI
        )
        bal_mid = await mid_contract.functions.balanceOf(self.account.address).call()
        await self._approve(token_mid, router2_addr, bal_mid)
        nonce += 1
        r2 = self.w3.eth.contract(
            address=self.w3.to_checksum_address(router2_addr), abi=_ROUTER_ABI
        )
        tx2 = await r2.functions.swapExactTokensForTokens(
            bal_mid, 0,
            [self.w3.to_checksum_address(token_mid),
             self.w3.to_checksum_address(token_start)],
            self.account.address, dl,
        ).build_transaction(
            {"from": self.account.address, "nonce": nonce, "gasPrice": gas_price}
        )
        signed2 = self.account.sign_transaction(tx2)
        hash2 = await self.w3.eth.send_raw_transaction(signed2.raw_transaction)
        await self.w3.eth.wait_for_transaction_receipt(hash2, timeout=120)
        logger.info(
            "GNN arb executed | %s↔%s | profit=%.6f ETH",
            token_start[-6:], token_mid[-6:], cycle["net_profit_eth"]
        )
        return hash2.hex()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("GraphArbBot running on %s …", self.cfg.chain)
        while True:
            try:
                cycle = await self.find_opportunity()
                if cycle:
                    logger.info(
                        "GNN opportunity | %s↔%s | net_profit=%.6f ETH | fwd_score=%.3f",
                        cycle["token_start"][-6:],
                        cycle["token_mid"][-6:],
                        cycle["net_profit_eth"],
                        cycle["fwd_score"],
                    )
                    await self.execute(cycle)
            except Exception as exc:
                logger.error("Main loop error: %s", exc)
            await asyncio.sleep(self.cfg.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg = GNNBotConfig(
        chain=sys.argv[1] if len(sys.argv) > 1 else "ethereum",
        token_pairs=[
            (
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            ),
            (
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            ),
        ],
    )
    asyncio.run(GraphArbBot(cfg).run())

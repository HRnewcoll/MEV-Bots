"""
Transformer Arbitrage Bot — async EVM bot.

Strategy
--------
1. For each monitored token pair, fetch on-chain reserves from every DEX
   on the configured chain.
2. Build a per-DEX feature matrix and feed it through the Transformer model.
3. The model outputs pairwise profit scores for all (buy_dex, sell_dex) combos.
4. If the top-scoring pair also passes the hard Uniswap V2 arithmetic check
   (to guard against model errors), execute the arbitrage.
5. Optionally submit via Flashbots for MEV-protection.

The Transformer learns long-range DEX interdependencies that simple spread
heuristics miss — e.g. "when DEX-A and DEX-B both have low liquidity the
spread on DEX-C is usually also small (correlated pools)".
"""

import asyncio
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

from config import TransformerBotConfig, UNISWAP_V2_PAIR_ABI, ERC20_ABI
from model import TransformerArbModel, _uniswap_v2_out

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("transformer_arb_bot")

FACTORY_ABI = [
    {
        "constant": True,
        "inputs": [
            {"name": "tokenA", "type": "address"},
            {"name": "tokenB", "type": "address"},
        ],
        "name": "getPair",
        "outputs": [{"name": "pair", "type": "address"}],
        "type": "function",
    }
]


class TransformerArbBot:
    """
    Arbitrage bot whose opportunity-scoring is driven by a Transformer encoder.

    All on-chain reads are async; execution uses the standard Uniswap V2 router.
    """

    def __init__(self, cfg: TransformerBotConfig) -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(chain_cfg["rpc_url"]))
        if cfg.chain in ("bsc", "polygon", "avalanche"):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        self.account = self.w3.eth.account.from_key(cfg.private_key)
        self.chain_cfg = chain_cfg
        self.dexs = chain_cfg["dexs"]
        self.dex_names = list(self.dexs.keys())

        # Transformer model
        self.model = TransformerArbModel(
            d_model=cfg.model_d_model,
            nhead=cfg.model_nhead,
            num_encoder_layers=cfg.model_num_layers,
            dim_feedforward=cfg.model_dim_feedforward,
            dropout=cfg.model_dropout,
        )
        if Path(cfg.model_checkpoint).exists():
            self.model.load(cfg.model_checkpoint)

        # Pair address cache
        self._pair_cache: Dict[str, Optional[str]] = {}

        logger.info(
            "TransformerArbBot ready | chain=%s | dexs=%s | wallet=%s",
            cfg.chain,
            self.dex_names,
            self.account.address,
        )

    # ------------------------------------------------------------------
    # On-chain data fetching
    # ------------------------------------------------------------------

    async def _get_pair_address(
        self, factory_addr: str, token_a: str, token_b: str
    ) -> Optional[str]:
        key = f"{factory_addr.lower()}:{token_a.lower()}:{token_b.lower()}"
        if key in self._pair_cache:
            return self._pair_cache[key]

        factory = self.w3.eth.contract(
            address=self.w3.to_checksum_address(factory_addr), abi=FACTORY_ABI
        )
        pair = await factory.functions.getPair(
            self.w3.to_checksum_address(token_a),
            self.w3.to_checksum_address(token_b),
        ).call()
        null = "0x0000000000000000000000000000000000000000"
        result = pair if pair != null else None
        self._pair_cache[key] = result
        return result

    async def _get_reserves_oriented(
        self,
        pair_addr: str,
        token_a: str,
    ) -> Tuple[int, int]:
        """Return (reserve_a, reserve_b) correctly oriented for token_a → token_b."""
        pair = self.w3.eth.contract(
            address=self.w3.to_checksum_address(pair_addr),
            abi=UNISWAP_V2_PAIR_ABI,
        )
        reserves = await pair.functions.getReserves().call()
        t0 = await pair.functions.token0().call()
        if t0.lower() == token_a.lower():
            return reserves[0], reserves[1]
        return reserves[1], reserves[0]

    async def fetch_dex_snapshots(
        self, token_a: str, token_b: str
    ) -> List[Dict]:
        """
        Fetch reserves from all DEXs for the given pair.

        Returns a list of snapshot dicts (one per DEX), preserving dex_names order.
        """
        gas_price = await self.w3.eth.gas_price
        gas_gwei = gas_price / 1e9

        async def _fetch(dex_name: str) -> Optional[Dict]:
            dex = self.dexs.get(dex_name, {})
            factory = dex.get("factory")
            if not factory:
                return None
            try:
                pair_addr = await self._get_pair_address(factory, token_a, token_b)
                if not pair_addr:
                    return None
                r_in, r_out = await self._get_reserves_oriented(pair_addr, token_a)
                return {
                    "dex_name": dex_name,
                    "reserve_in": r_in,
                    "reserve_out": r_out,
                    "fee_bps": dex.get("fee_bps", 30),
                }
            except Exception as exc:
                logger.debug("Failed to fetch from %s: %s", dex_name, exc)
                return None

        results = await asyncio.gather(*[_fetch(n) for n in self.dex_names])
        snapshots = [r for r in results if r is not None]
        return snapshots, gas_gwei

    # ------------------------------------------------------------------
    # Opportunity detection
    # ------------------------------------------------------------------

    async def find_opportunity(
        self, token_a: str, token_b: str
    ) -> Optional[Dict]:
        """
        Use the Transformer model to score all DEX pairs, then verify
        the best one arithmetically.
        """
        snapshots, gas_gwei = await self.fetch_dex_snapshots(token_a, token_b)
        if len(snapshots) < 2:
            return None

        amount_in = int(self.cfg.trade_amount_eth * 1e18)
        best = self.model.best_pair(snapshots, gas_price_gwei=gas_gwei)
        if best is None:
            return None

        buy_idx, sell_idx, score = best

        # Hard arithmetic verification — the model may be wrong
        buy_snap  = snapshots[buy_idx]
        sell_snap = snapshots[sell_idx]

        out_buy = _uniswap_v2_out(amount_in, buy_snap["reserve_in"], buy_snap["reserve_out"])
        if out_buy <= 0:
            return None
        out_sell = _uniswap_v2_out(out_buy, sell_snap["reserve_out"], sell_snap["reserve_in"])
        if out_sell <= 0:
            return None

        gross_profit_eth = (out_sell - amount_in) / 1e18
        gas_used = 350_000
        gas_cost_eth = (gas_used * gas_gwei * 1e9) / 1e18

        if gross_profit_eth - gas_cost_eth < self.cfg.min_profit_eth:
            return None

        return {
            "token_a": token_a,
            "token_b": token_b,
            "buy_dex": buy_snap["dex_name"],
            "sell_dex": sell_snap["dex_name"],
            "amount_in": amount_in,
            "out_buy": out_buy,
            "out_sell": out_sell,
            "gross_profit_eth": gross_profit_eth,
            "gas_cost_eth": gas_cost_eth,
            "net_profit_eth": gross_profit_eth - gas_cost_eth,
            "model_score": float(score),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _approve(self, token: str, spender: str, amount: int) -> None:
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=ERC20_ABI
        )
        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        gas_price = await self.w3.eth.gas_price
        tx = await contract.functions.approve(spender, amount).build_transaction(
            {"from": self.account.address, "nonce": nonce, "gasPrice": gas_price}
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    async def execute(self, opp: Dict) -> Optional[str]:
        """Execute the arbitrage described by *opp* and return the tx hash."""
        ROUTER_ABI = [
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
        slippage = Decimal(1) - Decimal(self.cfg.slippage_bps) / Decimal(10_000)
        min_out   = int(opp["out_buy"] * slippage)
        buy_router = self.dexs[opp["buy_dex"]]["router"]
        sell_router = self.dexs[opp["sell_dex"]]["router"]

        # Check gas
        gas_price = await self.w3.eth.gas_price
        if gas_price / 1e9 > self.cfg.max_gas_price_gwei:
            logger.warning("Gas price too high — skipping")
            return None

        dl = int(time.time()) + 60
        await self._approve(opp["token_a"], buy_router, opp["amount_in"])
        nonce = await self.w3.eth.get_transaction_count(self.account.address)

        router = self.w3.eth.contract(
            address=self.w3.to_checksum_address(buy_router), abi=ROUTER_ABI
        )
        tx = await router.functions.swapExactTokensForTokens(
            opp["amount_in"], min_out,
            [self.w3.to_checksum_address(opp["token_a"]),
             self.w3.to_checksum_address(opp["token_b"])],
            self.account.address, dl,
        ).build_transaction({"from": self.account.address, "nonce": nonce, "gasPrice": gas_price})
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt["status"] != 1:
            logger.error("Leg-1 failed")
            return None

        # Leg 2: sell token_b back
        token_b_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(opp["token_b"]), abi=ERC20_ABI
        )
        bal = await token_b_contract.functions.balanceOf(self.account.address).call()
        await self._approve(opp["token_b"], sell_router, bal)
        nonce += 1
        router2 = self.w3.eth.contract(
            address=self.w3.to_checksum_address(sell_router), abi=ROUTER_ABI
        )
        tx2 = await router2.functions.swapExactTokensForTokens(
            bal, 0,
            [self.w3.to_checksum_address(opp["token_b"]),
             self.w3.to_checksum_address(opp["token_a"])],
            self.account.address, dl,
        ).build_transaction({"from": self.account.address, "nonce": nonce, "gasPrice": gas_price})
        signed2 = self.account.sign_transaction(tx2)
        tx_hash2 = await self.w3.eth.send_raw_transaction(signed2.raw_transaction)
        await self.w3.eth.wait_for_transaction_receipt(tx_hash2, timeout=120)
        logger.info(
            "Arb executed | buy=%s sell=%s | net_profit=%.6f ETH",
            opp["buy_dex"], opp["sell_dex"], opp["net_profit_eth"]
        )
        return tx_hash2.hex()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        pairs = self.cfg.token_pairs
        if not pairs:
            # Default: use WETH/USDC from chain config
            chain_cfg = self.cfg.chain_config()
            token_pairs_cfg = chain_cfg.get("token_pairs_default", [])
            if token_pairs_cfg:
                pairs = token_pairs_cfg

        logger.info("TransformerArbBot running | pairs=%d", len(pairs))
        while True:
            for token_a, token_b in pairs:
                try:
                    opp = await self.find_opportunity(token_a, token_b)
                    if opp:
                        logger.info(
                            "Opportunity | buy=%s sell=%s | score=%.4f | profit=%.6f ETH",
                            opp["buy_dex"], opp["sell_dex"],
                            opp["model_score"], opp["net_profit_eth"],
                        )
                        await self.execute(opp)
                except Exception as exc:
                    logger.error("Error on %s/%s: %s", token_a[-6:], token_b[-6:], exc)
            await asyncio.sleep(self.cfg.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg = TransformerBotConfig(
        chain=sys.argv[1] if len(sys.argv) > 1 else "ethereum",
        token_pairs=[
            (
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            )
        ],
    )
    asyncio.run(TransformerArbBot(cfg).run())

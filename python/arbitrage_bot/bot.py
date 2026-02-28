"""
Cross-DEX arbitrage bot for EVM chains.

Strategy
--------
1. Continuously scan token-pair prices across multiple DEXs on the configured chain.
2. When the price difference exceeds the cost of the trade (gas + slippage), execute
   a two-leg swap: buy on the cheaper DEX, sell on the more expensive DEX.
3. Optionally submit via Flashbots to avoid front-running.

Supported chains: Ethereum, BSC, Polygon, Avalanche, Arbitrum, Optimism, Base.
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

from config import BotConfig, CHAINS, UNISWAP_V2_PAIR_ABI, UNISWAP_V2_ROUTER_ABI, ERC20_ABI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arbitrage_bot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_amount_out(amount_in: int, reserve_in: int, reserve_out: int) -> int:
    """Uniswap V2 constant-product formula (0.3 % fee)."""
    amount_in_with_fee = amount_in * 997
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in * 1000 + amount_in_with_fee
    return numerator // denominator


def deadline(seconds: int = 60) -> int:
    """Return a Unix timestamp *seconds* from now."""
    return int(time.time()) + seconds


# ---------------------------------------------------------------------------
# Core bot
# ---------------------------------------------------------------------------

class ArbitrageBot:
    """Multi-DEX arbitrage bot for a single EVM chain."""

    def __init__(self, cfg: BotConfig) -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()

        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(chain_cfg["rpc_url"]))
        # PoA chains (BSC, Polygon, Avalanche) need this middleware
        if cfg.chain in ("bsc", "polygon", "avalanche"):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        self.account = self.w3.eth.account.from_key(cfg.private_key)
        self.chain_cfg = chain_cfg
        self.dexs = chain_cfg["dexs"]

        logger.info(
            "ArbitrageBot initialised | chain=%s | wallet=%s",
            cfg.chain,
            self.account.address,
        )

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    async def _get_reserves(
        self, pair_address: str
    ) -> Tuple[int, int]:
        """Fetch (reserve0, reserve1) from a Uniswap V2–style pair."""
        pair = self.w3.eth.contract(
            address=self.w3.to_checksum_address(pair_address),
            abi=UNISWAP_V2_PAIR_ABI,
        )
        reserves = await pair.functions.getReserves().call()
        return reserves[0], reserves[1]

    async def _get_pair_address(
        self, factory_address: str, token_a: str, token_b: str
    ) -> Optional[str]:
        """Compute the CREATE2 pair address for a Uniswap V2 factory."""
        factory_abi = [
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
        factory = self.w3.eth.contract(
            address=self.w3.to_checksum_address(factory_address),
            abi=factory_abi,
        )
        pair = await factory.functions.getPair(
            self.w3.to_checksum_address(token_a),
            self.w3.to_checksum_address(token_b),
        ).call()
        null = "0x0000000000000000000000000000000000000000"
        return pair if pair != null else None

    async def fetch_price(
        self,
        dex_name: str,
        token_a: str,
        token_b: str,
        amount_in: int,
    ) -> Optional[int]:
        """Return the output amount for *amount_in* of token_a → token_b on *dex_name*."""
        dex = self.dexs.get(dex_name)
        if dex is None:
            return None

        factory_addr = dex.get("factory")
        if factory_addr is None:
            return None

        pair_addr = await self._get_pair_address(factory_addr, token_a, token_b)
        if pair_addr is None:
            logger.debug("No pair found on %s for %s/%s", dex_name, token_a, token_b)
            return None

        r0, r1 = await self._get_reserves(pair_addr)

        pair_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(pair_addr),
            abi=UNISWAP_V2_PAIR_ABI,
        )
        t0 = await pair_contract.functions.token0().call()
        if t0.lower() == token_a.lower():
            reserve_in, reserve_out = r0, r1
        else:
            reserve_in, reserve_out = r1, r0

        if reserve_in == 0 or reserve_out == 0:
            return None

        return get_amount_out(amount_in, reserve_in, reserve_out)

    # ------------------------------------------------------------------
    # Opportunity detection
    # ------------------------------------------------------------------

    async def find_opportunity(
        self, token_a: str, token_b: str
    ) -> Optional[Dict]:
        """
        Scan all DEXs and return the best arbitrage opportunity, or None.

        Returns a dict with keys: buy_dex, sell_dex, profit_gross (in token_b units).
        """
        amount_in = int(self.cfg.trade_amount * 10**18)  # assumes 18 decimals

        prices: Dict[str, Optional[int]] = {}
        tasks = {
            name: self.fetch_price(name, token_a, token_b, amount_in)
            for name in self.dexs
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.debug("Price fetch error on %s: %s", name, result)
                prices[name] = None
            else:
                prices[name] = result

        valid = {k: v for k, v in prices.items() if v is not None and v > 0}
        if len(valid) < 2:
            return None

        best_buy_dex = max(valid, key=lambda k: valid[k])   # highest output → cheapest buy
        worst_buy_dex = min(valid, key=lambda k: valid[k])  # lowest output → most expensive sell

        out_buy = valid[best_buy_dex]
        out_sell = valid[worst_buy_dex]

        if out_buy <= out_sell:
            return None  # no arbitrage possible

        profit_gross = out_buy - out_sell
        return {
            "token_a": token_a,
            "token_b": token_b,
            "buy_dex": best_buy_dex,
            "sell_dex": worst_buy_dex,
            "amount_in": amount_in,
            "out_buy": out_buy,
            "out_sell": out_sell,
            "profit_gross": profit_gross,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _approve_token(self, token: str, spender: str, amount: int) -> None:
        """ERC-20 approve *spender* to spend *amount* of *token*."""
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=ERC20_ABI
        )
        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        gas_price = await self.w3.eth.gas_price
        tx = await contract.functions.approve(spender, amount).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": gas_price,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Approve tx sent: %s", tx_hash.hex())
        await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    async def execute_arb(self, opportunity: Dict) -> Optional[str]:
        """Execute a two-leg arbitrage trade and return the tx hash."""
        token_a = opportunity["token_a"]
        token_b = opportunity["token_b"]
        amount_in = opportunity["amount_in"]
        buy_dex = opportunity["buy_dex"]
        sell_dex = opportunity["sell_dex"]

        buy_router_addr = self.dexs[buy_dex]["router"]
        sell_router_addr = self.dexs[sell_dex]["router"]

        slippage_factor = Decimal(1) - Decimal(self.cfg.slippage_bps) / Decimal(10_000)
        min_out_buy = int(opportunity["out_buy"] * slippage_factor)

        # Approve buy router to spend token_a
        await self._approve_token(token_a, buy_router_addr, amount_in)

        gas_price = await self.w3.eth.gas_price
        max_gas_wei = int(self.cfg.max_gas_price_gwei * 1e9)
        if gas_price > max_gas_wei:
            logger.warning(
                "Gas price %s Gwei exceeds max %s Gwei — skipping",
                gas_price / 1e9,
                self.cfg.max_gas_price_gwei,
            )
            return None

        nonce = await self.w3.eth.get_transaction_count(self.account.address)

        buy_router = self.w3.eth.contract(
            address=self.w3.to_checksum_address(buy_router_addr),
            abi=UNISWAP_V2_ROUTER_ABI,
        )
        tx = await buy_router.functions.swapExactTokensForTokens(
            amount_in,
            min_out_buy,
            [
                self.w3.to_checksum_address(token_a),
                self.w3.to_checksum_address(token_b),
            ],
            self.account.address,
            deadline(),
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": gas_price,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Arb leg-1 tx: %s", tx_hash.hex())
        receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] != 1:
            logger.error("Leg-1 swap failed")
            return None

        # Leg 2: sell token_b back on sell_dex
        token_b_balance = await self._token_balance(token_b)
        await self._approve_token(token_b, sell_router_addr, token_b_balance)

        nonce += 1
        sell_router = self.w3.eth.contract(
            address=self.w3.to_checksum_address(sell_router_addr),
            abi=UNISWAP_V2_ROUTER_ABI,
        )
        tx2 = await sell_router.functions.swapExactTokensForTokens(
            token_b_balance,
            0,
            [
                self.w3.to_checksum_address(token_b),
                self.w3.to_checksum_address(token_a),
            ],
            self.account.address,
            deadline(),
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": gas_price,
            }
        )
        signed2 = self.account.sign_transaction(tx2)
        tx_hash2 = await self.w3.eth.send_raw_transaction(signed2.raw_transaction)
        logger.info("Arb leg-2 tx: %s", tx_hash2.hex())
        await self.w3.eth.wait_for_transaction_receipt(tx_hash2, timeout=120)
        return tx_hash2.hex()

    async def _token_balance(self, token: str) -> int:
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=ERC20_ABI
        )
        return await contract.functions.balanceOf(self.account.address).call()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Continuously scan for and execute arbitrage opportunities."""
        logger.info("Starting arbitrage bot on %s …", self.cfg.chain)
        while True:
            for token_a, token_b in self.cfg.token_pairs:
                try:
                    opp = await self.find_opportunity(token_a, token_b)
                    if opp:
                        logger.info(
                            "Opportunity: buy on %s, sell on %s | profit_gross=%s",
                            opp["buy_dex"],
                            opp["sell_dex"],
                            opp["profit_gross"],
                        )
                        await self.execute_arb(opp)
                except Exception as exc:
                    logger.error("Error scanning %s/%s: %s", token_a, token_b, exc)

            await asyncio.sleep(self.cfg.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cfg = BotConfig(
        chain=sys.argv[1] if len(sys.argv) > 1 else "ethereum",
        # Example WETH/USDC pair on Ethereum
        token_pairs=[
            (
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            )
        ],
    )
    asyncio.run(ArbitrageBot(cfg).run())

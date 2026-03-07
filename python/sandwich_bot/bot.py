"""
Sandwich bot — educational reference implementation.

Strategy
--------
1. Monitor the public mempool for large pending swaps on Uniswap V2–style DEXs.
2. When a profitable victim transaction is detected:
   a. **Front-run**: submit a swap in the same direction *before* the victim.
   b. **Back-run**: submit the reverse swap *after* the victim to capture the price impact.
3. Both legs are submitted as a Flashbots bundle to guarantee ordering without
   competing with other searchers in the public mempool.

⚠️  WARNING: Sandwich attacks directly harm the victim's execution price.
    This code is provided for educational purposes only.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from web3 import AsyncWeb3
from web3.types import TxData

from config import SandwichConfig, UNISWAP_V2_ROUTER_ABI, ERC20_ABI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sandwich_bot")


# ---------------------------------------------------------------------------
# ABI for decoding swapExactTokensForTokens / swapExactETHForTokens
# ---------------------------------------------------------------------------

SWAP_SELECTORS = {
    "0x38ed1739": "swapExactTokensForTokens",
    "0x7ff36ab5": "swapExactETHForTokens",
    "0x18cbafe5": "swapExactTokensForETH",
}


@dataclass
class SandwichOpportunity:
    victim_tx: TxData
    token_in: str
    token_out: str
    amount_in: int
    amount_out_min: int
    router_address: str
    path: list


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class SandwichBot:
    """Mempool-watching sandwich bot."""

    def __init__(self, cfg: "SandwichConfig") -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()
        ws_url = chain_cfg.get("ws_url") or chain_cfg["rpc_url"].replace("https://", "wss://")
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncWebSocketProvider(ws_url))
        self.account = self.w3.eth.account.from_key(cfg.private_key)
        self.chain_cfg = chain_cfg
        logger.info("SandwichBot | chain=%s | wallet=%s", cfg.chain, self.account.address)

    # ------------------------------------------------------------------
    # Mempool monitoring
    # ------------------------------------------------------------------

    def _decode_swap(self, tx: TxData) -> Optional[SandwichOpportunity]:
        """Try to decode a pending transaction as a Uniswap V2–style swap."""
        if not tx.get("input") or len(tx["input"]) < 10:
            return None

        selector = tx["input"][:10].lower()
        if selector not in SWAP_SELECTORS:
            return None

        # Verify the recipient is a known DEX router
        to_addr = (tx.get("to") or "").lower()
        known_routers = {
            dex["router"].lower()
            for dex in self.chain_cfg["dexs"].values()
            if "router" in dex
        }
        if to_addr not in known_routers:
            return None

        try:
            router = self.w3.eth.contract(
                address=self.w3.to_checksum_address(tx["to"]),
                abi=UNISWAP_V2_ROUTER_ABI,
            )
            fn, params = router.decode_function_input(tx["input"])
        except Exception:
            return None

        path = params.get("path", [])
        if len(path) < 2:
            return None

        amount_in = params.get("amountIn", tx.get("value", 0))
        amount_out_min = params.get("amountOutMin", 0)

        return SandwichOpportunity(
            victim_tx=tx,
            token_in=path[0],
            token_out=path[-1],
            amount_in=amount_in,
            amount_out_min=amount_out_min,
            router_address=tx["to"],
            path=path,
        )

    async def _estimate_profit(self, opp: SandwichOpportunity) -> float:
        """
        Rough off-chain profit estimate.

        Returns expected profit in ETH; negative means unprofitable.
        """
        # Simplified: profit ≈ price impact from victim × our front-run amount
        victim_fraction = opp.amount_in / (opp.amount_in + 1e18)  # normalised
        estimated_price_impact = victim_fraction * 0.003  # rough 0.3% per unit
        our_profit_fraction = estimated_price_impact * self.cfg.front_run_amount
        gas_cost_eth = self.cfg.gas_limit * (await self.w3.eth.gas_price) / 1e18 * 2
        return our_profit_fraction - gas_cost_eth

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _send_bundle(
        self,
        front_tx: bytes,
        victim_tx_hash: str,
        back_tx: bytes,
        target_block: int,
    ) -> None:
        """Submit a Flashbots bundle containing front-run and back-run txs."""
        import aiohttp, json

        bundle = {
            "jsonrpc": "2.0",
            "method": "eth_sendBundle",
            "params": [
                {
                    "txs": [
                        "0x" + front_tx.hex(),
                        victim_tx_hash,
                        "0x" + back_tx.hex(),
                    ],
                    "blockNumber": hex(target_block),
                }
            ],
            "id": 1,
        }

        signer = self.w3.eth.account.from_key(self.cfg.flashbots_signer_key)
        body = json.dumps(bundle)
        sig = signer.sign_message(
            self.w3.eth.account.messages.defunct_hash_message(text=body)
        )
        headers = {
            "Content-Type": "application/json",
            "X-Flashbots-Signature": f"{signer.address}:{sig.signature.hex()}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.cfg.flashbots_relay_url, data=body, headers=headers
            ) as resp:
                result = await resp.json()
                logger.info("Flashbots bundle response: %s", result)

    async def _build_front_run(self, opp: SandwichOpportunity) -> bytes:
        """Build the front-run swap transaction (signed, not yet broadcast)."""
        router = self.w3.eth.contract(
            address=self.w3.to_checksum_address(opp.router_address),
            abi=UNISWAP_V2_ROUTER_ABI,
        )
        gas_price = await self.w3.eth.gas_price
        # Front-run with higher gas price to ensure ordering
        boosted_gas = int(gas_price * 1.15)
        nonce = await self.w3.eth.get_transaction_count(self.account.address)

        import time
        tx = await router.functions.swapExactTokensForTokens(
            self.cfg.front_run_amount,
            0,
            opp.path,
            self.account.address,
            int(time.time()) + 60,
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": boosted_gas,
                "gas": self.cfg.gas_limit,
            }
        )
        signed = self.account.sign_transaction(tx)
        return signed.raw_transaction

    async def _build_back_run(self, opp: SandwichOpportunity, nonce: int) -> bytes:
        """Build the back-run swap transaction (reverse path)."""
        router = self.w3.eth.contract(
            address=self.w3.to_checksum_address(opp.router_address),
            abi=UNISWAP_V2_ROUTER_ABI,
        )
        gas_price = await self.w3.eth.gas_price
        reversed_path = list(reversed(opp.path))

        # Use the balance acquired in the front-run
        token_out_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(opp.token_out), abi=ERC20_ABI
        )
        balance = await token_out_contract.functions.balanceOf(self.account.address).call()

        import time
        tx = await router.functions.swapExactTokensForTokens(
            balance,
            0,
            reversed_path,
            self.account.address,
            int(time.time()) + 60,
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": int(gas_price * 0.9),  # back-run slightly lower priority
                "gas": self.cfg.gas_limit,
            }
        )
        signed = self.account.sign_transaction(tx)
        return signed.raw_transaction

    async def _process_tx(self, tx_hash: str) -> None:
        try:
            tx = await self.w3.eth.get_transaction(tx_hash)
        except Exception:
            return

        opp = self._decode_swap(tx)
        if opp is None:
            return

        profit = await self._estimate_profit(opp)
        if profit <= 0:
            return

        logger.info(
            "Sandwich candidate | victim=%s | profit_est=%.6f ETH",
            tx_hash,
            profit,
        )

        front_tx_bytes = await self._build_front_run(opp)
        nonce = await self.w3.eth.get_transaction_count(self.account.address) + 1
        back_tx_bytes = await self._build_back_run(opp, nonce)
        target_block = await self.w3.eth.block_number + 1

        await self._send_bundle(front_tx_bytes, tx_hash, back_tx_bytes, target_block)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Subscribe to pending transactions and scan for sandwich opportunities."""
        logger.info("SandwichBot listening for pending transactions …")
        subscription_id = await self.w3.eth.subscribe("newPendingTransactions")
        async for tx_hash in self.w3.socket.process_subscriptions():
            if isinstance(tx_hash, dict):
                tx_hash = tx_hash.get("result") or tx_hash.get("params", {}).get(
                    "result", ""
                )
            asyncio.create_task(self._process_tx(tx_hash))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import SandwichConfig

    cfg = SandwichConfig()
    asyncio.run(SandwichBot(cfg).run())

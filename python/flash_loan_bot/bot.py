"""
Flash Loan Arbitrage bot using Aave V3.

Strategy
--------
1. Detect a profitable price discrepancy across two DEXs.
2. In a *single atomic transaction* via the FlashLoanArbitrage contract:
   a. Borrow the base token (e.g. USDC) from Aave V3 with zero upfront capital.
   b. Buy the target token on the cheaper DEX.
   c. Sell the target token on the more expensive DEX.
   d. Repay the flash loan + 0.05 % Aave fee.
   e. Keep the profit.
3. If the transaction reverts (e.g. profit < fee), nothing is lost.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

from config import FlashLoanConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("flash_loan_bot")

# Path to the compiled FlashLoanArbitrage ABI
_CONTRACT_ABI_PATH = (
    Path(__file__).resolve().parent.parent.parent / "contracts" / "FlashLoanArbitrage.abi.json"
)

# Inline minimal ABI so the bot works even without a compiled artifact
FLASH_LOAN_ARB_ABI = [
    {
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "dexA", "type": "address"},
            {"name": "dexB", "type": "address"},
            {"name": "tokenIn", "type": "address"},
            {"name": "tokenOut", "type": "address"},
        ],
        "name": "executeArbitrage",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


class FlashLoanBot:
    """Off-chain trigger for on-chain flash loan arbitrage."""

    def __init__(self, cfg: FlashLoanConfig) -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(chain_cfg["rpc_url"]))
        if cfg.chain in ("bsc", "polygon", "avalanche"):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.account = self.w3.eth.account.from_key(cfg.private_key)

        # Load ABI from file if available, otherwise use inline
        abi = FLASH_LOAN_ARB_ABI
        if _CONTRACT_ABI_PATH.exists():
            with open(_CONTRACT_ABI_PATH) as f:
                abi = json.load(f)

        self.contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(cfg.contract_address),
            abi=abi,
        )
        logger.info(
            "FlashLoanBot | chain=%s | contract=%s | wallet=%s",
            cfg.chain,
            cfg.contract_address,
            self.account.address,
        )

    # ------------------------------------------------------------------
    # Price checking (simplified — same logic as arbitrage_bot)
    # ------------------------------------------------------------------

    async def _check_arbitrage(
        self, token_in: str, token_out: str, amount: int
    ) -> Optional[dict]:
        """
        Return opportunity dict if arbitrage is profitable, else None.
        Uses the same DEX registry as the arbitrage_bot.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arbitrage_bot"))
        from bot import ArbitrageBot  # local import to avoid circular deps
        from config import BotConfig

        arb_cfg = BotConfig(
            private_key=self.cfg.private_key,
            chain=self.cfg.chain,
            trade_amount=amount / 10**18,
            token_pairs=[(token_in, token_out)],
        )
        arb_bot = ArbitrageBot(arb_cfg)
        return await arb_bot.find_opportunity(token_in, token_out)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def trigger_flash_loan(
        self,
        asset: str,
        amount: int,
        dex_a: str,
        dex_b: str,
        token_in: str,
        token_out: str,
    ) -> Optional[str]:
        """Call executeArbitrage on the deployed FlashLoanArbitrage contract."""
        gas_price = await self.w3.eth.gas_price
        max_gas_wei = int(self.cfg.max_gas_price_gwei * 1e9)
        if gas_price > max_gas_wei:
            logger.warning("Gas too high: %.1f Gwei", gas_price / 1e9)
            return None

        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        tx = await self.contract.functions.executeArbitrage(
            self.w3.to_checksum_address(asset),
            amount,
            self.w3.to_checksum_address(dex_a),
            self.w3.to_checksum_address(dex_b),
            self.w3.to_checksum_address(token_in),
            self.w3.to_checksum_address(token_out),
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": gas_price,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Flash loan arb tx: %s", tx_hash.hex())
        receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt["status"] == 1:
            logger.info("Flash loan succeeded!")
            return tx_hash.hex()
        logger.error("Flash loan transaction reverted")
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("FlashLoanBot scanning for opportunities …")
        while True:
            for token_in, token_out in self.cfg.token_pairs:
                try:
                    opp = await self._check_arbitrage(
                        token_in, token_out, self.cfg.flash_loan_amount
                    )
                    if opp:
                        chain_cfg = self.cfg.chain_config()
                        dex_names = list(chain_cfg["dexs"].keys())
                        dex_a_addr = chain_cfg["dexs"][opp["buy_dex"]]["router"]
                        dex_b_addr = chain_cfg["dexs"][opp["sell_dex"]]["router"]
                        await self.trigger_flash_loan(
                            asset=token_in,
                            amount=self.cfg.flash_loan_amount,
                            dex_a=dex_a_addr,
                            dex_b=dex_b_addr,
                            token_in=token_in,
                            token_out=token_out,
                        )
                except Exception as exc:
                    logger.error("Flash loan scan error: %s", exc)
            await asyncio.sleep(self.cfg.poll_interval)


if __name__ == "__main__":
    from config import FlashLoanConfig

    asyncio.run(FlashLoanBot(FlashLoanConfig()).run())

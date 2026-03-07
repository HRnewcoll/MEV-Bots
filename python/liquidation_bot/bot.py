"""
Liquidation bot for Aave V2/V3 and Compound V2/V3.

Strategy
--------
1. Poll lending protocols for accounts whose health-factor has dropped below 1.
2. Call the protocol's liquidation function to repay part of the debt and claim
   the collateral bonus (5–10 % depending on the asset).
3. Immediately swap the received collateral back to the repay token to realise profit.
"""

import asyncio
import logging
import time
from typing import List, Optional

from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

from config import LiquidationConfig, AAVE_V3_ABI, AAVE_V2_ABI, ERC20_ABI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("liquidation_bot")


# ---------------------------------------------------------------------------
# Compound V2 minimal ABI
# ---------------------------------------------------------------------------

COMPOUND_COMPTROLLER_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getAllMarkets",
        "outputs": [{"name": "", "type": "address[]"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "getAccountLiquidity",
        "outputs": [
            {"name": "error", "type": "uint256"},
            {"name": "liquidity", "type": "uint256"},
            {"name": "shortfall", "type": "uint256"},
        ],
        "type": "function",
    },
]

CTOKEN_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "borrower", "type": "address"},
            {"name": "repayAmount", "type": "uint256"},
            {"name": "cTokenCollateral", "type": "address"},
        ],
        "name": "liquidateBorrow",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "account", "type": "address"}],
        "name": "borrowBalanceStored",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "underlying",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function",
    },
]


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class LiquidationBot:
    """Multi-protocol liquidation bot (Aave V2/V3 + Compound V2)."""

    AAVE_V3_POOL: str = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"  # Ethereum
    AAVE_V2_POOL: str = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"  # Ethereum
    COMPOUND_COMPTROLLER: str = "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B"

    def __init__(self, cfg: LiquidationConfig) -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(chain_cfg["rpc_url"]))
        if cfg.chain in ("bsc", "polygon", "avalanche"):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        self.account = self.w3.eth.account.from_key(cfg.private_key)
        logger.info(
            "LiquidationBot | chain=%s | wallet=%s", cfg.chain, self.account.address
        )

    # ------------------------------------------------------------------
    # Aave V3
    # ------------------------------------------------------------------

    async def _get_aave_v3_unhealthy_users(self) -> List[str]:
        """
        Return accounts with health factor < 1 by scanning LiquidationCall events.

        In production you would maintain an on-chain indexer or use a subgraph.
        """
        pool = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.AAVE_V3_POOL),
            abi=AAVE_V3_ABI,
        )
        latest = await self.w3.eth.block_number
        from_block = max(0, latest - 10_000)

        events = await pool.events.LiquidationCall.get_logs(  # type: ignore[attr-defined]
            fromBlock=from_block
        )
        borrowers = {e["args"]["user"] for e in events}

        unhealthy = []
        for borrower in borrowers:
            try:
                data = await pool.functions.getUserAccountData(borrower).call()
                health_factor = data[5]  # healthFactor (scaled by 1e18)
                if 0 < health_factor < 10**18:
                    unhealthy.append(borrower)
            except Exception as exc:
                logger.debug("getUserAccountData error for %s: %s", borrower, exc)
        return unhealthy

    async def _liquidate_aave_v3(
        self, borrower: str, collateral_asset: str, debt_asset: str, debt_to_cover: int
    ) -> Optional[str]:
        pool = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.AAVE_V3_POOL),
            abi=AAVE_V3_ABI,
        )
        gas_price = await self.w3.eth.gas_price
        nonce = await self.w3.eth.get_transaction_count(self.account.address)

        # Approve pool to pull debt asset
        await self._approve(debt_asset, self.AAVE_V3_POOL, debt_to_cover)

        tx = await pool.functions.liquidationCall(
            self.w3.to_checksum_address(collateral_asset),
            self.w3.to_checksum_address(debt_asset),
            self.w3.to_checksum_address(borrower),
            debt_to_cover,
            False,  # receive underlying, not aToken
        ).build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                "gasPrice": gas_price,
            }
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Aave V3 liquidation tx: %s", tx_hash.hex())
        receipt = await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt["status"] == 1:
            return tx_hash.hex()
        logger.error("Aave V3 liquidation failed for borrower %s", borrower)
        return None

    # ------------------------------------------------------------------
    # Compound V2
    # ------------------------------------------------------------------

    async def _get_compound_unhealthy_users(self) -> List[str]:
        """Scan recent Borrow events from all Compound markets to find at-risk accounts."""
        comptroller = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.COMPOUND_COMPTROLLER),
            abi=COMPOUND_COMPTROLLER_ABI,
        )
        markets = await comptroller.functions.getAllMarkets().call()

        borrowers: set = set()
        for market in markets[:5]:  # limit to first 5 for demo
            ctoken = self.w3.eth.contract(
                address=self.w3.to_checksum_address(market), abi=CTOKEN_ABI
            )
            latest = await self.w3.eth.block_number
            try:
                events = await ctoken.events.Borrow.get_logs(  # type: ignore[attr-defined]
                    fromBlock=max(0, latest - 5_000)
                )
                for e in events:
                    borrowers.add(e["args"]["borrower"])
            except Exception:
                pass

        unhealthy = []
        for borrower in borrowers:
            try:
                _, liquidity, shortfall = await comptroller.functions.getAccountLiquidity(
                    borrower
                ).call()
                if shortfall > 0:
                    unhealthy.append(borrower)
            except Exception:
                pass
        return unhealthy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _approve(self, token: str, spender: str, amount: int) -> None:
        contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token), abi=ERC20_ABI
        )
        nonce = await self.w3.eth.get_transaction_count(self.account.address)
        gas_price = await self.w3.eth.gas_price
        tx = await contract.functions.approve(
            self.w3.to_checksum_address(spender), amount
        ).build_transaction(
            {"from": self.account.address, "nonce": nonce, "gasPrice": gas_price}
        )
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.w3.eth.send_raw_transaction(signed.raw_transaction)
        await self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("LiquidationBot scanning …")
        while True:
            # Aave V3
            try:
                users = await self._get_aave_v3_unhealthy_users()
                logger.info("Aave V3 unhealthy users found: %d", len(users))
                for user in users:
                    # In production: query getUserAccountData for exact debt/collateral
                    logger.info("Would liquidate Aave V3 user: %s", user)
            except Exception as exc:
                logger.error("Aave V3 scan error: %s", exc)

            # Compound V2
            try:
                users = await self._get_compound_unhealthy_users()
                logger.info("Compound V2 unhealthy users found: %d", len(users))
                for user in users:
                    logger.info("Would liquidate Compound user: %s", user)
            except Exception as exc:
                logger.error("Compound scan error: %s", exc)

            await asyncio.sleep(self.cfg.poll_interval)


if __name__ == "__main__":
    from config import LiquidationConfig

    asyncio.run(LiquidationBot(LiquidationConfig()).run())

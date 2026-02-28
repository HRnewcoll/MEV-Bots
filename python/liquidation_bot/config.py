"""Configuration for the liquidation bot."""

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arbitrage_bot"))
from config import CHAINS, ERC20_ABI  # noqa: F401

# Aave V3 Pool ABI (minimal — only functions we call)
AAVE_V3_ABI = [
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "getUserAccountData",
        "outputs": [
            {"name": "totalCollateralBase", "type": "uint256"},
            {"name": "totalDebtBase", "type": "uint256"},
            {"name": "availableBorrowsBase", "type": "uint256"},
            {"name": "currentLiquidationThreshold", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "healthFactor", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "collateralAsset", "type": "address"},
            {"name": "debtAsset", "type": "address"},
            {"name": "user", "type": "address"},
            {"name": "debtToCover", "type": "uint256"},
            {"name": "receiveAToken", "type": "bool"},
        ],
        "name": "liquidationCall",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "collateralAsset", "type": "address"},
            {"indexed": True, "name": "debtAsset", "type": "address"},
            {"indexed": True, "name": "user", "type": "address"},
            {"indexed": False, "name": "debtToCover", "type": "uint256"},
            {"indexed": False, "name": "liquidatedCollateralAmount", "type": "uint256"},
            {"indexed": False, "name": "liquidator", "type": "address"},
            {"indexed": False, "name": "receiveAToken", "type": "bool"},
        ],
        "name": "LiquidationCall",
        "type": "event",
    },
]

# Aave V2 Pool ABI (subset)
AAVE_V2_ABI = AAVE_V3_ABI  # V2 has same relevant interface for our purposes


@dataclass
class LiquidationConfig:
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    chain: str = os.getenv("CHAIN", "ethereum")
    poll_interval: float = float(os.getenv("POLL_INTERVAL", "15"))
    min_profit_usd: float = float(os.getenv("MIN_PROFIT_USD", "5.0"))
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))

    def chain_config(self) -> Dict:
        if self.chain not in CHAINS:
            raise ValueError(f"Unknown chain: {self.chain!r}")
        return CHAINS[self.chain]

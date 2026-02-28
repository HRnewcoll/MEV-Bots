"""Configuration for the flash loan bot."""

import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arbitrage_bot"))
from config import CHAINS  # noqa: F401

__all__ = ["FlashLoanConfig", "CHAINS"]


@dataclass
class FlashLoanConfig:
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    chain: str = os.getenv("CHAIN", "ethereum")
    # Deployed FlashLoanArbitrage contract address
    contract_address: str = os.getenv("FLASH_LOAN_CONTRACT", "0x0000000000000000000000000000000000000000")
    # Amount to borrow in base token units (e.g. 10_000 USDC = 10_000 * 1e6)
    flash_loan_amount: int = int(float(os.getenv("FLASH_LOAN_AMOUNT", "10000")) * 10**6)
    poll_interval: float = float(os.getenv("POLL_INTERVAL", "2"))
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))
    token_pairs: List[tuple] = field(default_factory=lambda: [
        (
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        )
    ])

    def chain_config(self) -> Dict:
        if self.chain not in CHAINS:
            raise ValueError(f"Unknown chain: {self.chain!r}")
        return CHAINS[self.chain]

"""Configuration for the sandwich bot."""

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# Re-use the chain registry from the arbitrage config
from sys import path as _sys_path
import os as _os

_sys_path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "arbitrage_bot"))
from config import CHAINS, UNISWAP_V2_ROUTER_ABI, ERC20_ABI  # noqa: F401

__all__ = ["SandwichConfig", "CHAINS", "UNISWAP_V2_ROUTER_ABI", "ERC20_ABI"]


@dataclass
class SandwichConfig:
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    chain: str = os.getenv("CHAIN", "ethereum")
    front_run_amount: int = int(float(os.getenv("TRADE_AMOUNT", "0.1")) * 10**18)
    gas_limit: int = int(os.getenv("GAS_LIMIT", "300000"))
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))
    flashbots_relay_url: str = field(
        default_factory=lambda: os.getenv(
            "FLASHBOTS_RELAY_URL", "https://relay.flashbots.net"
        )
    )
    flashbots_signer_key: str = field(
        default_factory=lambda: os.getenv("FLASHBOTS_SIGNER_KEY", "")
    )

    def chain_config(self) -> Dict:
        if self.chain not in CHAINS:
            raise ValueError(f"Unknown chain: {self.chain!r}")
        return CHAINS[self.chain]

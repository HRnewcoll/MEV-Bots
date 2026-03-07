"""Configuration for the Transformer Arbitrage Bot."""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# Re-use the chain/DEX registry from the arbitrage bot
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "arbitrage_bot"))
from config import CHAINS, UNISWAP_V2_PAIR_ABI, ERC20_ABI  # noqa: F401

__all__ = ["TransformerBotConfig", "CHAINS"]


@dataclass
class TransformerBotConfig:
    """Runtime configuration for the Transformer Arbitrage Bot."""

    # ---- Blockchain ----
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    chain: str = os.getenv("CHAIN", "ethereum")
    poll_interval: float = float(os.getenv("POLL_INTERVAL", "1.0"))

    # ---- Risk limits ----
    min_profit_eth: float = float(os.getenv("MIN_PROFIT_ETH", "0.001"))
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))
    trade_amount_eth: float = float(os.getenv("TRADE_AMOUNT", "0.1"))
    slippage_bps: int = int(os.getenv("SLIPPAGE_BPS", "50"))

    # ---- Transformer model hyperparameters ----
    model_d_model: int = int(os.getenv("TF_D_MODEL", "64"))
    model_nhead: int = int(os.getenv("TF_NHEAD", "4"))
    model_num_layers: int = int(os.getenv("TF_NUM_LAYERS", "3"))
    model_dim_feedforward: int = int(os.getenv("TF_DIM_FF", "256"))
    model_dropout: float = float(os.getenv("TF_DROPOUT", "0.1"))

    # ---- Token pairs to monitor ----
    token_pairs: List[tuple] = field(default_factory=list)

    # ---- Checkpoint ----
    model_checkpoint: str = os.getenv(
        "TF_CHECKPOINT", "checkpoints/transformer_arb.pt"
    )

    # ---- Flashbots ----
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
            raise ValueError(f"Unknown chain: {self.chain!r}. Available: {list(CHAINS)}")
        return CHAINS[self.chain]

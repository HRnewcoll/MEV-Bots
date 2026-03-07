"""Configuration for the GNN Arbitrage Bot."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "arbitrage_bot"))
from config import CHAINS  # noqa: F401

__all__ = ["GNNBotConfig", "CHAINS"]


@dataclass
class GNNBotConfig:
    """Runtime configuration for the GNN Arbitrage Bot."""

    # ---- Blockchain ----
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    chain: str = os.getenv("CHAIN", "ethereum")
    poll_interval: float = float(os.getenv("POLL_INTERVAL", "1.0"))

    # ---- Risk limits ----
    min_profit_eth: float = float(os.getenv("MIN_PROFIT_ETH", "0.001"))
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))
    trade_amount_eth: float = float(os.getenv("TRADE_AMOUNT", "0.1"))
    slippage_bps: int = int(os.getenv("SLIPPAGE_BPS", "50"))

    # ---- GNN model hyperparameters ----
    gnn_hidden_dim: int = int(os.getenv("GNN_HIDDEN_DIM", "64"))
    gnn_num_layers: int = int(os.getenv("GNN_NUM_LAYERS", "3"))

    # ---- Multi-hop settings ----
    max_path_length: int = int(os.getenv("MAX_PATH_LENGTH", "2"))  # 2 = single intermediate hop

    # ---- Token pairs / tokens to build the graph over ----
    # List of (token_a_addr, token_b_addr) tuples for graph construction
    token_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # ---- Checkpoint ----
    model_checkpoint: str = os.getenv(
        "GNN_CHECKPOINT", "checkpoints/gnn_arb.pt"
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

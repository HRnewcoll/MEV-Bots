"""Configuration for the AI MEV bot."""

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arbitrage_bot"))
from config import CHAINS  # noqa: F401

__all__ = ["AIBotConfig", "CHAINS"]


@dataclass
class AIBotConfig:
    # ---- Blockchain ----
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    chain: str = os.getenv("CHAIN", "ethereum")
    poll_interval: float = float(os.getenv("POLL_INTERVAL", "1.0"))

    # ---- Risk limits ----
    min_profit_eth: float = float(os.getenv("MIN_PROFIT_ETH", "0.002"))
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))
    trade_amount_eth: float = float(os.getenv("TRADE_AMOUNT", "0.1"))
    skip_on_bearish: bool = os.getenv("SKIP_ON_BEARISH", "true").lower() == "true"

    # ---- Price predictor (LSTM) ----
    price_predictor_input_size: int = int(os.getenv("PP_INPUT_SIZE", "16"))
    price_predictor_hidden_size: int = int(os.getenv("PP_HIDDEN_SIZE", "128"))
    price_predictor_num_layers: int = int(os.getenv("PP_NUM_LAYERS", "2"))
    price_predictor_seq_len: int = int(os.getenv("PP_SEQ_LEN", "30"))
    price_predictor_checkpoint: str = os.getenv(
        "PP_CHECKPOINT", "checkpoints/price_predictor.pt"
    )

    # ---- Opportunity classifier (MLP) ----
    classifier_feature_dim: int = int(os.getenv("CLF_FEATURE_DIM", "32"))
    classifier_checkpoint: str = os.getenv(
        "CLF_CHECKPOINT", "checkpoints/opp_classifier.pkl"
    )

    # ---- RL agent (PPO) ----
    rl_agent_checkpoint: str = os.getenv("RL_CHECKPOINT", "checkpoints/rl_agent")
    rl_total_timesteps: int = int(os.getenv("RL_TOTAL_TIMESTEPS", "100000"))
    update_every_n_steps: int = int(os.getenv("RL_UPDATE_EVERY", "500"))

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
            raise ValueError(f"Unknown chain: {self.chain!r}")
        return CHAINS[self.chain]

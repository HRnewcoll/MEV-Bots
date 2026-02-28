"""Configuration for the Python EVM arbitrage bot."""

import os
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Chain configurations
# ---------------------------------------------------------------------------

CHAINS: Dict[str, Dict] = {
    "ethereum": {
        "chain_id": 1,
        "rpc_url": os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com"),
        "ws_url": os.getenv("ETH_WS_URL", ""),
        "native_symbol": "ETH",
        "block_time": 12,
        "dexs": {
            "uniswap_v2": {
                "router": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "factory": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
            },
            "uniswap_v3": {
                "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
                "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
                "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
            },
            "sushiswap": {
                "router": "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
                "factory": "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac",
            },
        },
    },
    "bsc": {
        "chain_id": 56,
        "rpc_url": os.getenv("BSC_RPC_URL", "https://bsc-dataseed.binance.org/"),
        "ws_url": os.getenv("BSC_WS_URL", ""),
        "native_symbol": "BNB",
        "block_time": 3,
        "dexs": {
            "pancakeswap_v2": {
                "router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
                "factory": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
            },
            "biswap": {
                "router": "0x3a6d8cA21D1CF76F653A67577FA0D27453350dD8",
                "factory": "0x858E3312ed3A876947EA49d572A7C42DE08af7EE",
            },
        },
    },
    "polygon": {
        "chain_id": 137,
        "rpc_url": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
        "ws_url": os.getenv("POLYGON_WS_URL", ""),
        "native_symbol": "MATIC",
        "block_time": 2,
        "dexs": {
            "quickswap": {
                "router": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
                "factory": "0x5757371414417b8C6CAad45bAeF941aBc7d3Ab32",
            },
            "sushiswap": {
                "router": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
                "factory": "0xc35DADB65012eC5796536bD9864eD8773aBc74C4",
            },
        },
    },
    "avalanche": {
        "chain_id": 43114,
        "rpc_url": os.getenv("AVAX_RPC_URL", "https://api.avax.network/ext/bc/C/rpc"),
        "ws_url": os.getenv("AVAX_WS_URL", ""),
        "native_symbol": "AVAX",
        "block_time": 2,
        "dexs": {
            "traderjoe": {
                "router": "0x60aE616a2155Ee3d9A68541Ba4544862310933d4",
                "factory": "0x9Ad6C38BE94206cA50bb0d90783181662f0Cfa10",
            },
            "pangolin": {
                "router": "0xE54Ca86531e17Ef3616d22Ca28b0D458b6C89106",
                "factory": "0xefa94DE7a4656D787667C749f7E1223D71E9FD88",
            },
        },
    },
    "arbitrum": {
        "chain_id": 42161,
        "rpc_url": os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"),
        "ws_url": os.getenv("ARBITRUM_WS_URL", ""),
        "native_symbol": "ETH",
        "block_time": 1,
        "dexs": {
            "uniswap_v3": {
                "quoter": "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6",
                "router": "0xE592427A0AEce92De3Edee1F18E0157C05861564",
                "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
            },
            "camelot": {
                "router": "0xc873fEcbd354f5A56E00E710B90EF4201db2448d",
                "factory": "0x6EcCab422D763aC031210895C81787E87B43A652",
            },
        },
    },
}

# Uniswap V2–style pair ABI (minimal)
UNISWAP_V2_PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "_reserve0", "type": "uint112"},
            {"name": "_reserve1", "type": "uint112"},
            {"name": "_blockTimestampLast", "type": "uint32"},
        ],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function",
    },
]

UNISWAP_V2_ROUTER_ABI = [
    {
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
        "type": "function",
    },
    {
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "path", "type": "address[]"},
        ],
        "name": "getAmountsOut",
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
        "type": "function",
    },
]

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


@dataclass
class BotConfig:
    """Runtime configuration for the arbitrage bot."""

    # Wallet
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))

    # Chain to operate on
    chain: str = "ethereum"

    # Minimum profit in USD before executing a trade
    min_profit_usd: float = float(os.getenv("MIN_PROFIT_USD", "10.0"))

    # Maximum gas price in Gwei
    max_gas_price_gwei: float = float(os.getenv("MAX_GAS_PRICE_GWEI", "100"))

    # Slippage tolerance in basis points (1 bps = 0.01%)
    slippage_bps: int = int(os.getenv("SLIPPAGE_BPS", "50"))

    # Amount of base token to use per trade (in token units, e.g. WETH)
    trade_amount: float = float(os.getenv("TRADE_AMOUNT", "0.1"))

    # Token pairs to monitor  (token_a_address, token_b_address)
    token_pairs: List[tuple] = field(default_factory=list)

    # Poll interval in seconds when not using websockets
    poll_interval: float = 1.0

    # Flashbots relay (optional — leave empty to broadcast normally)
    flashbots_relay_url: str = field(
        default_factory=lambda: os.getenv(
            "FLASHBOTS_RELAY_URL", "https://relay.flashbots.net"
        )
    )
    flashbots_signer_key: str = field(
        default_factory=lambda: os.getenv("FLASHBOTS_SIGNER_KEY", "")
    )

    def chain_config(self) -> Dict:
        """Return the configuration dict for the active chain."""
        if self.chain not in CHAINS:
            raise ValueError(f"Unknown chain: {self.chain!r}. Available: {list(CHAINS)}")
        return CHAINS[self.chain]

# MEV Bots

A comprehensive collection of **Maximal Extractable Value (MEV)** bots implemented in Python and Rust, supporting multiple blockchain networks.

> ⚠️ **Disclaimer:** These implementations are provided for **educational and research purposes only**. MEV extraction can harm other users (e.g., sandwich attacks). Always understand the ethical and legal implications before deploying any bot. Never deploy on mainnet without thorough testing.

---

## What is MEV?

Maximal Extractable Value (MEV) refers to the maximum value that can be extracted from block production in excess of the standard block reward and gas fees by including, excluding, or reordering transactions within a block.

## Bot Types

| Bot | Language | Chains | Strategy |
|-----|----------|--------|----------|
| [Arbitrage Bot](python/arbitrage_bot/) | Python | EVM (ETH, BSC, Polygon, Avalanche, Arbitrum, Optimism) | Cross-DEX price arbitrage |
| [Sandwich Bot](python/sandwich_bot/) | Python | EVM | Front-run + back-run victim swaps |
| [Liquidation Bot](python/liquidation_bot/) | Python | EVM (Aave, Compound, dYdX) | Liquidate undercollateralized positions |
| [Flash Loan Bot](python/flash_loan_bot/) | Python | EVM (Aave v3) | Flash loan–powered atomic arbitrage |
| [Arbitrage Bot](rust/arbitrage_bot/) | Rust | EVM | High-performance cross-DEX arbitrage |
| [Sandwich Bot](rust/sandwich_bot/) | Rust | EVM | High-performance sandwich attacks |
| [Solana MEV Bot](rust/solana_bot/) | Rust | Solana | Arbitrage on Raydium / Orca |
| [Flash Loan Contract](contracts/) | Solidity | EVM | On-chain flash loan arbitrage |

---

## Directory Structure

```
MEV-Bots/
├── python/
│   ├── arbitrage_bot/       # Cross-DEX arbitrage (web3.py)
│   ├── sandwich_bot/        # Sandwich attack bot
│   ├── liquidation_bot/     # DeFi lending liquidations
│   └── flash_loan_bot/      # Flash loan arbitrage
├── rust/
│   ├── arbitrage_bot/       # High-performance arbitrage (ethers-rs)
│   ├── sandwich_bot/        # High-performance sandwich bot
│   └── solana_bot/          # Solana MEV (solana-client)
├── contracts/
│   ├── interfaces/          # Solidity interfaces
│   ├── FlashLoanArbitrage.sol
│   └── MevHelper.sol
└── scripts/
    ├── setup.sh             # Environment setup script
    └── deploy.sh            # Contract deployment script
```

---

## Quick Start

### Python Bots

```bash
# Install Python dependencies
cd python/arbitrage_bot
pip install -r requirements.txt

# Copy and edit configuration
cp .env.example .env
# Edit .env with your RPC URL and private key

# Run the bot
python bot.py
```

### Rust Bots

```bash
# Build all Rust bots
cd rust/arbitrage_bot
cargo build --release

# Run the bot
./target/release/arbitrage_bot
```

### Solana Bot

```bash
cd rust/solana_bot
cargo build --release
./target/release/solana_bot
```

### Smart Contracts

```bash
# Install Foundry
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Compile contracts
cd contracts
forge build

# Deploy (set environment variables first)
bash ../scripts/deploy.sh
```

---

## Supported Networks

### EVM Chains
| Chain | Chain ID | DEXs |
|-------|----------|------|
| Ethereum | 1 | Uniswap V2/V3, SushiSwap, Curve, Balancer |
| BSC | 56 | PancakeSwap V2/V3, BiSwap |
| Polygon | 137 | QuickSwap, SushiSwap, Uniswap V3 |
| Avalanche | 43114 | Trader Joe, Pangolin, SushiSwap |
| Arbitrum | 42161 | Uniswap V3, SushiSwap, Camelot |
| Optimism | 10 | Uniswap V3, Velodrome |
| Base | 8453 | Uniswap V3, BaseSwap, Aerodrome |

### Non-EVM Chains
| Chain | Language | DEXs |
|-------|----------|------|
| Solana | Rust | Raydium, Orca |

---

## Features

- **Multi-chain support** — One codebase, many networks
- **Gas optimization** — Dynamic gas pricing with EIP-1559 support
- **Mempool monitoring** — Real-time pending transaction tracking
- **Flashbots / MEV-Boost** — Private transaction bundles to avoid front-running
- **Flash loans** — Capital-efficient arbitrage with Aave V3
- **Profit simulation** — Off-chain simulation before on-chain execution
- **Risk management** — Configurable slippage, max gas, and position limits
- **Async execution** — High-throughput non-blocking I/O

---

## Environment Variables

```dotenv
# RPC endpoints
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY
ETH_WS_URL=wss://mainnet.infura.io/ws/v3/YOUR_KEY
BSC_RPC_URL=https://bsc-dataseed.binance.org/
POLYGON_RPC_URL=https://polygon-rpc.com

# Wallet
PRIVATE_KEY=0xYOUR_PRIVATE_KEY

# Flashbots
FLASHBOTS_RELAY_URL=https://relay.flashbots.net
FLASHBOTS_SIGNER_KEY=0xYOUR_FLASHBOTS_SIGNER_KEY

# Solana
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_KEYPAIR_PATH=/path/to/keypair.json

# Settings
MIN_PROFIT_USD=10.0
MAX_GAS_PRICE_GWEI=100
SLIPPAGE_BPS=50
```

---

## Security

- Never commit your private keys or `.env` file
- Use hardware wallets or KMS-managed keys in production
- Always test on testnets (Sepolia, Mumbai, BSC Testnet) first
- Monitor your bots continuously — network conditions change
- Set strict profit thresholds to avoid negative ROI trades

---

## License

MIT License — see [LICENSE](LICENSE) for details.
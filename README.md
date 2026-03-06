# MEV Bots

A comprehensive collection of **Maximal Extractable Value (MEV)** bots implemented in Python and Rust, supporting multiple blockchain networks — including **AI-powered bots** that use machine learning and reinforcement learning to identify and execute opportunities.

> ⚠️ **Disclaimer:** These implementations are provided for **educational and research purposes only**. MEV extraction can harm other users (e.g., sandwich attacks). Always understand the ethical and legal implications before deploying any bot. Never deploy on mainnet without thorough testing.

---

## ⚡ Quick Start — 3 Steps

> **No funds or private key needed for simulation mode!**

### 🐳 Option A — Docker (zero Python install required)

```bash
docker compose up          # builds image and starts dashboard
# Then open: http://localhost:5000
```

Stop with `Ctrl+C`, or run in background with `docker compose up -d`.

### 🐍 Option B — Python directly

### Step 1 — Install Python dependencies (once)

```bash
pip install -r requirements.txt
```

Or use the setup script (also sets up Rust bots and Solidity):

```bash
bash scripts/setup.sh --quick   # Python + dashboard only
bash scripts/setup.sh           # everything including Rust + Foundry
```

### Step 2 — Launch the dashboard

```bash
python run.py                   # Linux / macOS / Windows
bash start.sh                   # Linux / macOS shortcut
# or double-click start.bat     # Windows shortcut
```

The dashboard automatically opens in your browser at **http://127.0.0.1:5000**.

### Step 3 — Start a simulation

1. Open the **Simulation** tab
2. Choose a **chain** (Ethereum, BSC, Polygon)
3. Choose **strategies** (Arbitrage, Sandwich, Liquidation)
4. Click **▶ Start**

Watch real-time P&L, charts, and trade history update live — all without spending a penny.

#### Other launch options

```bash
python run.py --port 8080          # custom port
python run.py --host 0.0.0.0       # accessible from other devices on your LAN
python run.py --no-browser         # don't auto-open browser
python run.py --debug              # hot-reload mode for development
python run.py --help               # full option list
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

98 tests covering the simulation engine, paper wallet, trade recorder, market data math,
and all Flask API endpoints — all passing with no network calls required.

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
| [**AI MEV Bot**](python/ai_bot/) | **Python** | **EVM** | **LSTM price predictor + MLP classifier + PPO RL agent** |
| [Arbitrage Bot](rust/arbitrage_bot/) | Rust | EVM | High-performance cross-DEX arbitrage |
| [Sandwich Bot](rust/sandwich_bot/) | Rust | EVM | High-performance sandwich attacks |
| [Solana MEV Bot](rust/solana_bot/) | Rust | Solana | Arbitrage on Raydium / Orca |
| [**AI MEV Bot**](rust/ai_bot/) | **Rust** | **EVM** | **ONNX Runtime inference — runs exported PyTorch/SB3 models** |
| [Flash Loan Contract](contracts/) | Solidity | EVM | On-chain flash loan arbitrage |

---

## 🤖 AI-Powered MEV Bots

The AI bots combine three machine-learning components to outperform simple rule-based strategies:

### 1 — LSTM Price Predictor (`python/ai_bot/models/price_predictor.py`)
- **Architecture:** Stacked LSTM with LayerNorm + MLP head
- **Input:** Rolling window of OHLCV candles + on-chain signals (gas price, block time, mempool size)
- **Output:** 3-class direction prediction: `down` / `flat` / `up`
- **Use:** Pre-filter opportunities — skip entire blocks when a bearish signal is detected to protect capital

### 2 — Opportunity Classifier (`python/ai_bot/models/opportunity_classifier.py`)
- **Architecture:** MLP (PyTorch) or GradientBoostingClassifier (scikit-learn fallback)
- **Input:** 32-dimensional transaction feature vector (value, gas price, input selector, nonce, etc.)
- **Output:** 4-class label: `no_mev` / `sandwich_target` / `arb_trigger` / `liquidation_target`
- **Use:** Rank thousands of pending transactions per second to focus attention on the most profitable ones

### 3 — RL Agent (`python/ai_bot/models/rl_agent.py`)
- **Algorithm:** Proximal Policy Optimisation (PPO) via stable-baselines3
- **State:** 7-dimensional observation (gas, ETH price, mempool size, opportunities detected, block epoch, wallet balance)
- **Actions:** `do_nothing` / `execute_arb` / `execute_sandwich` / `execute_liquidation`
- **Reward:** Realised on-chain profit in ETH after gas costs
- **Use:** Learn an optimal strategy that balances opportunity size, gas cost, and risk across all MEV types

### Rust AI Bot (`rust/ai_bot/`)
Runs the same models at near-zero latency using **ONNX Runtime** (`ort` crate):
- Export trained PyTorch models to `.onnx` format once
- Load and run inference in Rust without any Python dependency
- Falls back to hand-crafted heuristics if model files are missing

```bash
# Export models from Python to ONNX
cd python/ai_bot
python - <<'EOF'
import torch
from models.price_predictor import PricePredictor
m = PricePredictor(); m.load("checkpoints/price_predictor.pt")
dummy = torch.randn(1, 30, 16)
torch.onnx.export(m.model, dummy, "../../rust/ai_bot/models/price_predictor.onnx",
                  input_names=["input"], output_names=["logits"])
EOF

# Build and run the Rust AI bot
cd rust/ai_bot
cargo build --release
./target/release/ai_bot
```

### Training the AI Models

```bash
cd python/ai_bot

# 1. Train the price predictor (requires OHLCV CSV data in data/ohlcv/)
python training/train_price_model.py --epochs 50 --checkpoint checkpoints/price_predictor.pt

# 2. Train the opportunity classifier (requires labelled mempool data)
python training/train_classifier.py --epochs 30 --checkpoint checkpoints/opp_classifier.pkl

# 3. Train the RL agent in simulation (no real funds needed)
python training/train_rl_agent.py --total-timesteps 200000 --checkpoint checkpoints/rl_agent

# 4. Run the live AI bot
python bot.py
```

---

## 🧪 Simulation Mode — Test Without Funds

Run any strategy against **real mainnet data** without sending a single transaction.
The simulation engine uses read-only RPC calls to fetch live prices and reserve data,
then applies the exact same Uniswap V2 math the bots use — so results accurately reflect
what would have happened on-chain.

```bash
cd simulator
pip install -r requirements.txt

# Run a quick arbitrage simulation from the command line
python -c "
import asyncio, sys
sys.path.insert(0, '..')
from simulator.engine import SimulationEngine, SimConfig

cfg = SimConfig(
    chain='ethereum',
    strategies=['arbitrage', 'sandwich', 'liquidation'],
    initial_eth=10.0,
    trade_amount_eth=0.1,
    min_profit_eth=0.0005,
    poll_interval_s=2.0,
)
asyncio.run(SimulationEngine(cfg).run(max_steps=20))
"
```

Key features of the simulation engine:
- **Real data** — reads live reserves from on-chain via read-only RPC (no private key needed)
- **LRU caching** — pair addresses cached permanently; reserves cached per block to minimise RPC calls
- **Batch fetching** — parallel `asyncio.gather` for all DEX quotes in a single event-loop tick
- **Paper wallet** — tracks virtual balances, per-trade P&L, win rate, and gas spend
- **Persistent history** — all simulated trades appended to `simulator_data/trades.jsonl` (JSONL)
- **CSV export** — one-click export from the UI or via `TradeRecorder.export_csv()`

---

## 🖥️ Web Dashboard (GUI)

A clean dark-theme web dashboard built with **Flask + Chart.js** — no Node.js build
step required, no React, just plain Python + HTML/CSS/JS.

```bash
# Install UI dependencies
cd ui
pip install -r requirements.txt

# Start the dashboard (default: http://127.0.0.1:5000)
python app.py

# Custom host/port
UI_HOST=0.0.0.0 UI_PORT=8080 python app.py
```

### Dashboard features

| Tab | What it shows |
|-----|---------------|
| **Dashboard** | Live P&L cards · Cumulative P&L line chart · Trades-by-Strategy doughnut · Gas history · Recent trades table |
| **Simulation** | Configuration panel (chain, RPC, balances, strategies, gas limits) · Start / Pause / Stop controls · Live metrics grid (step, block, cache sizes, win rate) |
| **Trades** | Full paginated trade history · Per-strategy aggregate cards · CSV export |
| **Logs** | Real-time SSE event stream · Auto-scroll · Clear button |

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/status` | Engine status + wallet stats |
| `GET`  | `/api/trades` | Recent simulated trades (JSON) |
| `GET`  | `/api/stats`  | Aggregate P&L from disk history |
| `GET`  | `/api/events` | Recent event log |
| `GET`  | `/api/chains` | Available chains + DEXs |
| `POST` | `/api/start`  | Start simulation with config body |
| `POST` | `/api/stop`   | Stop simulation |
| `POST` | `/api/pause`  | Pause / resume toggle |
| `GET`  | `/api/stream` | SSE real-time event stream |
| `GET`  | `/api/export/csv` | Download trades as CSV |

---

## Directory Structure

```
MEV-Bots/
├── simulator/               # 🧪 Paper-trading simulation engine
│   ├── engine.py            #   Main simulation loop (all strategies)
│   ├── paper_wallet.py      #   Virtual balance + P&L tracker
│   ├── market_data.py       #   Read-only live data fetcher (cached)
│   ├── recorder.py          #   JSONL trade history + CSV export
│   └── requirements.txt
├── ui/                      # 🖥️ Flask web dashboard
│   ├── app.py               #   Flask server + REST API + SSE stream
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html       #   Dashboard, Simulation, Trades, Logs tabs
│   └── static/
│       ├── css/dashboard.css
│       └── js/dashboard.js  #   Charts, SSE client, controls
├── python/
│   ├── arbitrage_bot/       # Cross-DEX arbitrage (web3.py)
│   ├── sandwich_bot/        # Sandwich attack bot
│   ├── liquidation_bot/     # DeFi lending liquidations
│   ├── flash_loan_bot/      # Flash loan arbitrage
│   └── ai_bot/              # AI-powered MEV bot
│       ├── bot.py           #   Main orchestrator
│       ├── config.py
│       ├── models/
│       │   ├── price_predictor.py      # LSTM price predictor
│       │   ├── opportunity_classifier.py # MLP / GBT classifier
│       │   └── rl_agent.py             # PPO RL agent (stable-baselines3)
│       └── training/
│           ├── train_price_model.py
│           ├── train_classifier.py
│           └── train_rl_agent.py
├── rust/
│   ├── arbitrage_bot/       # High-performance arbitrage (ethers-rs)
│   ├── sandwich_bot/        # High-performance sandwich bot
│   ├── solana_bot/          # Solana MEV (solana-client)
│   └── ai_bot/              # AI MEV bot (ONNX Runtime)
│       └── src/
│           ├── main.rs
│           ├── inference.rs # ONNX model loading + inference
│           ├── scanner.rs   # Market data + mempool scanning
│           └── executor.rs  # Action execution
├── contracts/
│   ├── interfaces/          # Solidity interfaces
│   ├── FlashLoanArbitrage.sol
│   └── MevHelper.sol        # On-chain simulation helpers
└── scripts/
    ├── setup.sh             # Environment setup script
    └── deploy.sh            # Contract deployment script
```

---

## Running Individual Bots (Live / Production Mode)

> ⚠️ Real bots send transactions and spend gas. Always simulate first.

### Python Bots

```bash
# Install Python dependencies for a specific bot
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
- **AI-powered** — LSTM price prediction, MLP opportunity classification, PPO reinforcement learning
- **ONNX inference in Rust** — Export trained models from Python; run at nanosecond latency in Rust
- **🧪 Simulation mode** — Paper-trade using real live mainnet data with zero capital required
- **🖥️ Web dashboard** — Full GUI with live charts, P&L tracking, trade history, and SSE log streaming
- **Gas optimization** — Dynamic gas pricing with EIP-1559 support
- **Mempool monitoring** — Real-time pending transaction tracking
- **Flashbots / MEV-Boost** — Private transaction bundles to avoid front-running
- **Flash loans** — Capital-efficient arbitrage with Aave V3
- **Profit simulation** — Off-chain simulation before on-chain execution
- **Risk management** — Configurable slippage, max gas, and position limits
- **Async execution** — High-throughput non-blocking I/O
- **Caching layer** — LRU pair address cache + TTL reserve cache to minimise RPC calls

---

## Environment Variables

```dotenv
# RPC endpoints
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY
ETH_WS_URL=wss://mainnet.infura.io/ws/v3/YOUR_KEY
BSC_RPC_URL=https://bsc-dataseed.binance.org/
POLYGON_RPC_URL=https://polygon-rpc.com
AVAX_RPC_URL=https://api.avax.network/ext/bc/C/rpc
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc

# Wallet
PRIVATE_KEY=0xYOUR_PRIVATE_KEY

# Flashbots
FLASHBOTS_RELAY_URL=https://relay.flashbots.net
FLASHBOTS_SIGNER_KEY=0xYOUR_FLASHBOTS_SIGNER_KEY

# Solana
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
SOLANA_KEYPAIR_PATH=/path/to/keypair.json

# General settings
MIN_PROFIT_USD=10.0
MAX_GAS_PRICE_GWEI=100
SLIPPAGE_BPS=50
TRADE_AMOUNT=0.1

# AI bot
PP_CHECKPOINT=checkpoints/price_predictor.pt
CLF_CHECKPOINT=checkpoints/opp_classifier.pkl
RL_CHECKPOINT=checkpoints/rl_agent
SKIP_ON_BEARISH=true
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

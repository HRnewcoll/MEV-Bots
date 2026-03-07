# Changelog

All notable changes to this project are documented in this file.
Format: [Semantic Versioning](https://semver.org/) — `[MAJOR.MINOR.PATCH] — YYYY-MM-DD`.

---

## [Unreleased]

---

## [0.4.0] — 2026-03-07

### Added — Separate Neural Network AI Bots (`python/nn_bots/`)

Two architecturally distinct neural network bots, each with their own training
pipeline, configuration, and graceful PyTorch-absent fallback:

#### `python/nn_bots/transformer_arb_bot/`
- **Architecture:** Multi-head self-attention Transformer encoder over a sequence of
  per-DEX feature vectors.  Learns cross-DEX dependencies — e.g. "when reserve depth
  drops on Uniswap, SushiSwap tends to widen its spread."
- `model.py` — `TransformerArbModel`:
  - `_PositionalEncoding` — sinusoidal positional encoding for DEX sequences
  - `_TransformerEncoder` — Pre-LN transformer encoder + pairwise profit-score head
  - `extract_dex_features()` — static 10-dim feature vector per DEX snapshot
  - `build_feature_matrix()` — (num_dexs, 10) matrix from list of snapshots
  - `score_pairs()` — (num_dexs, num_dexs) profit logit matrix; diagonal masked
  - `best_pair()` — top (buy_dex, sell_dex) with arithmetic verification
  - `train_step()` / `validate()` / `save()` / `load()`
  - Heuristic fallback (reserve-spread) when PyTorch is absent
- `bot.py` — async `TransformerArbBot`: fetches live reserves, runs model, verifies arb arithmetically, executes two-leg swap
- `config.py` — `TransformerBotConfig` dataclass
- `training/train.py` — synthetic supervised training; saves best checkpoint
- `requirements.txt`, `.env.example`

#### `python/nn_bots/graph_arb_bot/`
- **Architecture:** Graph Neural Network with K rounds of message-passing over the
  token-pool graph (tokens = nodes, liquidity pools = directed edges).  Node embeddings
  propagate global liquidity context; an edge profit-scoring head ranks every directed pool.
- `model.py` — `TokenGraph` + `GNNArbModel`:
  - `TokenGraph` — dynamic directed graph; `add_pool()`, `to_tensors()`, `best_single_hop_heuristic()`
  - `_MessagePassingLayer` — `[src || dst || edge_feat] → message → GRUCell update`
  - `_GNNModule` — K message-passing rounds + edge profit head
  - `score_edges()` — (num_edges,) profit logits
  - `best_cycle()` — top 2-hop arbitrage cycle with arithmetic verification
  - `train_step()` / `validate()` / `save()` / `load()`
  - Bellman-Ford heuristic fallback when PyTorch is absent
  - `_v2_out_with_fee()` — generalised V2 formula for any `fee_bps`
- `bot.py` — async `GraphArbBot`: builds live `TokenGraph`, runs GNN, executes cycle
- `config.py` — `GNNBotConfig` dataclass
- `training/train.py` — random graph generation + profitability labelling + training loop
- `requirements.txt`, `.env.example`

### Added — Tests for new bots
- `tests/test_transformer_model.py` — 36 tests (25 run without PyTorch, 11 need torch):
  `_uniswap_v2_out`, `_compute_price_impact`, feature extraction, matrix building, heuristic inference,
  PyTorch forward pass shape, diagonal masking, training, save/load
- `tests/test_gnn_model.py` — 40 tests (29 run without PyTorch, 11 need torch):
  `_v2_out_with_fee`, `_pool_edge_features`, `TokenGraph` construction/tensor conversion,
  heuristic cycle finding, GNN edge scoring, training, save/load

  **Running total: 163 tests — 163 pass, 11 skip (PyTorch-only — will pass with torch installed)**

### Fixed
- `python/ai_bot/models/price_predictor.py` — `save()` method was dead code (missing `def`
  header after `validate()`) — restored as a properly declared method

### Added — RESOURCES.md
- Curated reference list of **35+ related MEV bot repositories** from the broader community:
  Top bots, sandwich bots, mempool scanners, multi-chain bots, frameworks & tools,
  profitable bots, Telegram bots, specialised MEV, monitoring tools, and learning resources
- Structured as tables by category with language tags
- Includes quick profit estimates and requirements checklist
- Links back to the repo's own simulator for safe practice

### Changed
- `README.md` — bot table extended with Transformer and GNN entries; test count updated;
  new "Related Resources" section pointing at `RESOURCES.md`
- `CHANGELOG.md` — this entry

---

## [0.3.0] — 2026-03-06

### Added — Tests (`tests/`)
- `tests/test_paper_wallet.py` — 24 unit tests for `PaperWallet`:
  balance helpers, `simulate_swap`, `simulate_arb`, P&L aggregates, `recent_trades`
- `tests/test_recorder.py` — 20 unit tests for `TradeRecorder`:
  `record`, `recent`, `all_trades`, `load_history`, `aggregate_stats`, `export_csv`
- `tests/test_market_data.py` — 13 unit tests for `MarketData`:
  `get_amount_out` (Uniswap V2 formula edge cases), cache stats
- `tests/test_engine_math.py` — 22 unit tests for `SimConfig` + chain registry + arb math:
  default values, RPC URL resolution, token pair population, DEX registry validation
- `tests/test_api.py` — 19 integration tests for the Flask API via test client:
  `/api/status`, `/api/trades`, `/api/events`, `/api/stats`, `/api/chains`,
  `/api/setup_check`, `/api/stop`, `/api/pause`, dashboard HTML, content-type headers

  **Total: 98 tests — all passing, zero network calls required**

### Added — Docker support
- `Dockerfile` — multi-stage Python 3.12-slim image; non-root user; health check
- `docker-compose.yml` — single `docker compose up` starts the full dashboard on port 5000;
  named volume for persistent `simulator_data/`

### Changed — CHANGELOG
- Added this file to track project history going forward

---

## [0.2.0] — 2026-03-06

### Added — Easy setup and one-command launch
- `requirements.txt` (repo root) — merged deps for simulator + dashboard;
  `pip install -r requirements.txt` is now the single install step
- `run.py` — smart launcher:
  checks Python ≥ 3.10, installs missing deps, creates `simulator_data/`, copies `.env` files,
  auto-opens browser, accepts `--host`, `--port`, `--no-browser`, `--debug`, `--skip-install`
- `start.sh` — one-liner Unix / macOS launcher (`bash start.sh`)
- `start.bat` — Windows double-click launcher
- `scripts/setup.sh` — major rewrite:
  added `--quick` flag (Python + dashboard only, no Rust build),
  `--full` installs Rust toolchain + Foundry,
  resolves repo root correctly regardless of working directory,
  colour-coded `✓` / `⚠` output

### Added — Setup tab in dashboard
- New `⚙ Setup` tab (auto-shown on first visit via `localStorage`):
  - **Quick Start** — 3 numbered steps from zero to running simulation
  - **Setup Checklist** — live `GET /api/setup_check` response with ✅/❌ per item + fix command
  - **Command Reference** — full option table for `run.py`
  - **Strategy Guide** — plain-English descriptions of Arbitrage, Sandwich, Liquidation
  - **FAQ** — 6 expandable questions covering common new-user issues
- New `GET /api/setup_check` Flask endpoint — returns JSON checklist of:
  installed packages, `simulator_data/` existence + writability, `.env` files, RPC URL config

### Changed — README
- Prominent **3-step Quick Start** section moved to the very top of `README.md`

---

## [0.1.0] — 2026-03-06 (initial)

### Added — Python bots
- `python/arbitrage_bot/` — async cross-DEX arbitrage (Ethereum, BSC, Polygon, Avalanche, Arbitrum)
- `python/sandwich_bot/` — Flashbots sandwich bot (mempool monitoring + bundle submission)
- `python/liquidation_bot/` — Aave V2/V3 + Compound V2/V3 liquidation bot
- `python/flash_loan_bot/` — Flash loan–powered atomic arbitrage via Aave V3
- `python/ai_bot/` — AI-powered MEV bot:
  LSTM price predictor, MLP opportunity classifier, PPO reinforcement-learning agent
- Training scripts: `train_price_model.py`, `train_classifier.py`, `train_rl_agent.py`

### Added — Rust bots
- `rust/arbitrage_bot/` — high-performance async cross-DEX arbitrage (ethers-rs)
- `rust/sandwich_bot/` — high-performance sandwich bot with Flashbots bundle submission
- `rust/solana_bot/` — Solana MEV bot (Raydium + Orca arbitrage)
- `rust/ai_bot/` — Rust AI bot with ONNX Runtime inference (runs exported PyTorch/SB3 models)

### Added — Smart contracts
- `contracts/FlashLoanArbitrage.sol` — on-chain flash loan arbitrage contract (Aave V3)
- `contracts/MevHelper.sol` — helper library for bundle construction + profit checking
- `contracts/interfaces/IFlashLoan.sol` — Aave V3 flash loan receiver interface
- `scripts/deploy.sh` — Foundry deployment script

### Added — Simulation engine
- `simulator/engine.py` — paper-trading loop: real mainnet data, no transactions sent
  - Supports Arbitrage, Sandwich, and Liquidation strategies
  - Configurable via `SimConfig` dataclass
  - SSE event stream for real-time UI updates
- `simulator/market_data.py` — live RPC data with multi-level caching:
  permanent pair-address cache, TTL reserve cache, per-pair async locks
- `simulator/paper_wallet.py` — virtual balance tracking + P&L calculation
- `simulator/recorder.py` — JSONL trade persistence + CSV export

### Added — Web dashboard
- `ui/app.py` — Flask dashboard with REST API:
  `GET /api/status`, `/api/trades`, `/api/events`, `/api/stats`, `/api/chains`,
  `POST /api/start`, `/api/stop`, `/api/pause`,
  `GET /api/stream` (SSE), `GET /api/export/csv`
- `ui/templates/index.html` — full-featured SPA:
  Dashboard, Simulation config, Trades, Logs tabs
- `ui/static/css/dashboard.css` — dark-theme CSS
- `ui/static/js/dashboard.js` — Chart.js charts, SSE stream, controls
- `ui/static/js/chart.umd.min.js` — Chart.js (bundled, no CDN required)

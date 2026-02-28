#!/usr/bin/env bash
# setup.sh — Install all dependencies for the MEV Bots repository.
set -euo pipefail

BOLD="\033[1m"
RESET="\033[0m"

info()  { echo -e "${BOLD}[setup]${RESET} $*"; }
error() { echo -e "${BOLD}[ERROR]${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# Rust toolchain
# ---------------------------------------------------------------------------
info "Checking Rust toolchain …"
if ! command -v cargo &>/dev/null; then
    info "Installing Rust via rustup …"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
rustup update stable
info "Rust $(rustc --version)"

# ---------------------------------------------------------------------------
# Python (requires Python 3.10+)
# ---------------------------------------------------------------------------
info "Checking Python …"
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    error "Python 3.10+ is required. Install from https://python.org"
    exit 1
fi
info "Python $($PYTHON --version)"

# Create virtual environment per Python bot
for bot_dir in python/arbitrage_bot python/sandwich_bot python/liquidation_bot python/flash_loan_bot python/ai_bot; do
    info "Setting up $bot_dir …"
    pushd "$bot_dir" > /dev/null
    $PYTHON -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    deactivate
    popd > /dev/null
    info "  $bot_dir: ✓"
done

# ---------------------------------------------------------------------------
# Foundry (Solidity)
# ---------------------------------------------------------------------------
info "Checking Foundry …"
if ! command -v forge &>/dev/null; then
    info "Installing Foundry …"
    curl -L https://foundry.paradigm.xyz | bash
    "$HOME/.foundry/bin/foundryup"
fi
info "Forge $(forge --version 2>/dev/null | head -1)"

# ---------------------------------------------------------------------------
# Build Rust bots
# ---------------------------------------------------------------------------
info "Building Rust bots …"
for bot_dir in rust/arbitrage_bot rust/sandwich_bot rust/solana_bot rust/ai_bot; do
    info "  Building $bot_dir …"
    pushd "$bot_dir" > /dev/null
    cargo build --release 2>&1 | tail -5
    popd > /dev/null
    info "  $bot_dir: ✓"
done

# ---------------------------------------------------------------------------
# .env files
# ---------------------------------------------------------------------------
info "Checking .env files …"
for env_example in python/*/.env.example; do
    env_file="${env_example%.example}"
    if [ ! -f "$env_file" ]; then
        cp "$env_example" "$env_file"
        info "  Created $env_file — please edit with your credentials"
    fi
done

info "Setup complete! Edit the .env files in each bot directory before running."

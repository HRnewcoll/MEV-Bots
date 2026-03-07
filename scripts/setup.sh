#!/usr/bin/env bash
# setup.sh — Install all dependencies for the MEV Bots repository.
#
# Quick start (just the dashboard + simulator, no Rust/Solidity):
#   bash scripts/setup.sh --quick
#
# Full setup (all Python bots + Rust bots + Solidity/Foundry):
#   bash scripts/setup.sh
#
# After setup, launch the dashboard with ONE command:
#   python run.py          (Linux / macOS)
#   bash start.sh          (Linux / macOS — same thing)
#   start.bat              (Windows — double-click)

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

QUICK=false
for arg in "$@"; do [[ "$arg" == "--quick" ]] && QUICK=true; done

info()  { echo -e "${BOLD}[setup]${RESET} $*"; }
ok()    { echo -e "${GREEN}  ✓${RESET} $*"; }
warn()  { echo -e "${YELLOW}  ⚠${RESET} $*"; }
error() { echo -e "${BOLD}[ERROR]${RESET} $*" >&2; }

# Resolve repo root (works whether run from repo root or scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo
echo -e "${BOLD}⚡ MEV Bot Setup${RESET}"
echo "========================================"
echo

# ---------------------------------------------------------------------------
# Python (requires Python 3.10+)
# ---------------------------------------------------------------------------
info "Checking Python …"
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [ -z "$PYTHON" ]; then
    error "Python 3.10+ is required. Install from https://python.org"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi
ok "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# Dashboard + Simulator (always installed — this is the quick path)
# ---------------------------------------------------------------------------
info "Installing dashboard + simulator dependencies …"
$PYTHON -m pip install --upgrade pip -q
$PYTHON -m pip install -r requirements.txt -q
ok "Dashboard + simulator dependencies installed"

# Create runtime directories
mkdir -p simulator_data
ok "simulator_data/ directory ready"

# ---------------------------------------------------------------------------
# .env files
# ---------------------------------------------------------------------------
info "Checking .env files …"
for env_example in python/*/.env.example; do
    env_file="${env_example%.example}"
    if [ ! -f "$env_file" ]; then
        cp "$env_example" "$env_file"
        warn "Created $env_file — edit with your RPC URL / private key"
    else
        ok "$env_file already exists"
    fi
done

if $QUICK; then
    echo
    ok "Quick setup complete!"
    echo
    echo -e "  ${BOLD}Start the dashboard:${RESET}"
    echo "    python run.py"
    echo "    bash start.sh"
    echo
    exit 0
fi

# ---------------------------------------------------------------------------
# Python bots (individual venvs)
# ---------------------------------------------------------------------------
info "Setting up individual Python bot environments …"
for bot_dir in python/arbitrage_bot python/sandwich_bot python/liquidation_bot python/flash_loan_bot python/ai_bot; do
    info "  $bot_dir …"
    pushd "$bot_dir" > /dev/null
    $PYTHON -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    deactivate
    popd > /dev/null
    ok "  $bot_dir"
done

# ---------------------------------------------------------------------------
# Rust toolchain
# ---------------------------------------------------------------------------
info "Checking Rust toolchain …"
if ! command -v cargo &>/dev/null; then
    info "Installing Rust via rustup …"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
rustup update stable 2>&1 | tail -2
ok "Rust $(rustc --version)"

# ---------------------------------------------------------------------------
# Build Rust bots
# ---------------------------------------------------------------------------
info "Building Rust bots (this may take a few minutes) …"
for bot_dir in rust/arbitrage_bot rust/sandwich_bot rust/solana_bot rust/ai_bot; do
    info "  Building $bot_dir …"
    pushd "$bot_dir" > /dev/null
    cargo build --release 2>&1 | tail -3
    popd > /dev/null
    ok "  $bot_dir"
done

# ---------------------------------------------------------------------------
# Foundry (Solidity)
# ---------------------------------------------------------------------------
info "Checking Foundry (Solidity compiler) …"
if ! command -v forge &>/dev/null; then
    info "Installing Foundry …"
    curl -L https://foundry.paradigm.xyz | bash
    "$HOME/.foundry/bin/foundryup"
fi
ok "Forge $(forge --version 2>/dev/null | head -1)"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
ok "Full setup complete!"
echo
echo -e "  ${BOLD}Next steps:${RESET}"
echo "  1. Edit the .env files in python/*/  with your RPC URL and private key"
echo "  2. Launch the dashboard:"
echo "       python run.py"
echo "       bash start.sh"
echo

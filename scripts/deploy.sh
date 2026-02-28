#!/usr/bin/env bash
# deploy.sh — Deploy the FlashLoanArbitrage and MevHelper contracts using Foundry.
#
# Prerequisites:
#   - Foundry installed (forge, cast)
#   - Environment variables set (see below)
#
# Required environment variables:
#   PRIVATE_KEY              Deployer private key
#   ETH_RPC_URL              RPC endpoint for the target chain
#   AAVE_POOL                Aave V3 Pool address for the chain
#
# Optional:
#   ETHERSCAN_API_KEY        For contract verification
#   CHAIN_ID                 Chain ID (used for verification)
set -euo pipefail

BOLD="\033[1m"
RESET="\033[0m"
info()  { echo -e "${BOLD}[deploy]${RESET} $*"; }
error() { echo -e "${BOLD}[ERROR]${RESET} $*" >&2; }

: "${PRIVATE_KEY:?PRIVATE_KEY is required}"
: "${ETH_RPC_URL:?ETH_RPC_URL is required}"
: "${AAVE_POOL:?AAVE_POOL is required (Aave V3 Pool address)}"

cd "$(dirname "$0")/../contracts"

info "Compiling contracts …"
forge build

info "Deploying FlashLoanArbitrage …"
FLASH_LOAN_ADDRESS=$(forge create FlashLoanArbitrage \
    --rpc-url "$ETH_RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --constructor-args "$AAVE_POOL" \
    --json | jq -r '.deployedTo')

info "FlashLoanArbitrage deployed at: $FLASH_LOAN_ADDRESS"

info "Deploying MevHelper …"
MEV_HELPER_ADDRESS=$(forge create MevHelper \
    --rpc-url "$ETH_RPC_URL" \
    --private-key "$PRIVATE_KEY" \
    --json | jq -r '.deployedTo')

info "MevHelper deployed at: $MEV_HELPER_ADDRESS"

# Optional: verify on Etherscan
if [ -n "${ETHERSCAN_API_KEY:-}" ]; then
    info "Verifying FlashLoanArbitrage on Etherscan …"
    forge verify-contract "$FLASH_LOAN_ADDRESS" FlashLoanArbitrage \
        --etherscan-api-key "$ETHERSCAN_API_KEY" \
        --chain "${CHAIN_ID:-1}" \
        --constructor-args "$(cast abi-encode 'constructor(address)' "$AAVE_POOL")" || true
fi

# Write addresses to a local file for use by the bots
cat > ../deployed_addresses.json <<EOF
{
  "flash_loan_arbitrage": "$FLASH_LOAN_ADDRESS",
  "mev_helper": "$MEV_HELPER_ADDRESS",
  "aave_pool": "$AAVE_POOL"
}
EOF

info "Addresses written to deployed_addresses.json"
info "Deployment complete!"

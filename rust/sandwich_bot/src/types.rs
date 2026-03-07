//! Shared types for the sandwich bot.

use ethers::types::{Address, Bytes, U256};

/// A decoded victim swap transaction.
#[derive(Debug, Clone)]
pub struct VictimSwap {
    pub tx_hash: String,
    pub router: Address,
    pub token_in: Address,
    pub token_out: Address,
    pub amount_in: U256,
    pub amount_out_min: U256,
    pub path: Vec<Address>,
    pub raw_tx: Bytes,
}

/// A profitable sandwich opportunity.
#[derive(Debug, Clone)]
pub struct SandwichOpportunity {
    pub victim: VictimSwap,
    pub front_run_amount: U256,
    pub estimated_profit_wei: i128,
    pub target_block: u64,
}

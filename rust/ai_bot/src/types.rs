//! Shared types for the AI MEV bot.

use serde::{Deserialize, Serialize};

/// A snapshot of the current market state.
#[derive(Debug, Clone, Default)]
pub struct MarketSnapshot {
    pub gas_price_gwei: f32,
    pub eth_price_usd: f32,
    pub pending_tx_count: u32,
    pub arb_opportunities: u32,
    pub liquidation_opportunities: u32,
    pub block_number: u64,
    pub wallet_balance_eth: f32,
}

/// Features extracted from a pending transaction.
#[derive(Debug, Clone)]
pub struct TxFeatures {
    pub hash: String,
    pub value_eth: f32,
    pub gas_price_gwei: f32,
    pub input_length: usize,
    pub selector: String,
    pub nonce: u64,
    pub gas_limit: u64,
    pub estimated_profit_wei: u128,
}

/// A classified MEV opportunity.
#[derive(Debug, Clone)]
pub struct MevOpportunity {
    pub tx_hash: String,
    pub opportunity_type: String,
    pub estimated_profit_wei: u128,
}

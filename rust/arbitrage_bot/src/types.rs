//! Shared types used across the arbitrage bot.

use ethers::types::{Address, U256};

/// A detected arbitrage opportunity.
#[derive(Debug, Clone)]
pub struct Opportunity {
    pub token_a: Address,
    pub token_b: Address,
    pub buy_dex: String,
    pub sell_dex: String,
    pub buy_router: Address,
    pub sell_router: Address,
    pub amount_in: U256,
    pub out_buy: U256,
    pub out_sell: U256,
    /// Gross profit in token_b units (before gas).
    pub profit_gross: U256,
}

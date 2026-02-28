//! Shared types for the Solana MEV bot.

use solana_sdk::pubkey::Pubkey;

/// A pool price quote.
#[derive(Debug, Clone)]
pub struct PoolQuote {
    pub pool: Pubkey,
    pub dex: String,
    pub amount_out: u64,
}

/// A profitable arbitrage opportunity between two Solana AMM pools.
#[derive(Debug, Clone)]
pub struct SolanaOpportunity {
    pub mint_in: Pubkey,
    pub mint_out: Pubkey,
    pub buy_pool: Pubkey,
    pub buy_dex: String,
    pub sell_pool: Pubkey,
    pub sell_dex: String,
    pub amount_in: u64,
    pub out_buy: u64,
    pub out_sell: u64,
    /// Gross profit in lamports (before transaction fees).
    pub estimated_profit_lamports: u64,
}

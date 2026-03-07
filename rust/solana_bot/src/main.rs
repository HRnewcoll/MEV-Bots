//! Solana MEV arbitrage bot.
//!
//! # Strategy
//! 1. Fetch token prices from Raydium and Orca AMM pools on Solana.
//! 2. When a profitable spread exists, build and submit a Jito bundle
//!    (Solana's equivalent of Flashbots) for atomic execution.
//!
//! # Pools monitored
//! - Raydium V4 AMM
//! - Orca Whirlpool

mod config;
mod dex;
mod executor;
mod types;

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use config::SolanaConfig;
use dex::SolanaScanner;
use executor::SolanaExecutor;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("solana_bot=info".parse()?))
        .init();

    let cfg = SolanaConfig::from_env()?;
    info!(rpc = %cfg.rpc_url, "SolanaBot starting");

    let scanner = SolanaScanner::new(&cfg)?;
    let executor = SolanaExecutor::new(&cfg)?;

    loop {
        for pair in &cfg.token_pairs.clone() {
            match scanner.find_opportunity(&pair.0, &pair.1, cfg.trade_amount_lamports).await {
                Ok(Some(opp)) => {
                    info!(
                        buy_pool = %opp.buy_pool,
                        sell_pool = %opp.sell_pool,
                        profit = %opp.estimated_profit_lamports,
                        "Arbitrage opportunity on Solana"
                    );
                    if let Err(e) = executor.execute(&opp).await {
                        error!("Execution error: {e}");
                    }
                }
                Ok(None) => {}
                Err(e) => error!("Scan error: {e}"),
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(cfg.poll_interval_ms)).await;
    }
}

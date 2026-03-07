//! High-performance EVM cross-DEX arbitrage bot.
//!
//! # Strategy
//! 1. Poll multiple Uniswap V2-style DEXs for token-pair prices.
//! 2. When a profitable spread is detected, execute a two-leg swap atomically.
//! 3. Optionally submit via Flashbots for private ordering.
//!
//! # Supported chains
//! Ethereum, BSC, Polygon, Avalanche, Arbitrum, Optimism, Base.

mod config;
mod dex;
mod executor;
mod types;

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use config::BotConfig;
use dex::DexScanner;
use executor::Executor;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise logging (set RUST_LOG=info to see output)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("arbitrage_bot=info".parse()?))
        .init();

    let cfg = BotConfig::from_env()?;
    info!(
        chain = %cfg.chain,
        wallet = %cfg.wallet_address(),
        "ArbitrageBot starting"
    );

    let scanner = DexScanner::new(&cfg).await?;
    let executor = Executor::new(&cfg).await?;

    loop {
        for (token_a, token_b) in &cfg.token_pairs {
            match scanner.find_opportunity(token_a, token_b, cfg.trade_amount_wei).await {
                Ok(Some(opp)) => {
                    info!(
                        buy_dex = %opp.buy_dex,
                        sell_dex = %opp.sell_dex,
                        profit = %opp.profit_gross,
                        "Arbitrage opportunity found"
                    );
                    if let Err(e) = executor.execute(&opp).await {
                        error!("Execution error: {e}");
                    }
                }
                Ok(None) => {}
                Err(e) => error!("Scan error for {token_a}/{token_b}: {e}"),
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(cfg.poll_interval_ms)).await;
    }
}

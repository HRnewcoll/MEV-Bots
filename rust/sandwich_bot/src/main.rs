//! High-performance EVM sandwich bot.
//!
//! # Strategy
//! 1. Subscribe to the public mempool via WebSocket.
//! 2. Decode pending Uniswap V2–style swap transactions.
//! 3. Simulate the price impact and estimate profit.
//! 4. If profitable, submit a Flashbots bundle:
//!    - Front-run tx (same direction as victim, higher gas)
//!    - Victim tx (unchanged)
//!    - Back-run tx (reverse direction, lower gas)
//!
//! ⚠️  WARNING: Sandwich attacks harm victim execution prices.
//!     This code is for educational purposes only.

mod config;
mod mempool;
mod bundle;
mod types;

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use config::SandwichConfig;
use mempool::MempoolWatcher;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("sandwich_bot=info".parse()?))
        .init();

    let cfg = SandwichConfig::from_env()?;
    info!(chain = %cfg.chain, "SandwichBot starting");

    let watcher = MempoolWatcher::new(cfg).await?;
    watcher.run().await
}

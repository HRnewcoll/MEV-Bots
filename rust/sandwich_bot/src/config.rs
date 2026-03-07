//! Sandwich bot configuration.

use anyhow::{anyhow, Result};
use ethers::types::{Address, U256};

#[derive(Debug, Clone)]
pub struct SandwichConfig {
    pub chain: String,
    pub rpc_url: String,
    pub ws_url: String,
    pub private_key: String,
    pub front_run_amount_wei: U256,
    pub max_gas_price_gwei: u64,
    pub gas_limit: u64,
    pub flashbots_relay_url: String,
    pub flashbots_signer_key: String,
    /// Routers to watch for swap transactions
    pub watched_routers: Vec<Address>,
}

impl SandwichConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let chain = std::env::var("CHAIN").unwrap_or_else(|_| "ethereum".into());
        let rpc_url = std::env::var("ETH_RPC_URL")
            .unwrap_or_else(|_| "https://eth.llamarpc.com".into());
        let ws_url = std::env::var("ETH_WS_URL")
            .map_err(|_| anyhow!("ETH_WS_URL is required for sandwich bot"))?;
        let private_key = std::env::var("PRIVATE_KEY")
            .map_err(|_| anyhow!("PRIVATE_KEY not set"))?;

        let front_run_eth: f64 = std::env::var("TRADE_AMOUNT")
            .unwrap_or_else(|_| "0.1".into())
            .parse()?;
        let front_run_amount_wei = U256::from((front_run_eth * 1e18) as u128);

        let max_gas_price_gwei: u64 = std::env::var("MAX_GAS_PRICE_GWEI")
            .unwrap_or_else(|_| "100".into())
            .parse()?;

        Ok(Self {
            chain,
            rpc_url,
            ws_url,
            private_key,
            front_run_amount_wei,
            max_gas_price_gwei,
            gas_limit: 300_000,
            flashbots_relay_url: std::env::var("FLASHBOTS_RELAY_URL")
                .unwrap_or_else(|_| "https://relay.flashbots.net".into()),
            flashbots_signer_key: std::env::var("FLASHBOTS_SIGNER_KEY")
                .unwrap_or_default(),
            watched_routers: vec![
                // Uniswap V2
                "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D".parse().unwrap(),
                // SushiSwap
                "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F".parse().unwrap(),
            ],
        })
    }
}

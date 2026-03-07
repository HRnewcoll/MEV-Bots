//! Bot configuration loaded from environment variables.

use anyhow::{anyhow, Result};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, U256};
use std::str::FromStr;

/// Per-chain DEX registry entry.
#[derive(Debug, Clone)]
pub struct DexInfo {
    pub name: String,
    pub router: Address,
    pub factory: Address,
}

/// Full runtime configuration.
#[derive(Debug, Clone)]
pub struct BotConfig {
    pub chain: String,
    pub rpc_url: String,
    pub ws_url: Option<String>,
    pub private_key: String,
    pub dexs: Vec<DexInfo>,
    /// (tokenA, tokenB) pairs to monitor
    pub token_pairs: Vec<(Address, Address)>,
    pub trade_amount_wei: U256,
    pub min_profit_wei: U256,
    pub max_gas_price_gwei: u64,
    pub poll_interval_ms: u64,
    pub flashbots_relay_url: Option<String>,
    pub flashbots_signer_key: Option<String>,
}

impl BotConfig {
    /// Load configuration from `.env` / environment variables.
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let chain = std::env::var("CHAIN").unwrap_or_else(|_| "ethereum".into());
        let rpc_url = std::env::var("ETH_RPC_URL")
            .or_else(|_| std::env::var("RPC_URL"))
            .unwrap_or_else(|_| "https://eth.llamarpc.com".into());
        let ws_url = std::env::var("ETH_WS_URL").ok();
        let private_key = std::env::var("PRIVATE_KEY")
            .map_err(|_| anyhow!("PRIVATE_KEY not set"))?;

        let trade_amount_eth: f64 = std::env::var("TRADE_AMOUNT")
            .unwrap_or_else(|_| "0.1".into())
            .parse()?;
        let trade_amount_wei = U256::from((trade_amount_eth * 1e18) as u128);

        let min_profit_eth: f64 = std::env::var("MIN_PROFIT_ETH")
            .unwrap_or_else(|_| "0.001".into())
            .parse()?;
        let min_profit_wei = U256::from((min_profit_eth * 1e18) as u128);

        let max_gas_price_gwei: u64 = std::env::var("MAX_GAS_PRICE_GWEI")
            .unwrap_or_else(|_| "100".into())
            .parse()?;

        let poll_interval_ms: u64 = std::env::var("POLL_INTERVAL_MS")
            .unwrap_or_else(|_| "1000".into())
            .parse()?;

        // Default Ethereum DEXs
        let dexs = ethereum_dexs();

        // Default WETH/USDC pair
        let token_pairs = vec![(
            Address::from_str("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")?,
            Address::from_str("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")?,
        )];

        Ok(Self {
            chain,
            rpc_url,
            ws_url,
            private_key,
            dexs,
            token_pairs,
            trade_amount_wei,
            min_profit_wei,
            max_gas_price_gwei,
            poll_interval_ms,
            flashbots_relay_url: std::env::var("FLASHBOTS_RELAY_URL").ok(),
            flashbots_signer_key: std::env::var("FLASHBOTS_SIGNER_KEY").ok(),
        })
    }

    /// Derive the wallet address from the private key.
    pub fn wallet_address(&self) -> Address {
        self.private_key
            .parse::<LocalWallet>()
            .map(|w| w.address())
            .unwrap_or_default()
    }
}

fn ethereum_dexs() -> Vec<DexInfo> {
    vec![
        DexInfo {
            name: "uniswap_v2".into(),
            router: "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D".parse().unwrap(),
            factory: "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f".parse().unwrap(),
        },
        DexInfo {
            name: "sushiswap".into(),
            router: "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F".parse().unwrap(),
            factory: "0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac".parse().unwrap(),
        },
    ]
}

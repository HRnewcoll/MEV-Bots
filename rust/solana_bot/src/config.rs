//! Solana bot configuration.

use anyhow::{anyhow, Result};
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;

/// A (mintA, mintB) token pair identified by SPL token mint addresses.
pub type TokenPair = (Pubkey, Pubkey);

#[derive(Debug, Clone)]
pub struct SolanaConfig {
    pub rpc_url: String,
    pub ws_url: String,
    /// Path to a Solana keypair JSON file
    pub keypair_path: String,
    /// Jito block engine URL for bundle submission
    pub jito_url: String,
    /// Amount to trade in lamports (1 SOL = 1e9 lamports)
    pub trade_amount_lamports: u64,
    pub min_profit_lamports: u64,
    pub poll_interval_ms: u64,
    pub token_pairs: Vec<TokenPair>,
    /// Raydium AMM pool accounts to scan
    pub raydium_pools: Vec<Pubkey>,
    /// Orca Whirlpool accounts to scan
    pub orca_pools: Vec<Pubkey>,
}

impl SolanaConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let rpc_url = std::env::var("SOLANA_RPC_URL")
            .unwrap_or_else(|_| "https://api.mainnet-beta.solana.com".into());
        let ws_url = std::env::var("SOLANA_WS_URL")
            .unwrap_or_else(|_| "wss://api.mainnet-beta.solana.com".into());
        let keypair_path = std::env::var("SOLANA_KEYPAIR_PATH")
            .map_err(|_| anyhow!("SOLANA_KEYPAIR_PATH not set"))?;
        let jito_url = std::env::var("JITO_URL")
            .unwrap_or_else(|_| "https://mainnet.block-engine.jito.wtf".into());

        let trade_sol: f64 = std::env::var("TRADE_AMOUNT_SOL")
            .unwrap_or_else(|_| "0.1".into())
            .parse()?;
        let trade_amount_lamports = (trade_sol * 1e9) as u64;

        let min_profit_sol: f64 = std::env::var("MIN_PROFIT_SOL")
            .unwrap_or_else(|_| "0.001".into())
            .parse()?;
        let min_profit_lamports = (min_profit_sol * 1e9) as u64;

        let poll_interval_ms: u64 = std::env::var("POLL_INTERVAL_MS")
            .unwrap_or_else(|_| "500".into())
            .parse()?;

        // SOL/USDC pair (mint addresses)
        let token_pairs = vec![(
            // Wrapped SOL
            Pubkey::from_str("So11111111111111111111111111111111111111112")?,
            // USDC
            Pubkey::from_str("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")?,
        )];

        // Known Raydium SOL/USDC pool
        let raydium_pools = vec![
            Pubkey::from_str("58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWaS3SqCvBLDQ")
                .unwrap_or_default(),
        ];
        // Known Orca SOL/USDC Whirlpool
        let orca_pools = vec![
            Pubkey::from_str("HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ")
                .unwrap_or_default(),
        ];

        Ok(Self {
            rpc_url,
            ws_url,
            keypair_path,
            jito_url,
            trade_amount_lamports,
            min_profit_lamports,
            poll_interval_ms,
            token_pairs,
            raydium_pools,
            orca_pools,
        })
    }
}

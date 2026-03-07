//! Configuration for the Rust AI MEV bot.

use anyhow::{anyhow, Result};
use ethers::types::Address;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct AIBotConfig {
    pub chain: String,
    pub rpc_url: String,
    pub ws_url: Option<String>,
    pub private_key: String,
    pub max_gas_price_gwei: u64,
    pub trade_amount_wei: u128,
    pub min_profit_wei: u128,
    pub poll_interval_ms: u64,
    pub skip_on_bearish: bool,
    /// Path to the exported ONNX price predictor model
    pub price_predictor_onnx: String,
    /// Path to the exported ONNX opportunity classifier model
    pub opp_classifier_onnx: String,
    /// Path to the exported ONNX RL policy model
    pub rl_policy_onnx: String,
    /// Sequence length expected by the price predictor
    pub price_seq_len: usize,
    /// Feature dimension expected by the price predictor
    pub price_input_size: usize,
    /// Feature dimension expected by the opportunity classifier
    pub clf_feature_dim: usize,
    /// Observation dimension expected by the RL policy
    pub rl_obs_dim: usize,
}

impl AIBotConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let chain = std::env::var("CHAIN").unwrap_or_else(|_| "ethereum".into());
        let rpc_url = std::env::var("ETH_RPC_URL")
            .unwrap_or_else(|_| "https://eth.llamarpc.com".into());
        let ws_url = std::env::var("ETH_WS_URL").ok();
        let private_key = std::env::var("PRIVATE_KEY")
            .map_err(|_| anyhow!("PRIVATE_KEY not set"))?;

        let trade_amount_eth: f64 = std::env::var("TRADE_AMOUNT")
            .unwrap_or_else(|_| "0.1".into()).parse()?;
        let min_profit_eth: f64 = std::env::var("MIN_PROFIT_ETH")
            .unwrap_or_else(|_| "0.002".into()).parse()?;

        Ok(Self {
            chain,
            rpc_url,
            ws_url,
            private_key,
            max_gas_price_gwei: std::env::var("MAX_GAS_PRICE_GWEI")
                .unwrap_or_else(|_| "100".into()).parse()?,
            trade_amount_wei: (trade_amount_eth * 1e18) as u128,
            min_profit_wei: (min_profit_eth * 1e18) as u128,
            poll_interval_ms: std::env::var("POLL_INTERVAL_MS")
                .unwrap_or_else(|_| "1000".into()).parse()?,
            skip_on_bearish: std::env::var("SKIP_ON_BEARISH")
                .unwrap_or_else(|_| "true".into())
                .to_lowercase() == "true",
            price_predictor_onnx: std::env::var("PP_ONNX")
                .unwrap_or_else(|_| "models/price_predictor.onnx".into()),
            opp_classifier_onnx: std::env::var("CLF_ONNX")
                .unwrap_or_else(|_| "models/opp_classifier.onnx".into()),
            rl_policy_onnx: std::env::var("RL_ONNX")
                .unwrap_or_else(|_| "models/rl_policy.onnx".into()),
            price_seq_len: std::env::var("PP_SEQ_LEN")
                .unwrap_or_else(|_| "30".into()).parse()?,
            price_input_size: std::env::var("PP_INPUT_SIZE")
                .unwrap_or_else(|_| "16".into()).parse()?,
            clf_feature_dim: std::env::var("CLF_FEATURE_DIM")
                .unwrap_or_else(|_| "32".into()).parse()?,
            rl_obs_dim: std::env::var("RL_OBS_DIM")
                .unwrap_or_else(|_| "7".into()).parse()?,
        })
    }
}

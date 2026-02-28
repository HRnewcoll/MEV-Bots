//! Mempool transaction monitoring and swap decoding.

use anyhow::Result;
use ethers::{
    contract::abigen,
    providers::{Middleware, Provider, StreamExt, Ws},
    types::{Address, Bytes, Transaction, U256},
};
use std::sync::Arc;
use tracing::{debug, info};

use crate::config::SandwichConfig;
use crate::types::{SandwichOpportunity, VictimSwap};
use crate::bundle::BundleSubmitter;

// Uniswap V2 Router ABI — only the swap selector we need
abigen!(
    UniswapV2Router,
    r#"[
        function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)
        function swapExactETHForTokens(uint amountOutMin, address[] calldata path, address to, uint deadline) external payable returns (uint[] memory amounts)
        function swapExactTokensForETH(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)
    ]"#
);

// 4-byte selectors for the swap functions above
const SWAP_EXACT_TOKENS_FOR_TOKENS: &str = "38ed1739";
const SWAP_EXACT_ETH_FOR_TOKENS: &str = "7ff36ab5";
const SWAP_EXACT_TOKENS_FOR_ETH: &str = "18cbafe5";

pub struct MempoolWatcher {
    cfg: SandwichConfig,
    submitter: BundleSubmitter,
}

impl MempoolWatcher {
    pub async fn new(cfg: SandwichConfig) -> Result<Self> {
        let submitter = BundleSubmitter::new(&cfg);
        Ok(Self { cfg, submitter })
    }

    /// Attempt to decode a transaction as a watched swap.
    fn decode_swap(&self, tx: &Transaction) -> Option<VictimSwap> {
        let to = tx.to?;
        if !self.cfg.watched_routers.contains(&to) {
            return None;
        }
        let input = &tx.input;
        if input.len() < 4 {
            return None;
        }
        let selector = hex::encode(&input[..4]);

        let (amount_in, amount_out_min, path) = match selector.as_str() {
            SWAP_EXACT_TOKENS_FOR_TOKENS => {
                // amountIn (32) + amountOutMin (32) + path offset (32) + to (32) + deadline (32) …
                if input.len() < 4 + 32 * 5 {
                    return None;
                }
                let amount_in = U256::from_big_endian(&input[4..36]);
                let amount_out_min = U256::from_big_endian(&input[36..68]);
                // path is a dynamic array; offset at bytes 68-100
                let path_offset = U256::from_big_endian(&input[68..100]).as_usize() + 4;
                if input.len() < path_offset + 32 {
                    return None;
                }
                let path_len = U256::from_big_endian(&input[path_offset..path_offset + 32]).as_usize();
                let mut path: Vec<Address> = Vec::with_capacity(path_len);
                for i in 0..path_len {
                    let start = path_offset + 32 + i * 32;
                    if input.len() < start + 32 {
                        return None;
                    }
                    path.push(Address::from_slice(&input[start + 12..start + 32]));
                }
                (amount_in, amount_out_min, path)
            }
            SWAP_EXACT_ETH_FOR_TOKENS => {
                if input.len() < 4 + 32 * 4 {
                    return None;
                }
                let amount_out_min = U256::from_big_endian(&input[4..36]);
                let path_offset = U256::from_big_endian(&input[36..68]).as_usize() + 4;
                if input.len() < path_offset + 32 {
                    return None;
                }
                let path_len = U256::from_big_endian(&input[path_offset..path_offset + 32]).as_usize();
                let mut path: Vec<Address> = Vec::with_capacity(path_len);
                for i in 0..path_len {
                    let start = path_offset + 32 + i * 32;
                    if input.len() < start + 32 {
                        return None;
                    }
                    path.push(Address::from_slice(&input[start + 12..start + 32]));
                }
                (tx.value, amount_out_min, path)
            }
            _ => return None,
        };

        if path.len() < 2 {
            return None;
        }

        Some(VictimSwap {
            tx_hash: format!("{:?}", tx.hash),
            router: to,
            token_in: path[0],
            token_out: path[path.len() - 1],
            amount_in,
            amount_out_min,
            path,
            raw_tx: input.clone(),
        })
    }

    /// Very rough profit estimate.
    fn estimate_profit(&self, victim: &VictimSwap, gas_price: U256) -> i128 {
        // Simplified: assume price impact ≈ amountIn / total_liquidity
        // and we capture ~half of that impact.
        // In production, use on-chain simulation.
        let impact_bps: i128 = 10; // 0.1% estimate
        let our_amount: i128 = self.cfg.front_run_amount_wei.as_u128() as i128;
        let gross_profit = our_amount * impact_bps / 10_000;
        let gas_cost = (gas_price.as_u128() as i128) * (self.cfg.gas_limit as i128) * 2;
        gross_profit - gas_cost
    }

    pub async fn run(self) -> Result<()> {
        info!("Connecting to mempool WebSocket: {}", self.cfg.ws_url);
        let provider = Provider::<Ws>::connect(&self.cfg.ws_url).await?;
        let provider = Arc::new(provider);

        info!("Subscribing to pending transactions …");
        let mut stream = provider.subscribe_pending_txs().await?;

        while let Some(tx_hash) = stream.next().await {
            let provider_clone = Arc::clone(&provider);
            let cfg_clone = self.cfg.clone();
            let submitter_clone = self.submitter.clone();

            tokio::spawn(async move {
                let tx = match provider_clone.get_transaction(tx_hash).await {
                    Ok(Some(tx)) => tx,
                    _ => return,
                };

                let watcher = MempoolWatcher {
                    cfg: cfg_clone,
                    submitter: submitter_clone,
                };

                if let Some(victim) = watcher.decode_swap(&tx) {
                    let gas_price = tx.gas_price.unwrap_or_default();
                    let profit = watcher.estimate_profit(&victim, gas_price);
                    if profit > 0 {
                        info!(
                            victim_tx = %victim.tx_hash,
                            profit_wei = %profit,
                            "Sandwich opportunity detected"
                        );
                        let block = provider_clone.get_block_number().await.unwrap_or_default();
                        let opp = SandwichOpportunity {
                            victim,
                            front_run_amount: watcher.cfg.front_run_amount_wei,
                            estimated_profit_wei: profit,
                            target_block: block.as_u64() + 1,
                        };
                        if let Err(e) = watcher.submitter.submit_bundle(&opp, gas_price).await {
                            debug!("Bundle submission error: {e}");
                        }
                    }
                }
            });
        }
        Ok(())
    }
}

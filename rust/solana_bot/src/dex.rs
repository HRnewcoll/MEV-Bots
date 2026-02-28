//! Solana DEX pool scanning (Raydium V4 + Orca Whirlpool).

use anyhow::Result;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::sync::Arc;
use tracing::debug;

use crate::config::SolanaConfig;
use crate::types::{PoolQuote, SolanaOpportunity};

/// Constant-product AMM output calculation (0.25 % fee — Raydium default).
fn amm_out(amount_in: u64, reserve_in: u64, reserve_out: u64) -> u64 {
    if reserve_in == 0 || reserve_out == 0 {
        return 0;
    }
    // fee = 25 bps → multiply by 9975 / 10000
    let amount_in_after_fee = amount_in as u128 * 9975;
    let numerator = amount_in_after_fee * reserve_out as u128;
    let denominator = reserve_in as u128 * 10000 + amount_in_after_fee;
    (numerator / denominator) as u64
}

pub struct SolanaScanner {
    client: Arc<RpcClient>,
    cfg: SolanaConfig,
}

impl SolanaScanner {
    pub fn new(cfg: &SolanaConfig) -> Result<Self> {
        Ok(Self {
            client: Arc::new(RpcClient::new(cfg.rpc_url.clone())),
            cfg: cfg.clone(),
        })
    }

    /// Fetch Raydium V4 AMM reserves and compute the output amount.
    ///
    /// Raydium V4 AMM account layout (simplified):
    /// - offset 253: token coin reserve (u64, 8 bytes)
    /// - offset 261: token pc reserve (u64, 8 bytes)
    async fn raydium_quote(
        &self,
        pool: &Pubkey,
        mint_in: &Pubkey,
        amount_in: u64,
    ) -> Result<u64> {
        let data = self.client.get_account_data(pool).await?;
        if data.len() < 269 {
            return Ok(0);
        }
        let reserve_coin = u64::from_le_bytes(data[253..261].try_into()?);
        let reserve_pc = u64::from_le_bytes(data[261..269].try_into()?);

        // Raydium stores coin/pc order; we need to determine direction
        // For simplicity assume mint_in == coin and swap coin → pc
        // In production: read coinMint / pcMint from the pool account header
        let out = amm_out(amount_in, reserve_coin, reserve_pc);
        Ok(out)
    }

    /// Fetch Orca Whirlpool state and compute a rough CLMM quote.
    ///
    /// Whirlpool account layout (simplified):
    /// - offset 65: sqrt_price (u128, 16 bytes)
    /// - offset 101: liquidity (u128, 16 bytes)
    async fn orca_quote(
        &self,
        pool: &Pubkey,
        amount_in: u64,
    ) -> Result<u64> {
        let data = self.client.get_account_data(pool).await?;
        if data.len() < 117 {
            return Ok(0);
        }
        let sqrt_price = u128::from_le_bytes(data[65..81].try_into()?);
        let liquidity = u128::from_le_bytes(data[101..117].try_into()?);

        if sqrt_price == 0 || liquidity == 0 {
            return Ok(0);
        }

        // Simplified spot price approximation: price = (sqrt_price / 2^64)^2
        let price_x64 = sqrt_price;
        // amount_out ≈ amount_in * price  (constant-product approximation for small trades)
        let amount_out = (amount_in as u128)
            .checked_mul(price_x64)
            .and_then(|v| v.checked_div(1u128 << 64))
            .unwrap_or(0) as u64;
        Ok(amount_out)
    }

    /// Scan all pools and return the best arbitrage opportunity, if any.
    pub async fn find_opportunity(
        &self,
        mint_in: &Pubkey,
        mint_out: &Pubkey,
        amount_in: u64,
    ) -> Result<Option<SolanaOpportunity>> {
        let mut quotes: Vec<PoolQuote> = Vec::new();

        for pool in &self.cfg.raydium_pools {
            match self.raydium_quote(pool, mint_in, amount_in).await {
                Ok(out) if out > 0 => {
                    debug!(pool = %pool, out, "Raydium quote");
                    quotes.push(PoolQuote { pool: *pool, dex: "raydium".into(), amount_out: out });
                }
                Ok(_) => {}
                Err(e) => debug!(pool = %pool, error = %e, "Raydium quote error"),
            }
        }

        for pool in &self.cfg.orca_pools {
            match self.orca_quote(pool, amount_in).await {
                Ok(out) if out > 0 => {
                    debug!(pool = %pool, out, "Orca quote");
                    quotes.push(PoolQuote { pool: *pool, dex: "orca".into(), amount_out: out });
                }
                Ok(_) => {}
                Err(e) => debug!(pool = %pool, error = %e, "Orca quote error"),
            }
        }

        if quotes.len() < 2 {
            return Ok(None);
        }

        let best = quotes.iter().max_by_key(|q| q.amount_out).unwrap().clone();
        let worst = quotes.iter().min_by_key(|q| q.amount_out).unwrap().clone();

        if best.amount_out <= worst.amount_out {
            return Ok(None);
        }

        let profit = best.amount_out.saturating_sub(worst.amount_out);
        if profit < self.cfg.min_profit_lamports {
            return Ok(None);
        }

        Ok(Some(SolanaOpportunity {
            mint_in: *mint_in,
            mint_out: *mint_out,
            buy_pool: best.pool,
            buy_dex: best.dex,
            sell_pool: worst.pool,
            sell_dex: worst.dex,
            amount_in,
            out_buy: best.amount_out,
            out_sell: worst.amount_out,
            estimated_profit_lamports: profit,
        }))
    }
}

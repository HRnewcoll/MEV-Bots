//! DEX price scanning and opportunity detection.

use anyhow::Result;
use ethers::{
    contract::abigen,
    middleware::SignerMiddleware,
    providers::{Http, Middleware, Provider},
    signers::{LocalWallet, Signer},
    types::{Address, U256},
};
use std::sync::Arc;
use tracing::debug;

use crate::config::{BotConfig, DexInfo};
use crate::types::Opportunity;

// Generate type-safe bindings for Uniswap V2 Factory and Pair
abigen!(
    UniswapV2Factory,
    r#"[
        function getPair(address tokenA, address tokenB) external view returns (address pair)
    ]"#
);

abigen!(
    UniswapV2Pair,
    r#"[
        function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast)
        function token0() external view returns (address)
        function token1() external view returns (address)
    ]"#
);

type Client = Arc<Provider<Http>>;

/// Computes the Uniswap V2 output amount (0.3 % fee).
fn get_amount_out(amount_in: U256, reserve_in: U256, reserve_out: U256) -> U256 {
    if reserve_in.is_zero() || reserve_out.is_zero() {
        return U256::zero();
    }
    let amount_in_with_fee = amount_in * 997u64;
    let numerator = amount_in_with_fee * reserve_out;
    let denominator = reserve_in * 1000u64 + amount_in_with_fee;
    numerator / denominator
}

pub struct DexScanner {
    client: Client,
    dexs: Vec<DexInfo>,
}

impl DexScanner {
    pub async fn new(cfg: &BotConfig) -> Result<Self> {
        let provider = Provider::<Http>::try_from(cfg.rpc_url.as_str())?;
        Ok(Self {
            client: Arc::new(provider),
            dexs: cfg.dexs.clone(),
        })
    }

    /// Fetch the output amount for `amount_in` of tokenA → tokenB on a given DEX.
    async fn price_on_dex(
        &self,
        dex: &DexInfo,
        token_a: Address,
        token_b: Address,
        amount_in: U256,
    ) -> Result<U256> {
        let factory = UniswapV2Factory::new(dex.factory, Arc::clone(&self.client));
        let pair_addr = factory.get_pair(token_a, token_b).call().await?;
        if pair_addr == Address::zero() {
            return Ok(U256::zero());
        }
        let pair = UniswapV2Pair::new(pair_addr, Arc::clone(&self.client));
        let (r0, r1, _) = pair.get_reserves().call().await?;
        let t0 = pair.token_0().call().await?;

        let (reserve_in, reserve_out) = if t0 == token_a {
            (U256::from(r0), U256::from(r1))
        } else {
            (U256::from(r1), U256::from(r0))
        };

        Ok(get_amount_out(amount_in, reserve_in, reserve_out))
    }

    /// Scan all DEXs for a token pair and return the best opportunity if one exists.
    pub async fn find_opportunity(
        &self,
        token_a: &Address,
        token_b: &Address,
        amount_in: U256,
    ) -> Result<Option<Opportunity>> {
        let mut prices: Vec<(String, Address, U256)> = Vec::new();

        for dex in &self.dexs {
            match self.price_on_dex(dex, *token_a, *token_b, amount_in).await {
                Ok(out) if !out.is_zero() => {
                    debug!(dex = %dex.name, out = %out, "Price fetched");
                    prices.push((dex.name.clone(), dex.router, out));
                }
                Ok(_) => {}
                Err(e) => debug!(dex = %dex.name, error = %e, "Price fetch error"),
            }
        }

        if prices.len() < 2 {
            return Ok(None);
        }

        // Best buy: highest output (cheapest price for tokenA)
        let (buy_name, buy_router, out_buy) = prices
            .iter()
            .max_by_key(|(_, _, out)| *out)
            .cloned()
            .unwrap();
        // Best sell: lowest output (most expensive price, best to sell into)
        let (sell_name, sell_router, out_sell) = prices
            .iter()
            .min_by_key(|(_, _, out)| *out)
            .cloned()
            .unwrap();

        if out_buy <= out_sell {
            return Ok(None);
        }

        let profit = out_buy - out_sell;
        Ok(Some(Opportunity {
            token_a: *token_a,
            token_b: *token_b,
            buy_dex: buy_name,
            sell_dex: sell_name,
            buy_router,
            sell_router,
            amount_in,
            out_buy,
            out_sell,
            profit_gross: profit,
        }))
    }
}

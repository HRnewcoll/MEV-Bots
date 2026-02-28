//! Transaction building and submission.

use anyhow::{anyhow, Result};
use ethers::{
    contract::abigen,
    middleware::SignerMiddleware,
    providers::{Http, Middleware, Provider},
    signers::{LocalWallet, Signer},
    types::{Address, TransactionReceipt, U256},
};
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::BotConfig;
use crate::types::Opportunity;

abigen!(
    UniswapV2Router,
    r#"[
        function swapExactTokensForTokens(uint amountIn, uint amountOutMin, address[] calldata path, address to, uint deadline) external returns (uint[] memory amounts)
        function getAmountsOut(uint amountIn, address[] calldata path) external view returns (uint[] memory amounts)
    ]"#
);

abigen!(
    ERC20,
    r#"[
        function approve(address spender, uint256 amount) external returns (bool)
        function balanceOf(address account) external view returns (uint256)
    ]"#
);

type SignedClient = Arc<SignerMiddleware<Provider<Http>, LocalWallet>>;

pub struct Executor {
    client: SignedClient,
    max_gas_price: U256,
    wallet: Address,
}

impl Executor {
    pub async fn new(cfg: &BotConfig) -> Result<Self> {
        let provider = Provider::<Http>::try_from(cfg.rpc_url.as_str())?;
        let chain_id = provider.get_chainid().await?.as_u64();
        let wallet: LocalWallet = cfg.private_key.parse::<LocalWallet>()?.with_chain_id(chain_id);
        let wallet_addr = wallet.address();
        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        let max_gas_price = U256::from(cfg.max_gas_price_gwei) * U256::exp10(9);
        Ok(Self {
            client,
            max_gas_price,
            wallet: wallet_addr,
        })
    }

    async fn approve(&self, token: Address, spender: Address, amount: U256) -> Result<()> {
        let erc20 = ERC20::new(token, Arc::clone(&self.client));
        let tx = erc20.approve(spender, amount).send().await?;
        tx.await?;
        Ok(())
    }

    fn deadline() -> U256 {
        U256::from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 60,
        )
    }

    pub async fn execute(&self, opp: &Opportunity) -> Result<()> {
        let gas_price = self.client.get_gas_price().await?;
        if gas_price > self.max_gas_price {
            warn!(
                gas_gwei = %(gas_price / U256::exp10(9)),
                max_gwei = %(self.max_gas_price / U256::exp10(9)),
                "Gas price too high — skipping"
            );
            return Ok(());
        }

        // Approve buy router to spend token_a
        self.approve(opp.token_a, opp.buy_router, opp.amount_in).await?;

        let path = vec![opp.token_a, opp.token_b];
        let router = UniswapV2Router::new(opp.buy_router, Arc::clone(&self.client));

        // Leg 1: buy token_b on buy_dex
        let receipt: TransactionReceipt = router
            .swap_exact_tokens_for_tokens(
                opp.amount_in,
                U256::zero(), // accept any amount (slippage handled off-chain)
                path.clone(),
                self.wallet,
                Self::deadline(),
            )
            .send()
            .await?
            .await?
            .ok_or_else(|| anyhow!("Leg-1 receipt missing"))?;

        info!(tx = ?receipt.transaction_hash, "Leg-1 buy executed");
        if receipt.status != Some(1u64.into()) {
            return Err(anyhow!("Leg-1 transaction reverted"));
        }

        // Leg 2: sell token_b back on sell_dex
        let token_b_contract = ERC20::new(opp.token_b, Arc::clone(&self.client));
        let balance = token_b_contract.balance_of(self.wallet).call().await?;

        self.approve(opp.token_b, opp.sell_router, balance).await?;

        let sell_router = UniswapV2Router::new(opp.sell_router, Arc::clone(&self.client));
        let reverse_path = vec![opp.token_b, opp.token_a];
        let receipt2: TransactionReceipt = sell_router
            .swap_exact_tokens_for_tokens(
                balance,
                U256::zero(),
                reverse_path,
                self.wallet,
                Self::deadline(),
            )
            .send()
            .await?
            .await?
            .ok_or_else(|| anyhow!("Leg-2 receipt missing"))?;

        info!(tx = ?receipt2.transaction_hash, "Leg-2 sell executed — arbitrage complete");
        Ok(())
    }
}

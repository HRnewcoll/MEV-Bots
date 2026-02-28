//! Action executor for the AI MEV bot.

use anyhow::Result;
use ethers::{
    middleware::SignerMiddleware,
    providers::{Http, Middleware, Provider},
    signers::{LocalWallet, Signer},
    types::Address,
};
use std::sync::Arc;
use tracing::{info, warn};

use crate::config::AIBotConfig;
use crate::types::MevOpportunity;

type SignedClient = Arc<SignerMiddleware<Provider<Http>, LocalWallet>>;

pub struct AiExecutor {
    client: SignedClient,
    wallet: Address,
    max_gas_price_wei: u128,
}

impl AiExecutor {
    pub async fn new(cfg: &AIBotConfig) -> Result<Self> {
        let provider = Provider::<Http>::try_from(cfg.rpc_url.as_str())?;
        let chain_id = provider.get_chainid().await?.as_u64();
        let wallet: LocalWallet = cfg.private_key.parse::<LocalWallet>()?.with_chain_id(chain_id);
        let wallet_addr = wallet.address();
        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        Ok(Self {
            client,
            wallet: wallet_addr,
            max_gas_price_wei: cfg.max_gas_price_gwei as u128 * 1_000_000_000,
        })
    }

    /// Execute the RL-chosen action.
    pub async fn execute_action(
        &self,
        action: String,
        opportunity: Option<&MevOpportunity>,
    ) -> Result<()> {
        let gas_price = self.client.get_gas_price().await?.as_u128();
        if gas_price > self.max_gas_price_wei {
            warn!(
                gas_gwei = gas_price / 1_000_000_000,
                "Gas too high — skipping action"
            );
            return Ok(());
        }

        match action.as_str() {
            "do_nothing" => {
                info!("Action: do_nothing");
            }
            "arb" => {
                info!("Action: arb — would execute best arbitrage here");
                // In production: call the arbitrage_bot executor
            }
            "sandwich" => {
                if let Some(opp) = opportunity {
                    info!(tx = %opp.tx_hash, "Action: sandwich — would execute sandwich here");
                    // In production: call the sandwich_bot bundle submitter
                }
            }
            "liquidate" => {
                info!("Action: liquidate — would execute liquidation here");
                // In production: call the liquidation_bot executor
            }
            _ => warn!("Unknown action: {}", action),
        }
        Ok(())
    }
}

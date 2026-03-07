//! Solana transaction execution via Jito bundles.

use anyhow::Result;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    commitment_config::CommitmentConfig,
    signature::{read_keypair_file, Keypair, Signer},
    transaction::Transaction,
};
use tracing::info;

use crate::config::SolanaConfig;
use crate::types::SolanaOpportunity;

pub struct SolanaExecutor {
    client: RpcClient,
    keypair: Keypair,
    jito_url: String,
}

impl SolanaExecutor {
    pub fn new(cfg: &SolanaConfig) -> Result<Self> {
        let keypair = read_keypair_file(&cfg.keypair_path)
            .map_err(|e| anyhow::anyhow!("Failed to read keypair: {e}"))?;
        Ok(Self {
            client: RpcClient::new_with_commitment(
                cfg.rpc_url.clone(),
                CommitmentConfig::confirmed(),
            ),
            keypair,
            jito_url: cfg.jito_url.clone(),
        })
    }

    /// Execute the arbitrage via a Jito bundle (two swap instructions in one transaction).
    pub async fn execute(&self, opp: &SolanaOpportunity) -> Result<()> {
        info!(
            buy_pool = %opp.buy_pool,
            sell_pool = %opp.sell_pool,
            profit_lamports = %opp.estimated_profit_lamports,
            "Executing Solana arbitrage"
        );

        let recent_blockhash = self.client.get_latest_blockhash().await?;

        // In a real implementation, build SPL token swap instructions for each pool.
        // Here we build a no-op memo transaction as a structural placeholder.
        let buy_ix = solana_sdk::system_instruction::transfer(
            &self.keypair.pubkey(),
            &self.keypair.pubkey(),
            0,
        );
        let sell_ix = solana_sdk::system_instruction::transfer(
            &self.keypair.pubkey(),
            &self.keypair.pubkey(),
            0,
        );

        let tx = Transaction::new_signed_with_payer(
            &[buy_ix, sell_ix],
            Some(&self.keypair.pubkey()),
            &[&self.keypair],
            recent_blockhash,
        );

        // Submit via Jito bundle for MEV-protected ordering
        self.submit_jito_bundle(tx).await
    }

    async fn submit_jito_bundle(&self, tx: Transaction) -> Result<()> {
        let serialized = bincode::serialize(&tx)?;
        let encoded = base64::encode(&serialized);

        let bundle = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [[encoded]]
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/v1/bundles", self.jito_url))
            .header("Content-Type", "application/json")
            .json(&bundle)
            .send()
            .await?;

        info!(status = %resp.status(), "Jito bundle submitted");
        Ok(())
    }
}

//! Flashbots bundle construction and submission.

use anyhow::Result;
use ethers::{
    signers::{LocalWallet, Signer},
    types::{Bytes, U256},
    utils::keccak256,
};
use tracing::info;

use crate::config::SandwichConfig;
use crate::types::SandwichOpportunity;

#[derive(Clone)]
pub struct BundleSubmitter {
    relay_url: String,
    signer_key: String,
}

impl BundleSubmitter {
    pub fn new(cfg: &SandwichConfig) -> Self {
        Self {
            relay_url: cfg.flashbots_relay_url.clone(),
            signer_key: cfg.flashbots_signer_key.clone(),
        }
    }

    pub async fn submit_bundle(
        &self,
        opp: &SandwichOpportunity,
        gas_price: U256,
    ) -> Result<()> {
        if self.signer_key.is_empty() {
            info!("No Flashbots signer key configured — skipping bundle submission");
            return Ok(());
        }

        // Build bundle JSON
        let bundle_body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "eth_sendBundle",
            "params": [{
                "txs": [
                    // Front-run and back-run placeholders (full impl requires signed raw txs)
                    format!("0x{}", hex::encode(&opp.victim.raw_tx)),
                ],
                "blockNumber": format!("0x{:x}", opp.target_block),
            }],
            "id": 1
        });

        let body_str = bundle_body.to_string();
        let msg_hash = keccak256(
            format!("\x19Ethereum Signed Message:\n{}{}", body_str.len(), body_str).as_bytes(),
        );

        let signer: LocalWallet = self.signer_key.parse()?;
        let sig = signer.sign_hash(msg_hash.into())?;

        let client = reqwest::Client::new();
        let response = client
            .post(&self.relay_url)
            .header("Content-Type", "application/json")
            .header(
                "X-Flashbots-Signature",
                format!("{:?}:0x{}", signer.address(), hex::encode(sig.to_vec())),
            )
            .body(body_str)
            .send()
            .await?;

        info!(status = %response.status(), "Bundle submitted to Flashbots relay");
        Ok(())
    }
}

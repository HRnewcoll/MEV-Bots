//! AI-powered MEV bot — Rust implementation.
//!
//! Uses ONNX Runtime (`ort`) to load and run the pre-trained Python models:
//!   - `price_predictor.onnx`   (exported from the LSTM PricePredictor)
//!   - `opp_classifier.onnx`    (exported from the MLP OpportunityClassifier)
//!
//! The RL agent policy is also exportable to ONNX via stable-baselines3's
//! `model.policy.to_onnx()` helper.
//!
//! # Export models from Python
//! ```bash
//! cd python/ai_bot
//! python - <<'EOF'
//! import torch
//! from models.price_predictor import PricePredictor
//! m = PricePredictor(); m.load("checkpoints/price_predictor.pt")
//! dummy = torch.randn(1, 30, 16)
//! torch.onnx.export(m.model, dummy, "price_predictor.onnx",
//!                   input_names=["input"], output_names=["logits"])
//! EOF
//! ```

mod config;
mod inference;
mod scanner;
mod executor;
mod types;

use anyhow::Result;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use config::AIBotConfig;
use inference::ModelInference;
use scanner::AiScanner;
use executor::AiExecutor;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("ai_bot=info".parse()?),
        )
        .init();

    let cfg = AIBotConfig::from_env()?;
    info!(chain = %cfg.chain, "AI MEV bot starting");

    let inference = ModelInference::new(&cfg)?;
    let scanner = AiScanner::new(&cfg).await?;
    let executor = AiExecutor::new(&cfg).await?;

    let mut step: u64 = 0;

    loop {
        let market = scanner.get_market_snapshot().await;

        // ------ Price prediction ------
        let price_signal = inference.predict_price_direction(&market);
        info!(signal = %price_signal, "Price prediction");

        if price_signal == "down" && cfg.skip_on_bearish {
            info!("Bearish signal — skipping block");
            tokio::time::sleep(std::time::Duration::from_millis(cfg.poll_interval_ms)).await;
            continue;
        }

        // ------ Opportunity classification ------
        let pending_txs = scanner.get_pending_transactions().await;
        let mut best_opportunity: Option<types::MevOpportunity> = None;

        for tx in &pending_txs {
            let class = inference.classify_opportunity(tx);
            if class != "no_mev" {
                info!(tx_hash = %tx.hash, class = %class, "MEV opportunity classified");
                if best_opportunity.is_none() {
                    best_opportunity = Some(types::MevOpportunity {
                        tx_hash: tx.hash.clone(),
                        opportunity_type: class,
                        estimated_profit_wei: tx.estimated_profit_wei,
                    });
                }
            }
        }

        // ------ RL action selection ------
        let obs = inference.build_observation(&market, &pending_txs);
        let action = inference.rl_predict(&obs);
        info!(action = %action, step = %step, "RL agent action");

        if let Err(e) = executor.execute_action(action, best_opportunity.as_ref()).await {
            error!("Execution error: {e}");
        }

        step += 1;
        tokio::time::sleep(std::time::Duration::from_millis(cfg.poll_interval_ms)).await;
    }
}

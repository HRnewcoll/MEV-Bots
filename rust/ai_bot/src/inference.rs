//! ONNX Runtime–based ML inference for price prediction, opportunity
//! classification, and RL action selection.
//!
//! All three models are loaded from `.onnx` files at startup.
//! If a model file does not exist, the corresponding method falls back
//! to a simple heuristic so the bot can still run without trained models.

use anyhow::Result;
use ndarray::{Array1, Array3};
use tracing::{debug, warn};

use crate::config::AIBotConfig;
use crate::types::{MarketSnapshot, TxFeatures};

// Swap selectors — matches the Python classifier
const SWAP_SELECTORS: &[&str] = &[
    "38ed1739", // swapExactTokensForTokens
    "7ff36ab5", // swapExactETHForTokens
    "18cbafe5", // swapExactTokensForETH
    "414bf389", // exactInputSingle (V3)
    "c04b8d59", // exactInput (V3)
];

const PRICE_CLASSES: &[&str] = &["down", "flat", "up"];
const OPP_CLASSES: &[&str] = &["no_mev", "sandwich_target", "arb_trigger", "liquidation_target"];
const ACTION_NAMES: &[&str] = &["do_nothing", "arb", "sandwich", "liquidate"];

pub struct ModelInference {
    cfg: AIBotConfig,
    // ONNX sessions are wrapped in Option so the bot starts even if
    // the model files are missing (falls back to heuristics).
    price_session: Option<ort::Session>,
    clf_session: Option<ort::Session>,
    rl_session: Option<ort::Session>,
}

impl ModelInference {
    pub fn new(cfg: &AIBotConfig) -> Result<Self> {
        let price_session = load_onnx_session(&cfg.price_predictor_onnx);
        let clf_session = load_onnx_session(&cfg.opp_classifier_onnx);
        let rl_session = load_onnx_session(&cfg.rl_policy_onnx);
        Ok(Self {
            cfg: cfg.clone(),
            price_session,
            clf_session,
            rl_session,
        })
    }

    // ------------------------------------------------------------------
    // Price direction prediction
    // ------------------------------------------------------------------

    /// Predict price direction from a market snapshot.
    ///
    /// Returns "down", "flat", or "up".
    pub fn predict_price_direction(&self, market: &MarketSnapshot) -> String {
        if let Some(session) = &self.price_session {
            // Build a (1, seq_len, input_size) tensor filled with the latest
            // snapshot values (a real bot would maintain a rolling buffer).
            let seq_len = self.cfg.price_seq_len;
            let input_size = self.cfg.price_input_size;
            let mut data = vec![0.0f32; seq_len * input_size];
            // Fill last time-step with current market features
            let offset = (seq_len - 1) * input_size;
            data[offset] = market.gas_price_gwei / 200.0;
            data[offset + 1] = market.eth_price_usd / 5000.0;
            data[offset + 2] = market.pending_tx_count as f32 / 1000.0;

            let input_array =
                Array3::from_shape_vec((1, seq_len, input_size), data).unwrap_or_default();

            match run_onnx_f32(session, input_array.into_raw_vec()) {
                Some(logits) => {
                    let idx = argmax(&logits);
                    return PRICE_CLASSES[idx.min(2)].to_string();
                }
                None => {}
            }
        }
        // Heuristic fallback: high gas → likely up, very low gas → possibly down
        if market.gas_price_gwei > 80.0 { "up".into() } else { "flat".into() }
    }

    // ------------------------------------------------------------------
    // Opportunity classification
    // ------------------------------------------------------------------

    /// Classify a pending transaction.
    ///
    /// Returns one of: "no_mev", "sandwich_target", "arb_trigger", "liquidation_target".
    pub fn classify_opportunity(&self, tx: &TxFeatures) -> String {
        if let Some(session) = &self.clf_session {
            let features = self.extract_clf_features(tx);
            if let Some(logits) = run_onnx_f32(session, features) {
                let idx = argmax(&logits);
                return OPP_CLASSES[idx.min(3)].to_string();
            }
        }
        // Heuristic fallback: any large swap is a potential sandwich target
        let is_swap = SWAP_SELECTORS.iter().any(|s| tx.selector.contains(s));
        if is_swap && tx.value_eth > 5.0 { "sandwich_target".into() } else { "no_mev".into() }
    }

    fn extract_clf_features(&self, tx: &TxFeatures) -> Vec<f32> {
        let mut v = vec![0.0f32; self.cfg.clf_feature_dim];
        v[0] = (tx.value_eth + 1e-18_f32).log10();
        v[1] = tx.gas_price_gwei.max(1e-9).log10();
        v[2] = (tx.input_length as f32 + 1.0).log10();
        v[3] = if SWAP_SELECTORS.iter().any(|s| tx.selector.contains(s)) { 1.0 } else { 0.0 };
        v[4] = if tx.selector.contains("095ea7b3") { 1.0 } else { 0.0 };
        v[5] = (tx.nonce as f32 / 1000.0).min(10.0);
        v[6] = (tx.gas_limit as f32 / 1e6).min(1.0);
        v
    }

    // ------------------------------------------------------------------
    // RL policy inference
    // ------------------------------------------------------------------

    /// Build an observation vector from the market snapshot and pending txs.
    pub fn build_observation(
        &self,
        market: &MarketSnapshot,
        _pending_txs: &[TxFeatures],
    ) -> Vec<f32> {
        vec![
            market.gas_price_gwei / 200.0,
            market.eth_price_usd / 5000.0,
            market.pending_tx_count as f32 / 1000.0,
            market.arb_opportunities as f32 / 10.0,
            market.liquidation_opportunities as f32 / 5.0,
            (market.block_number % 100) as f32 / 100.0,
            market.wallet_balance_eth / 10.0,
        ]
    }

    /// Run the RL policy and return the chosen action name.
    pub fn rl_predict(&self, obs: &[f32]) -> String {
        if let Some(session) = &self.rl_session {
            if let Some(logits) = run_onnx_f32(session, obs.to_vec()) {
                let idx = argmax(&logits);
                return ACTION_NAMES[idx.min(3)].to_string();
            }
        }
        // Heuristic fallback
        "do_nothing".into()
    }
}

// ---------------------------------------------------------------------------
// ONNX Runtime helpers
// ---------------------------------------------------------------------------

fn load_onnx_session(path: &str) -> Option<ort::Session> {
    if !std::path::Path::new(path).exists() {
        warn!("ONNX model not found at {} — using heuristic fallback", path);
        return None;
    }
    match ort::Session::builder()
        .and_then(|b| b.commit_from_file(path))
    {
        Ok(s) => {
            debug!("Loaded ONNX model: {}", path);
            Some(s)
        }
        Err(e) => {
            warn!("Failed to load ONNX model {}: {}", path, e);
            None
        }
    }
}

fn run_onnx_f32(session: &ort::Session, input: Vec<f32>) -> Option<Vec<f32>> {
    use ort::inputs;
    let len = input.len();
    let array = ndarray::Array1::from_vec(input);
    let input_tensor = ort::Value::from_array(array.into_dyn()).ok()?;
    let outputs = session.run(inputs![input_tensor].ok()?).ok()?;
    let output = outputs[0].extract_tensor::<f32>().ok()?;
    let view = output.view();
    Some(view.iter().cloned().collect())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

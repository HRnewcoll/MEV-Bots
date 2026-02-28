"""
AI-Powered MEV Bot — Python implementation.

This bot combines three AI/ML components to maximise MEV capture:

1. **PricePredictor** (LSTM neural network)
   - Input:  last N candles of OHLCV data + on-chain features (gas, block time)
   - Output: predicted price direction over the next 1–3 blocks
   - Use:    pre-filter arbitrage and sandwich opportunities before costly on-chain calls

2. **OpportunityClassifier** (gradient-boosted tree / MLP)
   - Input:  mempool transaction features (gas, value, input selector, from-address history)
   - Output: probability that a pending tx is a profitable sandwich target
   - Use:    rank pending transactions to focus attention on the best candidates

3. **RLAgent** (Proximal Policy Optimisation via stable-baselines3)
   - State:  current market state (prices, liquidity, gas price, pending txs)
   - Action: one of {do_nothing, arb_dex_pair_0, sandwich_tx_i, liquidate_user_j}
   - Reward: realised profit after gas (negative for losses)
   - Use:    learn an optimal policy that balances risk/reward across all strategies

The three components are orchestrated by `AIMevBot`, which runs an async main loop.
"""

import asyncio
import logging
import time
from pathlib import Path

from web3 import AsyncWeb3

from config import AIBotConfig
from models.price_predictor import PricePredictor
from models.opportunity_classifier import OpportunityClassifier
from models.rl_agent import RLAgent, MevEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_bot")


class AIMevBot:
    """
    AI-powered MEV bot that combines price prediction, opportunity classification,
    and reinforcement learning to execute optimal MEV strategies.
    """

    def __init__(self, cfg: AIBotConfig) -> None:
        self.cfg = cfg
        chain_cfg = cfg.chain_config()
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(chain_cfg["rpc_url"]))

        # ------------------------------------------------------------------ #
        # Instantiate AI models
        # ------------------------------------------------------------------ #
        self.price_predictor = PricePredictor(
            input_size=cfg.price_predictor_input_size,
            hidden_size=cfg.price_predictor_hidden_size,
            num_layers=cfg.price_predictor_num_layers,
            output_size=3,  # [down, flat, up]
        )

        self.opp_classifier = OpportunityClassifier(
            feature_dim=cfg.classifier_feature_dim,
        )

        self.rl_env = MevEnvironment(cfg=cfg, w3=self.w3)
        self.rl_agent = RLAgent(env=self.rl_env, cfg=cfg)

        # Load pre-trained weights if available
        self._load_models()

        logger.info(
            "AIMevBot initialised | chain=%s | models=%s",
            cfg.chain,
            ["price_predictor", "opp_classifier", "rl_agent"],
        )

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load model checkpoints if they exist."""
        if Path(self.cfg.price_predictor_checkpoint).exists():
            self.price_predictor.load(self.cfg.price_predictor_checkpoint)
            logger.info("Loaded price predictor from %s", self.cfg.price_predictor_checkpoint)

        if Path(self.cfg.classifier_checkpoint).exists():
            self.opp_classifier.load(self.cfg.classifier_checkpoint)
            logger.info("Loaded opportunity classifier from %s", self.cfg.classifier_checkpoint)

        if Path(self.cfg.rl_agent_checkpoint).exists():
            self.rl_agent.load(self.cfg.rl_agent_checkpoint)
            logger.info("Loaded RL agent from %s", self.cfg.rl_agent_checkpoint)

    def _save_models(self) -> None:
        """Persist model checkpoints."""
        self.price_predictor.save(self.cfg.price_predictor_checkpoint)
        self.opp_classifier.save(self.cfg.classifier_checkpoint)
        self.rl_agent.save(self.cfg.rl_agent_checkpoint)

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------

    async def _get_market_features(self) -> dict:
        """
        Collect on-chain market features for model inputs.

        Returns a dict with:
          - gas_price_gwei: current base fee
          - block_number: latest block
          - pending_tx_count: mempool size
          - (in production: OHLCV data from a price oracle or WebSocket stream)
        """
        gas_price = await self.w3.eth.gas_price
        block_number = await self.w3.eth.block_number
        return {
            "gas_price_gwei": gas_price / 1e9,
            "block_number": block_number,
            "timestamp": time.time(),
        }

    async def _get_pending_tx_features(self, tx_hash: str) -> dict:
        """
        Extract features from a pending transaction for the opportunity classifier.
        """
        try:
            tx = await self.w3.eth.get_transaction(tx_hash)
        except Exception:
            return {}

        return {
            "value_eth": float(tx.get("value", 0)) / 1e18,
            "gas_price_gwei": float(tx.get("gasPrice", 0)) / 1e9,
            "input_length": len(tx.get("input", b"")),
            "selector": tx["input"][:10].lower() if tx.get("input") and len(tx["input"]) >= 10 else "0x",
            "nonce": tx.get("nonce", 0),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main bot loop:
        1. Collect market features.
        2. Run price predictor → skip if bearish signal.
        3. Subscribe to pending txs → classify each with OpportunityClassifier.
        4. Build RL environment state → ask RLAgent for action.
        5. Execute the chosen action.
        6. Collect reward → online-update models periodically.
        """
        logger.info("AIMevBot starting main loop …")
        step = 0

        while True:
            try:
                # ---- Step 1: Market features ----
                market = await self._get_market_features()

                # ---- Step 2: Price prediction ----
                price_signal = await asyncio.get_event_loop().run_in_executor(
                    None, self.price_predictor.predict_from_market, market
                )
                logger.debug("Price signal: %s", price_signal)

                # Skip if strong bearish signal (protect capital)
                if price_signal == "down" and self.cfg.skip_on_bearish:
                    logger.info("Bearish price signal — skipping this block")
                    await asyncio.sleep(self.cfg.poll_interval)
                    continue

                # ---- Step 3: RL agent decides action ----
                obs = await self.rl_env.get_observation(market)
                action, _states = await asyncio.get_event_loop().run_in_executor(
                    None, self.rl_agent.predict, obs
                )

                # ---- Step 4: Execute action ----
                reward = await self.rl_env.execute_action(action, market)
                logger.info(
                    "Step=%d | action=%s | reward=%.6f ETH", step, action, reward
                )

                # ---- Step 5: Periodic model update ----
                step += 1
                if step % self.cfg.update_every_n_steps == 0:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.rl_agent.update
                    )
                    self._save_models()
                    logger.info("Models updated and saved at step %d", step)

            except KeyboardInterrupt:
                logger.info("Shutting down …")
                self._save_models()
                return
            except Exception as exc:
                logger.error("Main loop error: %s", exc, exc_info=True)

            await asyncio.sleep(self.cfg.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import AIBotConfig

    cfg = AIBotConfig()
    asyncio.run(AIMevBot(cfg).run())

"""
Offline training script for the Reinforcement Learning MEV agent.

Usage
-----
    python training/train_rl_agent.py \
        --total-timesteps 200000 \
        --checkpoint checkpoints/rl_agent

The agent trains in a fully simulated MevEnvironment.
Once trained, the checkpoint is loaded automatically by the live bot.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rl_agent")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def train(args):
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        logger.error(
            "stable-baselines3 and gymnasium are required for RL training.\n"
            "Install with: pip install stable-baselines3 gymnasium"
        )
        return

    # Import here to avoid issues when torch/gym not installed
    from models.rl_agent import MevEnvironment

    # Dummy config for offline training
    class DummyConfig:
        chain = "ethereum"
        min_profit_eth = 0.002
        max_gas_price_gwei = 100.0
        trade_amount_eth = 0.1
        rl_total_timesteps = args.total_timesteps
        update_every_n_steps = 500
        price_predictor_checkpoint = ""
        classifier_checkpoint = ""
        rl_agent_checkpoint = args.checkpoint
        flashbots_relay_url = ""
        flashbots_signer_key = ""
        skip_on_bearish = False
        poll_interval = 1.0
        price_predictor_input_size = 16
        price_predictor_hidden_size = 128
        price_predictor_num_layers = 2
        price_predictor_seq_len = 30
        classifier_feature_dim = 32

        def chain_config(self):
            return {"rpc_url": ""}

    cfg = DummyConfig()
    env = MevEnvironment(cfg=cfg, w3=None)

    logger.info("Checking environment …")
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    logger.info("Training PPO for %d timesteps …", args.total_timesteps)
    model.learn(total_timesteps=args.total_timesteps)

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.checkpoint)
    logger.info("RL agent saved to %s", args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--checkpoint", default="checkpoints/rl_agent")
    args = parser.parse_args()
    train(args)

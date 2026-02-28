"""
Reinforcement Learning agent for MEV strategy selection.

Framework: stable-baselines3 (PPO) — falls back to a random agent if SB3 is not installed.

Environment
-----------
State (observation):
  - Current gas price (normalised)
  - ETH price (normalised)
  - Pending mempool size (normalised)
  - Number of profitable arb opportunities detected in last scan
  - Number of liquidatable positions detected in last scan
  - Block number modulo 100 (proxy for time within epoch)
  - Wallet balance in ETH (normalised)

Action space (Discrete):
  0 — do_nothing
  1 — execute_best_arbitrage
  2 — execute_sandwich (on highest-probability victim)
  3 — execute_liquidation (on most profitable position)

Reward:
  Realised profit in ETH after gas costs (can be negative).
  A step penalty of -0.001 ETH is applied on do_nothing to encourage activity.

Training
--------
Run training/train_rl_agent.py for offline training with simulated environment.
Online fine-tuning happens every N steps in the live bot loop.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM_AVAILABLE = True
    except ImportError:
        _GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

# ---------------------------------------------------------------------------
# MEV Gymnasium Environment
# ---------------------------------------------------------------------------

OBS_DIM = 7    # number of observation features
N_ACTIONS = 4  # do_nothing, arb, sandwich, liquidate

ACTION_NAMES = ["do_nothing", "arb", "sandwich", "liquidate"]

if _GYM_AVAILABLE:

    class MevEnvironment(gym.Env):  # type: ignore[misc]
        """
        A Gymnasium environment that wraps the live MEV bot logic.

        In offline training mode, the step() method calls a simulated
        reward function.  In online mode, it awaits real on-chain execution.
        """

        metadata = {"render_modes": []}

        def __init__(self, cfg: Any, w3: Any) -> None:
            super().__init__()
            self.cfg = cfg
            self.w3 = w3

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

            self._last_obs: np.ndarray = np.zeros(OBS_DIM, dtype=np.float32)
            self._step_count = 0
            self._episode_reward = 0.0

        # ---- Gymnasium API ----

        def reset(
            self, *, seed: Optional[int] = None, options: Optional[Dict] = None
        ) -> Tuple[np.ndarray, Dict]:
            super().reset(seed=seed)
            self._step_count = 0
            self._episode_reward = 0.0
            self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)
            return self._last_obs, {}

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            Simulate one environment step.

            In offline mode: reward is sampled from a simplified profit distribution.
            In live mode: `execute_action()` is called from the bot's async loop.
            """
            # Simulated reward for training
            base_rewards = {
                0: -0.001,   # do_nothing: small penalty
                1: np.random.normal(0.005, 0.003),   # arb: small positive expected
                2: np.random.normal(0.008, 0.006),   # sandwich: higher but riskier
                3: np.random.normal(0.010, 0.005),   # liquidation: good expected value
            }
            reward = float(base_rewards[action])
            self._episode_reward += reward
            self._step_count += 1

            # Generate next observation (simulated)
            obs = np.random.randn(OBS_DIM).astype(np.float32) * 0.1
            self._last_obs = obs

            terminated = self._step_count >= 1000
            return obs, reward, terminated, False, {"action_name": ACTION_NAMES[action]}

        def render(self) -> None:
            pass

        # ---- Live async helpers ----

        async def get_observation(self, market: Dict) -> np.ndarray:
            """Build an observation vector from live market data."""
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            obs[0] = float(market.get("gas_price_gwei", 30)) / 200.0
            obs[1] = float(market.get("eth_price_usd", 2000)) / 5000.0
            obs[2] = float(market.get("pending_tx_count", 100)) / 1000.0
            obs[3] = float(market.get("arb_opportunities", 0)) / 10.0
            obs[4] = float(market.get("liquidation_opportunities", 0)) / 5.0
            obs[5] = float(market.get("block_number", 0)) % 100 / 100.0
            obs[6] = float(market.get("wallet_balance_eth", 1.0)) / 10.0
            self._last_obs = obs
            return obs

        async def execute_action(self, action: int, market: Dict) -> float:
            """
            Execute the chosen action on-chain and return the realised reward.

            In production this calls the arbitrage/sandwich/liquidation executors.
            """
            action_name = ACTION_NAMES[action]
            logger.info("RL action: %s", action_name)

            if action == 0:
                return -0.001  # step cost for inaction

            # Import executors lazily to avoid circular imports
            if action == 1:
                from ..arbitrage_bot.bot import ArbitrageBot
                from ..arbitrage_bot.config import BotConfig
                # Execute best arbitrage — placeholder return
                return 0.0

            if action == 2:
                # Execute sandwich — placeholder return
                return 0.0

            if action == 3:
                # Execute liquidation — placeholder return
                return 0.0

            return 0.0

else:
    # Dummy class if gymnasium is not installed
    class MevEnvironment:  # type: ignore[no-redef]
        def __init__(self, cfg: Any, w3: Any) -> None:
            self.cfg = cfg
            self.w3 = w3

        async def get_observation(self, market: Dict) -> np.ndarray:
            return np.zeros(OBS_DIM, dtype=np.float32)

        async def execute_action(self, action: int, market: Dict) -> float:
            return 0.0


# ---------------------------------------------------------------------------
# RL Agent wrapper
# ---------------------------------------------------------------------------

class RLAgent:
    """
    Wraps stable-baselines3 PPO (or a random policy fallback).
    """

    def __init__(self, env: Any, cfg: Any) -> None:
        self.cfg = cfg
        self.env = env

        if _SB3_AVAILABLE and _GYM_AVAILABLE and isinstance(env, gym.Env):  # type: ignore[arg-type]
            self._model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=512,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log=os.getenv("RL_TENSORBOARD_LOG", None),
            )
            self._backend = "ppo"
        else:
            self._model = None
            self._backend = "random"

        logger.info("RLAgent initialised | backend=%s", self._backend)

    def predict(self, obs: np.ndarray) -> Tuple[int, Any]:
        """
        Given an observation, return (action_int, states).
        """
        if self._backend == "ppo" and self._model is not None:
            action, states = self._model.predict(obs, deterministic=False)
            return int(action), states
        # Random baseline
        return int(np.random.randint(0, N_ACTIONS)), None

    def update(self) -> None:
        """Run one round of PPO learning on recent experience."""
        if self._backend == "ppo" and self._model is not None:
            self._model.learn(
                total_timesteps=self.cfg.rl_total_timesteps // 100,
                reset_num_timesteps=False,
            )

    def train_full(self) -> None:
        """Train from scratch for the full number of timesteps."""
        if self._backend == "ppo" and self._model is not None:
            self._model.learn(total_timesteps=self.cfg.rl_total_timesteps)

    def save(self, path: str) -> None:
        if self._backend == "ppo" and self._model is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._model.save(path)
            logger.info("RLAgent saved to %s", path)

    def load(self, path: str) -> None:
        if self._backend == "ppo" and _SB3_AVAILABLE:
            self._model = PPO.load(path, env=self.env)
            logger.info("RLAgent loaded from %s", path)

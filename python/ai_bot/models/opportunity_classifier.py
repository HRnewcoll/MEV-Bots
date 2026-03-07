"""
MEV Opportunity Classifier.

Architecture
------------
An MLP (multi-layer perceptron) that classifies pending transactions into:
  0 — NOT a profitable MEV opportunity (skip)
  1 — Potential sandwich target
  2 — Potential arbitrage trigger
  3 — Potential liquidation target

Alternatively, a gradient-boosted tree (scikit-learn) is used when PyTorch
is unavailable.

Input features (32-dimensional vector per transaction):
  0   log10(value_eth + 1e-18)      — transaction value
  1   log10(gas_price_gwei)          — urgency signal
  2   log10(input_length + 1)        — payload size
  3   selector_is_swap               — 1.0 if input matches known swap selectors
  4   selector_is_approve            — 1.0 if input matches ERC-20 approve
  5   nonce_norm                     — nonce / 1000 (proxy for account age)
  6   gas_limit_norm                 — gas_limit / 1e6
  7   mempool_age_ms_norm            — how long the tx has been pending / 1000
  8-31 (reserved for future features: address reputation, DEX-specific signals)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Selectors for common swap functions
SWAP_SELECTORS = {
    "0x38ed1739",  # swapExactTokensForTokens
    "0x7ff36ab5",  # swapExactETHForTokens
    "0x18cbafe5",  # swapExactTokensForETH
    "0x5c11d795",  # swapExactTokensForTokensSupportingFeeOnTransferTokens
    "0xfb3bdb41",  # swapETHForExactTokens
    "0x414bf389",  # exactInputSingle (V3)
    "0xc04b8d59",  # exactInput (V3)
}

CLASS_NAMES = ["no_mev", "sandwich_target", "arb_trigger", "liquidation_target"]


class _MLPModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    def __init__(self, feature_dim: int, num_classes: int = 4) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class OpportunityClassifier:
    """
    Classifies pending transactions as MEV opportunities.

    Uses a PyTorch MLP if available, otherwise falls back to a
    scikit-learn GradientBoostingClassifier.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_classes: int = 4,
        device: str = "cpu",
    ) -> None:
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device

        if _TORCH_AVAILABLE:
            self.model = _MLPModel(feature_dim, num_classes)
            self.model.to(device)
            self.model.eval()
            self._backend = "torch"
        elif _SKLEARN_AVAILABLE:
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
            )
            self._backend = "sklearn"
        else:
            self.model = None
            self._backend = "random"

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(tx_features: Dict, feature_dim: int = 32) -> np.ndarray:
        """Convert a transaction feature dict into a fixed-length numpy vector."""
        value_eth = tx_features.get("value_eth", 0.0)
        gas_price_gwei = tx_features.get("gas_price_gwei", 0.0)
        input_length = tx_features.get("input_length", 0)
        selector = tx_features.get("selector", "0x")
        nonce = tx_features.get("nonce", 0)
        gas_limit = tx_features.get("gas_limit", 200_000)
        mempool_age_ms = tx_features.get("mempool_age_ms", 0)

        vec = np.zeros(feature_dim, dtype=np.float32)
        vec[0] = np.log10(value_eth + 1e-18)
        vec[1] = np.log10(max(gas_price_gwei, 1e-9))
        vec[2] = np.log10(input_length + 1)
        vec[3] = 1.0 if selector in SWAP_SELECTORS else 0.0
        vec[4] = 1.0 if selector == "0x095ea7b3" else 0.0  # approve
        vec[5] = min(nonce / 1000.0, 10.0)
        vec[6] = min(gas_limit / 1e6, 1.0)
        vec[7] = min(mempool_age_ms / 1000.0, 60.0)
        return vec

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, tx_features: Dict) -> np.ndarray:
        """
        Return class probabilities [P(no_mev), P(sandwich), P(arb), P(liquidation)].
        """
        feat = self.extract_features(tx_features, self.feature_dim)

        if self._backend == "torch" and self.model is not None:
            import torch

            self.model.eval()
            x = torch.tensor(feat).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            return probs

        if self._backend == "sklearn" and hasattr(self.model, "classes_"):
            return self.model.predict_proba(feat.reshape(1, -1))[0]

        # Random baseline
        p = np.random.dirichlet(np.ones(self.num_classes))
        return p

    def predict(self, tx_features: Dict) -> str:
        """Return the most likely class name."""
        probs = self.predict_proba(tx_features)
        return CLASS_NAMES[int(np.argmax(probs))]

    def is_mev_opportunity(self, tx_features: Dict, threshold: float = 0.5) -> bool:
        """Return True if any MEV class probability exceeds *threshold*."""
        probs = self.predict_proba(tx_features)
        return bool(np.max(probs[1:]) >= threshold)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_batch(
        self, features: np.ndarray, labels: np.ndarray, lr: float = 1e-3
    ) -> float:
        """Train on a batch; return loss."""
        if self._backend == "torch" and self.model is not None:
            import torch
            import torch.nn as nn

            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            y = torch.tensor(labels, dtype=torch.long).to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            optimizer.step()
            self.model.eval()
            return float(loss.item())

        if self._backend == "sklearn":
            self.model.fit(features, labels)
            return 0.0

        return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._backend == "torch":
            import torch
            torch.save(self.model.state_dict(), path)
        else:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
        logger.info("OpportunityClassifier saved to %s", path)

    def load(self, path: str) -> None:
        if self._backend == "torch":
            import torch
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        else:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        logger.info("OpportunityClassifier loaded from %s", path)

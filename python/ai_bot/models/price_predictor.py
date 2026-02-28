"""
LSTM-based price direction predictor.

Architecture
------------
Input:  sequence of shape (batch, seq_len, input_size)
        Features per time-step:
          0  open_price
          1  high_price
          2  low_price
          3  close_price
          4  volume
          5  gas_price_gwei
          6  block_time_ms
          7  pending_tx_count
          8-15  (reserved for additional on-chain signals)

Output: logits of shape (batch, 3)  →  softmax → [P(down), P(flat), P(up)]

Training
--------
See training/train_price_model.py for the supervised training loop.
Label generation: if next-block mid-price moves > threshold → "up"; < -threshold → "down"; else "flat".
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — PricePredictor will use random baseline")


LABELS = ["down", "flat", "up"]


class _LSTMModel(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Stacked LSTM with a final linear classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ) -> None:
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):  # type: ignore[override]
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take last time-step hidden state
        last = self.norm(last)
        return self.head(last)


class PricePredictor:
    """
    Wraps the LSTM model with convenience methods for training and inference.

    If PyTorch is unavailable the predictor falls back to a random baseline so
    the rest of the bot can still run (useful for unit tests).
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 3,
        seq_len: int = 30,
        device: str = "cpu",
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_len = seq_len
        self.device = device

        # Rolling buffer for live inference
        self._history: List[List[float]] = []

        if _TORCH_AVAILABLE:
            self.model = _LSTMModel(input_size, hidden_size, num_layers, output_size)
            self.model.to(device)
            self.model.eval()
        else:
            self.model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, sequence: np.ndarray) -> str:
        """
        Predict price direction from a (seq_len, input_size) numpy array.

        Returns one of "down", "flat", "up".
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return np.random.choice(LABELS)

        import torch

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return LABELS[int(np.argmax(probs))]

    def predict_from_market(self, market_snapshot: Dict) -> str:
        """
        Update the rolling history buffer and predict from the latest snapshot.

        The snapshot dict should contain numeric features; missing ones default to 0.
        """
        feature_keys = [
            "open", "high", "low", "close", "volume",
            "gas_price_gwei", "block_time_ms", "pending_tx_count",
        ]
        row = [float(market_snapshot.get(k, 0.0)) for k in feature_keys]
        # Pad to input_size
        row += [0.0] * max(0, self.input_size - len(row))
        row = row[: self.input_size]

        self._history.append(row)
        if len(self._history) > self.seq_len:
            self._history.pop(0)

        if len(self._history) < self.seq_len:
            # Not enough history yet — predict flat
            return "flat"

        seq = np.array(self._history, dtype=np.float32)
        return self.predict(seq)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        lr: float = 1e-3,
    ) -> float:
        """
        Run one gradient-descent step on a batch.

        Args:
            sequences: (batch, seq_len, input_size)
            labels:    (batch,) integer labels in {0, 1, 2}
        Returns:
            cross-entropy loss value
        """
        if not _TORCH_AVAILABLE:
            return 0.0

        import torch
        import torch.nn as nn

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        optimizer.zero_grad()
        logits = self.model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        self.model.eval()
        return float(loss.item())

    def validate(self, sequences: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute cross-entropy loss on a validation batch without updating weights.

        Args:
            sequences: (batch, seq_len, input_size)
            labels:    (batch,) integer labels in {0, 1, 2}
        Returns:
            mean cross-entropy loss
        """
        if not _TORCH_AVAILABLE:
            return 0.0

        import torch
        import torch.nn as nn

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            y = torch.tensor(labels, dtype=torch.long).to(self.device)
            logits = self.model(x)
            loss = criterion(logits, y)
        return float(loss.item())


        if not _TORCH_AVAILABLE or self.model is None:
            return
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("PricePredictor saved to %s", path)

    def load(self, path: str) -> None:
        if not _TORCH_AVAILABLE or self.model is None:
            return
        import torch

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("PricePredictor loaded from %s", path)

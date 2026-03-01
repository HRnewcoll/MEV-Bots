"""
Trade Recorder — persists simulation history to a JSON file.

The recorder appends each trade as a JSON line (JSONL format) so the file
can be streamed and parsed without loading everything into memory.
It also maintains an in-memory buffer for fast recent-trade queries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional

from simulator.paper_wallet import Trade


DEFAULT_PATH = Path("simulator_data") / "trades.jsonl"


class TradeRecorder:
    """Persists trades to a JSONL file and provides query helpers."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: List[dict] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, trade: Trade) -> None:
        """Append a trade to the JSONL file and in-memory buffer."""
        record = {
            "ts": trade.timestamp,
            "strategy": trade.strategy,
            "token_in": trade.token_in,
            "token_out": trade.token_out,
            "amount_in": trade.amount_in,
            "amount_out": trade.amount_out,
            "gas_cost_eth": trade.gas_cost_eth,
            "profit_eth": trade.profit_eth,
            "buy_dex": trade.buy_dex,
            "sell_dex": trade.sell_dex,
            "block": trade.block_number,
            "simulated": trade.simulated,
            "notes": trade.notes,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._buffer.append(record)
        # Keep buffer bounded
        if len(self._buffer) > 1000:
            self._buffer = self._buffer[-1000:]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recent(self, n: int = 50) -> List[dict]:
        """Return *n* most recent trades from the in-memory buffer."""
        return list(reversed(self._buffer[-n:]))

    def all_trades(self) -> Iterator[dict]:
        """Iterate over all persisted trades (reads from disk)."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def load_history(self) -> List[dict]:
        """Load all historical trades into the in-memory buffer."""
        self._buffer = list(self.all_trades())
        return self._buffer

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def aggregate_stats(self) -> dict:
        """Compute aggregate P&L from the persisted history."""
        trades = list(self.all_trades())
        if not trades:
            return {
                "total_trades": 0,
                "total_profit_eth": 0.0,
                "total_gas_eth": 0.0,
                "win_rate": 0.0,
                "by_strategy": {},
            }
        profits = [t["profit_eth"] for t in trades]
        wins = sum(1 for p in profits if p > 0)
        by_strategy: dict = {}
        for t in trades:
            s = t["strategy"]
            if s not in by_strategy:
                by_strategy[s] = {"count": 0, "profit": 0.0, "gas": 0.0}
            by_strategy[s]["count"] += 1
            by_strategy[s]["profit"] += t["profit_eth"]
            by_strategy[s]["gas"] += t["gas_cost_eth"]
        return {
            "total_trades": len(trades),
            "total_profit_eth": sum(profits),
            "total_gas_eth": sum(t["gas_cost_eth"] for t in trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "by_strategy": by_strategy,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, out_path: str) -> None:
        """Export all trades to a CSV file."""
        import csv
        fieldnames = [
            "ts", "strategy", "token_in", "token_out", "amount_in",
            "amount_out", "gas_cost_eth", "profit_eth", "buy_dex",
            "sell_dex", "block", "simulated", "notes",
        ]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.all_trades():
                writer.writerow({k: trade.get(k, "") for k in fieldnames})

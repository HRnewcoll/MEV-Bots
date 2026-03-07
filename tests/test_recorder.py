"""Unit tests for simulator.recorder.TradeRecorder."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from simulator.paper_wallet import PaperWallet, Trade
from simulator.recorder import TradeRecorder


WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def _make_trade(strategy: str = "arbitrage", profit: float = 0.001) -> Trade:
    return Trade(
        timestamp=1_700_000_000.0,
        strategy=strategy,
        token_in=WETH,
        token_out=USDC,
        amount_in=0.1,
        amount_out=0.101,
        gas_cost_eth=0.0005,
        profit_eth=profit,
        buy_dex="uniswap_v2",
        sell_dex="sushiswap",
        block_number=18_000_001,
        simulated=True,
        notes="test",
    )


@pytest.fixture
def recorder(tmp_path):
    return TradeRecorder(path=str(tmp_path / "trades.jsonl"))


@pytest.fixture
def recorder_with_trades(tmp_path):
    r = TradeRecorder(path=str(tmp_path / "trades.jsonl"))
    r.record(_make_trade("arbitrage",  0.005))
    r.record(_make_trade("arbitrage",  0.002))
    r.record(_make_trade("sandwich",  -0.001))
    return r


# ---------------------------------------------------------------------------
# Basic record / read
# ---------------------------------------------------------------------------

def test_record_creates_file(recorder, tmp_path):
    recorder.record(_make_trade())
    assert (tmp_path / "trades.jsonl").exists()


def test_record_persists_json_line(recorder, tmp_path):
    recorder.record(_make_trade(profit=0.007))
    lines = (tmp_path / "trades.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert abs(data["profit_eth"] - 0.007) < 1e-12
    assert data["strategy"] == "arbitrage"


def test_record_appends_multiple_trades(recorder, tmp_path):
    for _ in range(3):
        recorder.record(_make_trade())
    lines = (tmp_path / "trades.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3


def test_record_populates_buffer(recorder):
    recorder.record(_make_trade())
    assert len(recorder._buffer) == 1


def test_buffer_capped_at_1000(recorder):
    for _ in range(1100):
        recorder.record(_make_trade())
    assert len(recorder._buffer) <= 1000


# ---------------------------------------------------------------------------
# recent()
# ---------------------------------------------------------------------------

def test_recent_empty(recorder):
    assert recorder.recent() == []


def test_recent_returns_n_items(recorder_with_trades):
    result = recorder_with_trades.recent(n=2)
    assert len(result) == 2


def test_recent_most_recent_first(recorder_with_trades):
    # sandwich trade was recorded last — should be first in recent()
    result = recorder_with_trades.recent(n=10)
    assert result[0]["strategy"] == "sandwich"


# ---------------------------------------------------------------------------
# all_trades()
# ---------------------------------------------------------------------------

def test_all_trades_yields_correct_count(recorder_with_trades):
    trades = list(recorder_with_trades.all_trades())
    assert len(trades) == 3


def test_all_trades_empty_for_nonexistent_file(tmp_path):
    r = TradeRecorder(path=str(tmp_path / "nonexistent.jsonl"))
    assert list(r.all_trades()) == []


def test_all_trades_skips_invalid_json_lines(recorder, tmp_path):
    # Write one valid + one invalid line
    path = tmp_path / "trades.jsonl"
    path.write_text('{"strategy":"arbitrage","profit_eth":0.001}\n{invalid json}\n')
    r = TradeRecorder(path=str(path))
    trades = list(r.all_trades())
    assert len(trades) == 1


# ---------------------------------------------------------------------------
# load_history()
# ---------------------------------------------------------------------------

def test_load_history_populates_buffer(recorder_with_trades, tmp_path):
    # Create a fresh recorder pointing at the same file
    r2 = TradeRecorder(path=recorder_with_trades.path)
    r2.load_history()
    assert len(r2._buffer) == 3


# ---------------------------------------------------------------------------
# aggregate_stats()
# ---------------------------------------------------------------------------

def test_aggregate_stats_empty(recorder):
    stats = recorder.aggregate_stats()
    assert stats["total_trades"] == 0
    assert stats["total_profit_eth"] == 0.0
    assert stats["by_strategy"] == {}


def test_aggregate_stats_totals(recorder_with_trades):
    stats = recorder_with_trades.aggregate_stats()
    assert stats["total_trades"] == 3
    # profits: 0.005 + 0.002 + (-0.001) = 0.006
    assert abs(stats["total_profit_eth"] - 0.006) < 1e-12


def test_aggregate_stats_win_rate(recorder_with_trades):
    stats = recorder_with_trades.aggregate_stats()
    # 2 wins (arb x2) out of 3
    assert abs(stats["win_rate"] - 2 / 3) < 1e-12


def test_aggregate_stats_by_strategy(recorder_with_trades):
    stats = recorder_with_trades.aggregate_stats()
    assert "arbitrage" in stats["by_strategy"]
    assert "sandwich" in stats["by_strategy"]
    arb = stats["by_strategy"]["arbitrage"]
    assert arb["count"] == 2
    assert abs(arb["profit"] - 0.007) < 1e-12


# ---------------------------------------------------------------------------
# export_csv()
# ---------------------------------------------------------------------------

def test_export_csv_creates_file(recorder_with_trades, tmp_path):
    out = str(tmp_path / "export.csv")
    recorder_with_trades.export_csv(out)
    assert Path(out).exists()


def test_export_csv_header(recorder_with_trades, tmp_path):
    out = str(tmp_path / "export.csv")
    recorder_with_trades.export_csv(out)
    first_line = Path(out).read_text().splitlines()[0]
    assert "strategy" in first_line
    assert "profit_eth" in first_line


def test_export_csv_row_count(recorder_with_trades, tmp_path):
    out = str(tmp_path / "export.csv")
    recorder_with_trades.export_csv(out)
    lines = Path(out).read_text().strip().splitlines()
    # header + 3 data rows
    assert len(lines) == 4


def test_export_csv_empty_file_for_no_trades(recorder, tmp_path):
    out = str(tmp_path / "export.csv")
    recorder.export_csv(out)
    # Only header line
    lines = Path(out).read_text().strip().splitlines()
    assert len(lines) == 1

"""
Integration tests for the Flask dashboard API.

Uses Flask's built-in test client — no server needs to be running.
All tests operate without starting the simulation engine (no network calls).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a Flask test client with a temporary simulator_data directory."""
    tmp = tmp_path_factory.mktemp("simdata")

    # Point the app's recorder_path and simulator_data at a temp directory
    import ui.app as app_module

    # Patch _REPO_ROOT to use tmp dir so setup_check doesn't look at real dirs
    original_repo_root = app_module._REPO_ROOT
    app_module._REPO_ROOT = tmp

    # Create the necessary structure under tmp so setup_check works
    (tmp / "simulator_data").mkdir(exist_ok=True)
    (tmp / "python").mkdir(exist_ok=True)

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c

    # Restore
    app_module._REPO_ROOT = original_repo_root


# ---------------------------------------------------------------------------
# GET /api/status  (no engine running)
# ---------------------------------------------------------------------------

def test_status_default(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "running" in data
    assert data["running"] is False


def test_status_has_required_keys(client):
    data = client.get("/api/status").get_json()
    for key in ("running", "paused", "step", "strategies", "elapsed_s"):
        assert key in data, f"Key '{key}' missing from /api/status"


# ---------------------------------------------------------------------------
# GET /api/trades
# ---------------------------------------------------------------------------

def test_trades_returns_list(client):
    data = client.get("/api/trades").get_json()
    assert isinstance(data, list)


def test_trades_empty_when_no_engine(client):
    data = client.get("/api/trades").get_json()
    assert data == []


def test_trades_n_parameter_accepted(client):
    resp = client.get("/api/trades?n=5")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/events
# ---------------------------------------------------------------------------

def test_events_returns_list(client):
    data = client.get("/api/events").get_json()
    assert isinstance(data, list)


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

def test_stats_returns_dict(client):
    data = client.get("/api/stats").get_json()
    assert isinstance(data, dict)
    assert "total_trades" in data


def test_stats_zero_when_empty(client):
    data = client.get("/api/stats").get_json()
    assert data["total_trades"] == 0


# ---------------------------------------------------------------------------
# GET /api/chains
# ---------------------------------------------------------------------------

def test_chains_returns_dict(client):
    data = client.get("/api/chains").get_json()
    assert isinstance(data, dict)


def test_chains_contains_known_chains(client):
    data = client.get("/api/chains").get_json()
    for chain in ("ethereum", "bsc", "polygon"):
        assert chain in data, f"Chain '{chain}' missing from /api/chains"


def test_chains_has_dexs_and_pairs(client):
    data = client.get("/api/chains").get_json()
    for chain, info in data.items():
        assert "dexs" in info,  f"{chain} missing 'dexs'"
        assert "pairs" in info, f"{chain} missing 'pairs'"
        assert isinstance(info["dexs"], list)
        assert isinstance(info["pairs"], list)


# ---------------------------------------------------------------------------
# GET /api/setup_check
# ---------------------------------------------------------------------------

def test_setup_check_returns_dict(client):
    data = client.get("/api/setup_check").get_json()
    assert isinstance(data, dict)


def test_setup_check_has_all_ok_and_checks(client):
    data = client.get("/api/setup_check").get_json()
    assert "all_ok" in data
    assert "checks" in data
    assert isinstance(data["checks"], list)


def test_setup_check_items_have_required_fields(client):
    data = client.get("/api/setup_check").get_json()
    for item in data["checks"]:
        assert "id" in item
        assert "label" in item
        assert "ok" in item
        assert isinstance(item["ok"], bool)


def test_setup_check_all_ok_reflects_items(client):
    data = client.get("/api/setup_check").get_json()
    computed = all(c["ok"] for c in data["checks"])
    assert data["all_ok"] == computed


# ---------------------------------------------------------------------------
# POST /api/stop  (no engine running)
# ---------------------------------------------------------------------------

def test_stop_no_engine(client):
    resp = client.post("/api/stop")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is False


# ---------------------------------------------------------------------------
# POST /api/pause  (no engine running)
# ---------------------------------------------------------------------------

def test_pause_no_engine(client):
    resp = client.post("/api/pause")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is False


# ---------------------------------------------------------------------------
# GET /  (dashboard page)
# ---------------------------------------------------------------------------

def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"MEV Bot Dashboard" in resp.data


# ---------------------------------------------------------------------------
# Content-type checks
# ---------------------------------------------------------------------------

def test_api_endpoints_return_json_content_type(client):
    endpoints = [
        "/api/status", "/api/trades", "/api/events",
        "/api/stats", "/api/chains", "/api/setup_check",
    ]
    for ep in endpoints:
        resp = client.get(ep)
        assert "application/json" in resp.content_type, \
            f"{ep} should return application/json, got {resp.content_type}"

"""
MEV Bots Dashboard — Flask web application.

Provides:
  GET  /                   — Main dashboard page
  GET  /api/status         — JSON: current engine status + wallet stats
  GET  /api/trades         — JSON: recent simulated trades
  GET  /api/events         — JSON: recent event log
  GET  /api/stats          — JSON: aggregate P&L from recorder
  POST /api/start          — Start / restart simulation with given config
  POST /api/stop           — Stop the running simulation
  POST /api/pause          — Pause / resume
  GET  /api/stream         — SSE stream of real-time events
  GET  /api/export/csv     — Download trades as CSV
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

# Make the repo root importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from simulator.engine import SimConfig, SimulationEngine
from simulator.recorder import TradeRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ui")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Global simulation state (one engine at a time)
# ---------------------------------------------------------------------------

_engine: SimulationEngine | None = None
_sim_thread: threading.Thread | None = None
_event_queue: queue.Queue = queue.Queue(maxsize=1000)
_recorder = TradeRecorder()


def _event_callback(event_type: str, data: dict) -> None:
    """Push events to the SSE queue."""
    try:
        _event_queue.put_nowait({"type": event_type, "ts": time.time(), **data})
    except queue.Full:
        pass


def _run_simulation(cfg: SimConfig) -> None:
    """Run the async simulation engine in its own thread + event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_engine.run())
    except Exception as exc:
        logger.error("Simulation thread error: %s", exc)
    finally:
        loop.close()


def _shutdown() -> None:
    """Called by atexit to stop a running simulation gracefully so the
    TradeRecorder can flush any in-flight data before process exit."""
    if _engine and _engine._running:
        logger.info("atexit: stopping simulation engine …")
        _engine.stop()
        if _sim_thread and _sim_thread.is_alive():
            _sim_thread.join(timeout=5)


atexit.register(_shutdown)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API — Read
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    if _engine is None:
        return jsonify({"running": False, "paused": False, "step": 0,
                        "chain": "—", "strategies": [], "elapsed_s": 0,
                        "wallet": {}, "cache": {}, "recent_errors": []})
    return jsonify(_engine.status())


@app.route("/api/trades")
def api_trades():
    n = int(request.args.get("n", 30))
    if _engine:
        return jsonify(_engine.wallet.recent_trades(n))
    return jsonify([])


@app.route("/api/events")
def api_events():
    n = int(request.args.get("n", 50))
    if _engine:
        return jsonify(_engine.recent_events(n))
    return jsonify([])


@app.route("/api/stats")
def api_stats():
    return jsonify(_recorder.aggregate_stats())


@app.route("/api/chains")
def api_chains():
    from simulator.engine import CHAIN_DEXS
    return jsonify({
        chain: {
            "dexs": list(info.get("dexs", {}).keys()),
            "pairs": [f"{a[-6:]}/{b[-6:]}" for a, b in info.get("token_pairs", [])],
        }
        for chain, info in CHAIN_DEXS.items()
    })


@app.route("/api/setup_check")
def api_setup_check():
    """Return a checklist of setup items so the UI can guide new users."""
    import importlib

    checks = []

    # Python packages
    for pkg in ("flask", "web3", "aiohttp", "dotenv"):
        ok = False
        try:
            importlib.import_module(pkg)
            ok = True
        except ImportError:
            pass
        checks.append({"id": f"pkg_{pkg}", "label": f"Python package: {pkg}",
                        "ok": ok, "fix": f"pip install {pkg}"})

    # simulator_data directory
    sim_data = _REPO_ROOT / "simulator_data"
    checks.append({
        "id": "sim_data_dir",
        "label": "simulator_data/ directory exists",
        "ok": sim_data.is_dir(),
        "fix": "mkdir simulator_data  (or run: python run.py)",
    })

    # .env files
    for bot in ("arbitrage_bot", "sandwich_bot", "liquidation_bot", "flash_loan_bot"):
        env_path = _REPO_ROOT / "python" / bot / ".env"
        example_path = env_path.with_suffix(".env.example")
        has_env = env_path.exists()
        has_rpc = False
        if has_env:
            try:
                content = env_path.read_text()
                has_rpc = "RPC_URL" in content and "your" not in content.lower()
            except Exception:
                pass
        checks.append({
            "id": f"env_{bot}",
            "label": f"python/{bot}/.env exists",
            "ok": has_env,
            "fix": f"cp python/{bot}/.env.example python/{bot}/.env  then edit it",
        })
        if has_env:
            checks.append({
                "id": f"rpc_{bot}",
                "label": f"python/{bot}/.env — RPC URL configured",
                "ok": has_rpc,
                "fix": f"Edit python/{bot}/.env and set ETH_RPC_URL (or chain equivalent)",
            })

    # Simulator_data writable
    try:
        test_file = sim_data / ".write_test"
        test_file.touch()
        test_file.unlink()
        writable = True
    except Exception:
        writable = False
    checks.append({
        "id": "sim_data_writable",
        "label": "simulator_data/ is writable",
        "ok": writable,
        "fix": "chmod 755 simulator_data",
    })

    all_ok = all(c["ok"] for c in checks)
    return jsonify({"all_ok": all_ok, "checks": checks})


# ---------------------------------------------------------------------------
# API — Control
# ---------------------------------------------------------------------------

@app.route("/api/start", methods=["POST"])
def api_start():
    global _engine, _sim_thread, _recorder

    # Stop existing simulation if running
    if _engine and _engine._running:
        _engine.stop()
        if _sim_thread:
            _sim_thread.join(timeout=5)

    body = request.get_json(silent=True) or {}

    cfg = SimConfig(
        chain=body.get("chain", "ethereum"),
        rpc_url=body.get("rpc_url", ""),
        strategies=body.get("strategies", ["arbitrage"]),
        initial_eth=float(body.get("initial_eth", 10.0)),
        trade_amount_eth=float(body.get("trade_amount_eth", 0.1)),
        min_profit_eth=float(body.get("min_profit_eth", 0.0005)),
        max_gas_price_gwei=float(body.get("max_gas_price_gwei", 100.0)),
        slippage_bps=int(body.get("slippage_bps", 50)),
        poll_interval_s=float(body.get("poll_interval_s", 2.0)),
        event_callback=_event_callback,
        recorder_path=str(_REPO_ROOT / "simulator_data" / "trades.jsonl"),
    )

    _recorder = TradeRecorder(path=cfg.recorder_path)
    _engine = SimulationEngine(cfg)

    _sim_thread = threading.Thread(target=_run_simulation, args=(cfg,), daemon=True)
    # daemon=True so the thread does not block interpreter shutdown; the
    # atexit handler above calls _engine.stop() first to allow a clean flush.
    _sim_thread.start()

    return jsonify({"ok": True, "chain": cfg.chain, "strategies": cfg.strategies})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    if _engine:
        _engine.stop()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "No simulation running"})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    if _engine is None:
        return jsonify({"ok": False, "msg": "No simulation running"})
    if _engine._paused:
        _engine.resume()
        return jsonify({"ok": True, "paused": False})
    else:
        _engine.pause()
        return jsonify({"ok": True, "paused": True})


# ---------------------------------------------------------------------------
# SSE — real-time event stream
# ---------------------------------------------------------------------------

@app.route("/api/stream")
def api_stream():
    def generate():
        while True:
            try:
                event = _event_queue.get(timeout=20)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send a keep-alive comment
                yield ": keep-alive\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.route("/api/export/csv")
def api_export_csv():
    out = str(_REPO_ROOT / "simulator_data" / "trades_export.csv")
    _recorder.export_csv(out)
    return send_file(out, as_attachment=True, download_name="mev_sim_trades.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("UI_PORT", "5000"))
    host = os.getenv("UI_HOST", "127.0.0.1")
    debug = os.getenv("UI_DEBUG", "false").lower() == "true"
    logger.info("MEV Bot Dashboard starting on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug, threaded=True)

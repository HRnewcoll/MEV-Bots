#!/usr/bin/env python3
"""
run.py — One-command launcher for the MEV Bot Dashboard.

Usage
-----
    python run.py                  # start on default port 5000
    python run.py --port 8080      # custom port
    python run.py --no-browser     # don't auto-open browser
    python run.py --host 0.0.0.0   # listen on all interfaces (LAN access)
    python run.py --debug          # enable Flask debug / hot-reload

What this script does
---------------------
1. Checks that Python 3.10+ is being used.
2. Installs any missing Python dependencies from requirements.txt.
3. Creates the simulator_data/ directory if needed.
4. Copies the .env.example files to .env if no .env file exists yet.
5. Starts the Flask dashboard server.
6. Auto-opens the dashboard in the default browser (unless --no-browser).
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
UI_DIR = REPO_ROOT / "ui"
SIM_DATA_DIR = REPO_ROOT / "simulator_data"

REQUIRED_PACKAGES = {
    "flask": "flask",
    "web3": "web3",
    "aiohttp": "aiohttp",
    "dotenv": "python-dotenv",
}

BOLD  = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"


def _c(color: str, msg: str) -> str:
    """Apply ANSI colour if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{color}{msg}{RESET}"
    return msg


def info(msg: str)  -> None: print(_c(BOLD, f"[run] ") + msg)
def ok(msg: str)    -> None: print(_c(GREEN, f"  ✓ ") + msg)
def warn(msg: str)  -> None: print(_c(YELLOW, f"  ⚠ ") + msg)
def error(msg: str) -> None: print(_c(RED, f"  ✗ ") + msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1 — Python version check
# ---------------------------------------------------------------------------

def check_python() -> None:
    info("Checking Python version …")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        error(f"Python 3.10+ required (found {major}.{minor}). "
              "Download from https://python.org")
        sys.exit(1)
    ok(f"Python {major}.{minor}")


# ---------------------------------------------------------------------------
# Step 2 — Install missing dependencies
# ---------------------------------------------------------------------------

def install_deps() -> None:
    info("Checking Python dependencies …")

    missing: list[str] = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        ok("All dependencies already installed")
        return

    warn(f"Missing packages: {', '.join(missing)}")
    info(f"Installing from {REQUIREMENTS.name} …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS), "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error("pip install failed:\n" + result.stderr)
        sys.exit(1)
    ok("Dependencies installed")


# ---------------------------------------------------------------------------
# Step 3 — Create runtime directories
# ---------------------------------------------------------------------------

def create_dirs() -> None:
    info("Checking runtime directories …")
    SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok(f"simulator_data/ ready")


# ---------------------------------------------------------------------------
# Step 4 — Copy .env.example → .env if needed
# ---------------------------------------------------------------------------

def init_env_files() -> None:
    info("Checking .env configuration files …")
    env_examples = list(REPO_ROOT.glob("python/*/.env.example"))
    created = []
    for example in env_examples:
        env_file = example.with_suffix("")  # removes .example
        if not env_file.exists():
            shutil.copy(example, env_file)
            created.append(str(env_file.relative_to(REPO_ROOT)))

    if created:
        warn("Created default .env files (edit these with your RPC URL / keys):")
        for f in created:
            print(f"      {f}")
    else:
        ok(".env files already exist")


# ---------------------------------------------------------------------------
# Step 5 — Launch Flask server
# ---------------------------------------------------------------------------

def launch(host: str, port: int, debug: bool, open_browser: bool) -> None:
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
    info(f"Starting MEV Bot Dashboard on {_c(BOLD, url)} …")
    print()

    # Add repo root and ui directory to path so imports work
    sys.path.insert(0, str(REPO_ROOT))

    # Delayed browser open (give Flask 1.5 s to bind)
    if open_browser:
        import threading
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    # Import and run the Flask app
    os.environ.setdefault("UI_HOST", host)
    os.environ.setdefault("UI_PORT", str(port))
    if debug:
        os.environ["UI_DEBUG"] = "true"

    # Change working directory so relative paths inside ui/app.py resolve correctly
    os.chdir(str(REPO_ROOT))

    from ui.app import app as flask_app
    flask_app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=debug)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MEV Bot Dashboard launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                  Start on http://127.0.0.1:5000
  python run.py --port 8080      Start on port 8080
  python run.py --host 0.0.0.0   Expose on local network
  python run.py --no-browser     Don't auto-open browser
  python run.py --debug          Hot-reload mode
""",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to listen on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug / hot-reload mode")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip dependency installation check")
    args = parser.parse_args()

    print()
    print(_c(BOLD, "⚡ MEV Bot Dashboard — Launcher"))
    print(_c(BOLD, "=" * 40))
    print()

    check_python()
    if not args.skip_install:
        install_deps()
    create_dirs()
    init_env_files()

    print()
    launch(
        host=args.host,
        port=args.port,
        debug=args.debug,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()

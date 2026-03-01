"""
Simulator package — paper-trading engine using real mainnet data.

Components
----------
engine.py       — Simulation loop that mirrors every bot strategy
paper_wallet.py — Virtual wallet with simulated token balances
market_data.py  — Read-only live data fetcher with caching
recorder.py     — Persistent trade-history log (JSON)
"""

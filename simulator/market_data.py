"""
Market Data — live on-chain data fetcher with LRU caching.

Wraps read-only calls to mainnet (or any EVM chain) via an async HTTP
provider and caches results to reduce RPC load.

Key optimisations over the baseline arbitrage_bot:
  - Pair address cache (addresses never change once deployed)
  - Reserve TTL cache (invalidated per-block)
  - Batch RPC calls using web3.py's ``batch_requests`` where supported
  - Shared provider instance (connection pooling via aiohttp session)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Dict, Optional, Tuple

from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal ABIs
# ---------------------------------------------------------------------------

FACTORY_ABI = [
    {
        "constant": True,
        "inputs": [
            {"name": "tokenA", "type": "address"},
            {"name": "tokenB", "type": "address"},
        ],
        "name": "getPair",
        "outputs": [{"name": "pair", "type": "address"}],
        "type": "function",
    }
]

PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "_reserve0", "type": "uint112"},
            {"name": "_reserve1", "type": "uint112"},
            {"name": "_blockTimestampLast", "type": "uint32"},
        ],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "type": "function",
    },
]

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]

NULL_ADDRESS = "0x0000000000000000000000000000000000000000"


class MarketData:
    """
    Read-only live market data with multi-level caching.

    Cache levels:
      1. ``_pair_cache``    — pair address per (factory, tokenA, tokenB); permanent
      2. ``_token0_cache``  — token0 address per pair; permanent
      3. ``_reserve_cache`` — reserves per pair; invalidated every *reserve_ttl_s* seconds
      4. ``_decimals_cache``— token decimals; permanent
    """

    def __init__(
        self,
        rpc_url: str,
        chain: str = "ethereum",
        reserve_ttl_s: float = 2.0,
    ) -> None:
        self.rpc_url = rpc_url
        self.chain = chain
        self.reserve_ttl_s = reserve_ttl_s

        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        if chain in ("bsc", "polygon", "avalanche"):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        # Caches
        self._pair_cache: Dict[str, str] = {}           # key → pair_address
        self._token0_cache: Dict[str, str] = {}         # pair_address → token0
        self._reserve_cache: Dict[str, Tuple] = {}      # pair_address → (r0, r1, ts, fetched_at)
        self._decimals_cache: Dict[str, int] = {}       # token_address → decimals
        self._symbol_cache: Dict[str, str] = {}         # token_address → symbol
        # Per-pair locks prevent duplicate in-flight RPC calls for the same pair
        self._reserve_locks: Dict[str, asyncio.Lock] = {}

        self._block_number: int = 0
        self._gas_price_gwei: float = 0.0
        self._chain_info_ts: float = 0.0

    # ------------------------------------------------------------------
    # Chain metadata (cached 3 s)
    # ------------------------------------------------------------------

    async def get_chain_info(self) -> Dict:
        """Return current block number and gas price (cached 3 s)."""
        now = time.monotonic()
        if now - self._chain_info_ts > 3.0:
            block, gas = await asyncio.gather(
                self.w3.eth.block_number,
                self.w3.eth.gas_price,
            )
            self._block_number = int(block)
            self._gas_price_gwei = int(gas) / 1e9
            self._chain_info_ts = now
        return {
            "block_number": self._block_number,
            "gas_price_gwei": self._gas_price_gwei,
        }

    # ------------------------------------------------------------------
    # Pair address (permanent cache — addresses never change)
    # ------------------------------------------------------------------

    async def get_pair_address(
        self, factory_address: str, token_a: str, token_b: str
    ) -> Optional[str]:
        key = f"{factory_address.lower()}:{token_a.lower()}:{token_b.lower()}"
        if key in self._pair_cache:
            return self._pair_cache[key]

        factory = self.w3.eth.contract(
            address=self.w3.to_checksum_address(factory_address),
            abi=FACTORY_ABI,
        )
        pair = await factory.functions.getPair(
            self.w3.to_checksum_address(token_a),
            self.w3.to_checksum_address(token_b),
        ).call()

        if pair == NULL_ADDRESS:
            self._pair_cache[key] = None
            return None

        self._pair_cache[key] = pair
        return pair

    # ------------------------------------------------------------------
    # Reserves (TTL cache — stale after reserve_ttl_s seconds)
    # ------------------------------------------------------------------

    async def get_reserves(
        self, pair_address: str
    ) -> Tuple[int, int]:
        """Return (reserve0, reserve1), refreshed every *reserve_ttl_s* seconds.

        A per-pair asyncio.Lock prevents duplicate in-flight RPC calls when
        multiple coroutines request the same pair concurrently (e.g. via
        batch_get_reserves).
        """
        now = time.monotonic()
        cached = self._reserve_cache.get(pair_address)
        if cached and now - cached[3] < self.reserve_ttl_s:
            return cached[0], cached[1]

        # Acquire a per-pair lock so only one coroutine fetches at a time
        if pair_address not in self._reserve_locks:
            self._reserve_locks[pair_address] = asyncio.Lock()
        async with self._reserve_locks[pair_address]:
            # Re-check cache after acquiring the lock (another coroutine may
            # have already populated it while we were waiting)
            cached = self._reserve_cache.get(pair_address)
            if cached and time.monotonic() - cached[3] < self.reserve_ttl_s:
                return cached[0], cached[1]

            pair = self.w3.eth.contract(
                address=self.w3.to_checksum_address(pair_address),
                abi=PAIR_ABI,
            )
            r0, r1, ts = await pair.functions.getReserves().call()
            self._reserve_cache[pair_address] = (r0, r1, ts, time.monotonic())
            return r0, r1

    async def get_token0(self, pair_address: str) -> str:
        """Return the token0 address for a pair (cached permanently)."""
        if pair_address in self._token0_cache:
            return self._token0_cache[pair_address]
        pair = self.w3.eth.contract(
            address=self.w3.to_checksum_address(pair_address),
            abi=PAIR_ABI,
        )
        t0 = await pair.functions.token0().call()
        self._token0_cache[pair_address] = t0
        return t0

    # ------------------------------------------------------------------
    # Batch reserve fetching (parallel coroutines)
    # ------------------------------------------------------------------

    async def batch_get_reserves(
        self, pair_addresses: list
    ) -> Dict[str, Tuple[int, int]]:
        """
        Fetch reserves for multiple pairs concurrently.
        Returns dict: pair_address → (reserve0, reserve1).
        """
        results = await asyncio.gather(
            *[self.get_reserves(p) for p in pair_addresses],
            return_exceptions=True,
        )
        return {
            addr: res if not isinstance(res, Exception) else (0, 0)
            for addr, res in zip(pair_addresses, results)
        }

    # ------------------------------------------------------------------
    # Token metadata
    # ------------------------------------------------------------------

    async def get_decimals(self, token_address: str) -> int:
        key = token_address.lower()
        if key in self._decimals_cache:
            return self._decimals_cache[key]
        try:
            token = self.w3.eth.contract(
                address=self.w3.to_checksum_address(token_address),
                abi=ERC20_ABI,
            )
            decimals = await token.functions.decimals().call()
            self._decimals_cache[key] = decimals
            return decimals
        except Exception:
            self._decimals_cache[key] = 18
            return 18

    async def get_symbol(self, token_address: str) -> str:
        key = token_address.lower()
        if key in self._symbol_cache:
            return self._symbol_cache[key]
        try:
            token = self.w3.eth.contract(
                address=self.w3.to_checksum_address(token_address),
                abi=ERC20_ABI,
            )
            symbol = await token.functions.symbol().call()
            self._symbol_cache[key] = symbol
            return symbol
        except Exception:
            short = token_address[-6:].upper()
            self._symbol_cache[key] = short
            return short

    # ------------------------------------------------------------------
    # Uniswap V2 price simulation (pure, no network call)
    # ------------------------------------------------------------------

    @staticmethod
    def get_amount_out(amount_in: int, reserve_in: int, reserve_out: int) -> int:
        """Uniswap V2 constant-product formula (0.3 % fee)."""
        if reserve_in == 0 or reserve_out == 0 or amount_in == 0:
            return 0
        amount_in_with_fee = amount_in * 997
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 1000 + amount_in_with_fee
        return numerator // denominator

    async def quote(
        self,
        factory_address: str,
        token_a: str,
        token_b: str,
        amount_in: int,
    ) -> int:
        """
        Return the simulated output amount for *amount_in* of token_a → token_b.
        Uses live on-chain reserves.
        """
        pair = await self.get_pair_address(factory_address, token_a, token_b)
        if pair is None:
            return 0
        r0, r1 = await self.get_reserves(pair)
        t0 = await self.get_token0(pair)
        if t0.lower() == token_a.lower():
            reserve_in, reserve_out = r0, r1
        else:
            reserve_in, reserve_out = r1, r0
        return self.get_amount_out(amount_in, reserve_in, reserve_out)

    # ------------------------------------------------------------------
    # Cache stats (for UI display)
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        return {
            "pair_cache_size": len(self._pair_cache),
            "reserve_cache_size": len(self._reserve_cache),
            "token_cache_size": len(self._decimals_cache),
        }

#!/usr/bin/env python3
"""
deribit_pricer.py

WebSocket-only Deribit barrier probability pricer (live streaming).
- Continuous updates every interval seconds.
- Outputs JSON lines to stdout for integration.
- Displays live bar chart of barrier probabilities.
- Run separate instances for multiple assets if needed.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import websockets
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------- Config ----------
DERIBIT_WS = "wss://www.deribit.com/ws/api/v2"
WS_RPC_TIMEOUT = 8.0
MIN_IV = 0.5
MAX_IV = 1000.0
MONTHLY_DAY_MIN = 24
MONTHLY_DAY_MAX = 3  # day <= 3 also considered monthly-start
DEFAULT_INTERVAL = 5  # seconds between outputs
MAX_POINTS = 100  # keep last N points per instrument
RECONNECT_DELAY = 5  # seconds to wait before reconnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deribit_pricer")

# Set matplotlib backend for macOS compatibility
plt.switch_backend('TkAgg')  # Use Tkinter backend, common for macOS

# ---------- Math helpers (r=0, q=0) ----------
def prob_hit_upper(F: np.ndarray, H: float, T: np.ndarray, sigma: np.ndarray, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    nu = r - q - 0.5 * sigma**2
    mask = (F >= H)
    result = np.zeros_like(F)
    result[mask] = 1.0
    valid = ~mask & (F > 0) & (H > 0) & (T > 0) & (sigma > 0)
    if not valid.any():
        return result
    b = np.log(H / F[valid])
    denom = sigma[valid] * np.sqrt(T[valid])
    term1 = norm.cdf((-b + nu[valid] * T[valid]) / denom)
    term2 = np.power(H / F[valid], 2 * nu[valid] / sigma[valid]**2) * norm.cdf((-b - nu[valid] * T[valid]) / denom)
    p = term1 + term2
    result[valid] = np.clip(p, 0, 1)
    return result


def prob_hit_lower(F: np.ndarray, L: float, T: np.ndarray, sigma: np.ndarray, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    nu = r - q - 0.5 * sigma**2
    mask = (F <= L)
    result = np.zeros_like(F)
    result[mask] = 1.0
    valid = ~mask & (F > 0) & (L > 0) & (T > 0) & (sigma > 0)
    if not valid.any():
        return result
    b = np.log(L / F[valid])
    denom = sigma[valid] * np.sqrt(T[valid])
    term1 = norm.cdf((b + nu[valid] * T[valid]) / denom)
    power_term = np.power(L / F[valid], 2 * nu[valid] / sigma[valid]**2)
    cdf_term = norm.cdf((b - nu[valid] * T[valid]) / denom)
    term2 = power_term * cdf_term
    p = term1 + term2
    result[valid] = np.clip(p, 0, 1)
    return result


# ---------- Utilities ----------
def now_unix() -> int:
    return int(time.time())


def iso_from_unix(s: int) -> str:
    return datetime.fromtimestamp(int(s), tz=timezone.utc).isoformat()


def ms_to_s_if_needed(ts: int) -> int:
    if ts > 10**12:
        return int(ts / 1000)
    return int(ts)


# ---------- Minimal WebSocket JSON-RPC client ----------
class DeribitWS:
    def __init__(self, uri: str = DERIBIT_WS, timeout: float = WS_RPC_TIMEOUT):
        self.uri = uri
        self.timeout = timeout
        self._ws = None
        self._id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self.last_messages: List[dict] = []
        self._reader_task_obj: Optional[asyncio.Task] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

    async def connect(self):
        try:
            self._ws = await websockets.connect(self.uri, ping_interval=20)
        except Exception as e:
            logger.exception("Failed to connect to Deribit WS: %s", e)
            raise RuntimeError("Cannot connect to Deribit WebSocket") from e
        self._reader_task_obj = asyncio.create_task(self._reader_task())
        logger.info("Connected to Deribit WS")

    async def disconnect(self):
        if self._reader_task_obj:
            self._reader_task_obj.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("WS disconnected")

    async def _reader_task(self):
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                self.last_messages.append(msg)
                # match id -> future
                if isinstance(msg, dict) and "id" in msg:
                    rid = msg["id"]
                    fut = self._pending.pop(rid, None)
                    if fut and not fut.done():
                        fut.set_result(msg)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("WS reader error: %s", e)
            raise

    async def _rpc(self, method: str, params: dict | None = None, wait: Optional[float] = None) -> dict:
        if self._ws is None:
            raise RuntimeError("WS not connected")
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        fut = asyncio.get_event_loop().create_future()
        self._pending[self._id] = fut
        await self._ws.send(json.dumps(payload))
        wait = wait or self.timeout
        try:
            resp = await asyncio.wait_for(fut, timeout=wait)
        except asyncio.TimeoutError:
            self._pending.pop(self._id, None)
            raise RuntimeError(f"Timeout waiting for response to {method}")
        if "error" in resp:
            raise RuntimeError(f"Error reply for {method}: {resp['error']}")
        return resp.get("result", resp)

    async def subscribe(self, channel: str):
        await self._rpc("public/subscribe", {"channels": [channel]})
        logger.info("Subscribed to channel %s", channel)

    async def get_index(self, currency: str) -> dict:
        return await self._rpc("public/get_index", {"currency": currency})

    async def get_instruments(self, currency: str) -> list:
        res = await self._rpc("public/get_instruments", {"currency": currency, "kind": "option", "expired": False})
        if isinstance(res, dict) and "instruments" in res:
            return res["instruments"]
        return res


# ---------- Expiry selection ----------
def pick_nearest_monthly_expiry(candidates: List[int], now_s: int) -> Optional[int]:
    if not candidates:
        return None
    monthly = []
    for ts in candidates:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        if dt.day >= MONTHLY_DAY_MIN or dt.day <= MONTHLY_DAY_MAX:
            monthly.append(ts)
    if not monthly:
        return None
    return min(monthly, key=lambda x: abs(x - now_s))


# ---------- Compute barrier from arrays ----------
def compute_barrier_probabilities_from_arrays(unix: np.ndarray, underlying: np.ndarray, iv: np.ndarray,
                                             strike: float, expiry_ts: int, option_type: str) -> Dict[str, Any]:
    mask = unix <= expiry_ts
    unix = unix[mask]
    underlying = underlying[mask]
    iv = iv[mask]
    if unix.size == 0:
        return {
            "strike": strike,
            "option_type": option_type,
            "expiry_ts": expiry_ts,
            "expiry_iso": iso_from_unix(expiry_ts),
            "last_unix": None,
            "last_iso": None,
            "last_underlying": None,
            "last_iv_pct": None,
            "last_barrier_prob": None,
            "n_points": 0,
        }
    iv_mask = (iv >= MIN_IV) & (iv <= MAX_IV)
    unix = unix[iv_mask]
    underlying = underlying[iv_mask]
    iv = iv[iv_mask]
    if unix.size == 0:
        return {
            "strike": strike,
            "option_type": option_type,
            "expiry_ts": expiry_ts,
            "expiry_iso": iso_from_unix(expiry_ts),
            "last_unix": None,
            "last_iso": None,
            "last_underlying": None,
            "last_iv_pct": None,
            "last_barrier_prob": None,
            "n_points": 0,
        }
    F = underlying.copy()
    T = (expiry_ts - unix) / (365.25 * 86400.0)
    T = np.clip(T, 1e-12, None)
    sigma = iv / 100.0
    if option_type.lower().startswith("c"):
        barrier = prob_hit_upper(F, strike, T, sigma, r=0.0, q=0.0)
    else:
        barrier = prob_hit_lower(F, strike, T, sigma, r=0.0, q=0.0)
    idx = -1
    return {
        "strike": strike,
        "option_type": option_type,
        "expiry_ts": int(expiry_ts),
        "expiry_iso": iso_from_unix(int(expiry_ts)),
        "last_unix": int(unix[idx]),
        "last_iso": iso_from_unix(int(unix[idx])),
        "last_underlying": float(F[idx]),
        "last_iv_pct": float(iv[idx]),
        "last_barrier_prob": float(barrier[idx]),
        "n_points": int(len(F)),
    }


# ---------- Plotting ----------
fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size for more strikes
def update_bar_chart(asset: str, results: List[Dict], expiry_ts: int):
    logger.debug("Updating chart with results: %s", results)
    ax.clear()
    strikes = [f"{int(r['strike']):,}" for r in results]  # Comma-separated thousands
    probs = [r["last_barrier_prob"] if r["last_barrier_prob"] is not None else 0 for r in results]
    expiry_dt = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
    expiry_str = expiry_dt.strftime("%d%b").upper()  # e.g., 29AUG
    bars = ax.bar(range(len(strikes)), probs, align='center', width=0.8, alpha=0.8, color='skyblue', label='Barrier Prob')
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Barrier Probability")
    ax.set_title(f"Asset = {asset}, Expiration = {expiry_str}")
    ax.set_ylim(0, 1.1)  # Add padding above 1.0 to prevent clipping
    ax.set_xticks(range(len(strikes)))
    ax.set_xticklabels(strikes, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}' if height > 0 else '',
                ha='center', va='bottom' if height < 0.95 else 'top',  # Adjust text position for high values
                fontsize=8)
    plt.tight_layout(pad=2.0)  # Increase padding
    plt.draw()
    plt.pause(0.1)  # Brief pause to ensure display


# ---------- Live streaming ----------
async def live_stream(client: DeribitWS, asset: str, strikes: List[float], interval: int, minimal: bool = False):
    asset = asset.upper()
    # Get initial spot
    index_resp = await client.get_index(asset)
    spot = None
    if isinstance(index_resp, dict):
        if asset in index_resp:
            spot = float(index_resp[asset])
        elif "edp" in index_resp:
            spot = float(index_resp["edp"])
    if spot is None:
        raise RuntimeError(f"Could not parse spot for {asset}")

    logger.info("Initial spot for %s = %s", asset, spot)

    # Subscribe to perpetual for live spot updates
    perp_ch = f"ticker.{asset}-PERPETUAL.100ms"
    await client.subscribe(perp_ch)

    # Get instruments
    instruments = await client.get_instruments(asset)
    if not instruments:
        raise RuntimeError(f"Empty instrument list for {asset}")

    by_strike = defaultdict(list)
    now_s = now_unix()
    for inst in instruments:
        try:
            if inst.get("expired"):
                continue
            strike_val = float(inst.get("strike"))
            opt_type = inst.get("option_type")
            exp_ts = ms_to_s_if_needed(int(inst.get("expiration_timestamp", 0)))
            name = inst.get("instrument_name")
            by_strike[int(round(strike_val))].append({
                "instrument_name": name,
                "strike": strike_val,
                "option_type": opt_type,
                "expiration_ts": exp_ts,
            })
        except Exception:
            continue

    decision = {s: ("call" if s > spot else "put") for s in strikes}

    chosen = {}
    for s in strikes:
        key = int(round(s))
        cand = [i for i in by_strike.get(key, []) if i["option_type"] == decision[s]]
        if not cand:
            raise RuntimeError(f"No live instruments for strike {s} and side {decision[s]}")
        candidate_ts = [c["expiration_ts"] for c in cand]
        picked = pick_nearest_monthly_expiry(candidate_ts, now_s)
        if picked is None:
            raise RuntimeError(f"No monthly expiry found for strike {s}")
        picked_inst = next((c for c in cand if c["expiration_ts"] == picked), None)
        if not picked_inst:
            raise RuntimeError(f"Could not map expiry for strike {s}")
        chosen[s] = picked_inst

    # Subscribe to option tickers
    for s, inst in chosen.items():
        ch = f"ticker.{inst['instrument_name']}.100ms"
        await client.subscribe(ch)

    # Storage
    storage = {s: {"unix": [], "underlying": [], "iv": []} for s in strikes}

    last_output = 0
    plt.ion()  # Enable interactive mode
    plt.show(block=False)  # Initialize plot window without blocking
    while True:
        try:
            # Process recent messages
            msgs = client.last_messages
            client.last_messages = []  # Clear to avoid reprocessing
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                params = msg.get("params") or {}
                ch = params.get("channel", "")
                data = params.get("data") or {}
                ts_ms = data.get("timestamp")
                ts = ms_to_s_if_needed(int(ts_ms)) if ts_ms else now_unix()

                # Update spot from perpetual
                if ch == perp_ch:
                    new_spot = data.get("index_price") or data.get("estimated_delivery_price")
                    if new_spot:
                        spot = float(new_spot)

                # Update options
                for s, inst in chosen.items():
                    name = inst["instrument_name"]
                    if ch == f"ticker.{name}.100ms":
                        underlying_val = data.get("underlying_price")
                        iv_val = data.get("mark_iv")
                        if underlying_val and iv_val:
                            storage[s]["unix"].append(int(ts))
                            storage[s]["underlying"].append(float(underlying_val))
                            storage[s]["iv"].append(float(iv_val))
                            # Trim to max points
                            for key in storage[s]:
                                storage[s][key] = storage[s][key][-MAX_POINTS:]

            # Periodic output
            current_time = time.time()
            if current_time - last_output >= interval:
                results = []
                for s in strikes:
                    inst = chosen[s]
                    expiry_ts = inst["expiration_ts"]
                    opt_type = inst["option_type"]
                    arr_unix = np.array(storage[s]["unix"], dtype=float)
                    arr_underlying = np.array(storage[s]["underlying"], dtype=float)
                    arr_iv = np.array(storage[s]["iv"], dtype=float)
                    rec = compute_barrier_probabilities_from_arrays(arr_unix, arr_underlying, arr_iv, float(s), int(expiry_ts), opt_type)
                    results.append(rec)
                if minimal:
                    minimal_output = {
                        "asset": asset,
                        "timestamp": now_unix(),
                        "probabilities": [
                            {"strike": r["strike"], "barrier_prob": r["last_barrier_prob"]} for r in results
                        ],
                    }
                    print(json.dumps(minimal_output), flush=True)
                else:
                    output = {
                        "asset": asset,
                        "spot": spot,
                        "timestamp": now_unix(),
                        "results": results,
                    }
                    print(json.dumps(output), flush=True)
                # Update bar chart
                if results and any(r["last_barrier_prob"] is not None for r in results):
                    update_bar_chart(asset, results, expiry_ts)
                last_output = current_time

            await asyncio.sleep(0.1)  # Small sleep to yield
        except Exception as e:
            logger.error("Error in live stream: %s", e)
            await client.disconnect()
            await asyncio.sleep(RECONNECT_DELAY)
            await client.connect()
            await client.subscribe(perp_ch)
            for s, inst in chosen.items():
                ch = f"ticker.{inst['instrument_name']}.100ms"
                await client.subscribe(ch)


# ---------- CLI & entry ----------
def parse_strikes_list(arr: List[str]) -> List[float]:
    out = []
    for s in arr:
        try:
            out.append(float(s))
        except Exception:
            logger.warning("Skipping invalid strike: %s", s)
    return out


def main():
    parser = argparse.ArgumentParser(description="Deribit live barrier pricer.")
    parser.add_argument("--asset", required=True, help="Asset, e.g. BTC")
    parser.add_argument("--strike", nargs="+", required=True, help="Strikes e.g. 110000 120000")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Update interval seconds")
    parser.add_argument("--minimal", action="store_true", help="Minimal output mode with asset, strike, prob")
    args = parser.parse_args()

    strikes = parse_strikes_list(args.strike)

    try:
        async def _run():
            async with DeribitWS() as client:
                await live_stream(client, args.asset, strikes, args.interval, args.minimal)
        asyncio.run(_run())
    except Exception as e:
        logger.exception("Failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
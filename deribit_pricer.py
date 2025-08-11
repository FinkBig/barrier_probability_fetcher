#!/usr/bin/env python3
"""
deribit_pricer.py

WebSocket-only Deribit barrier probability pricer (live streaming).
- Continuous updates every interval seconds.
- Outputs clean formatted table to terminal or JSON lines to stdout with --minimal.
- Displays live bar chart of barrier probabilities.
- Stores all historical data for comparison with Polymarket.
- Appends data to a single CSV log file per asset in data_logs/<asset> folder.
- Uses current time and spot price for real-time calculations.
- Configurable via YAML file.
"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import websockets
from scipy.stats import norm
import matplotlib.pyplot as plt
import yaml
import calendar

# ---------- Config Loading ----------
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

CONFIG = load_config()
DERIBIT_WS = CONFIG.get("websocket_uri", "wss://www.deribit.com/ws/api/v2")
WS_RPC_TIMEOUT = CONFIG.get("rpc_timeout", 8.0)
MIN_IV = CONFIG.get("min_iv", 0.5)
MAX_IV = CONFIG.get("max_iv", 1000.0)
MONTHLY_DAY_MIN = CONFIG.get("monthly_day_min", 24)
MONTHLY_DAY_MAX = CONFIG.get("monthly_day_max", 3)
DEFAULT_INTERVAL = CONFIG.get("interval", 5)
RECONNECT_DELAYS = CONFIG.get("reconnect_delays", [5, 10, 20, 40])
IV_TIMEOUT = CONFIG.get("iv_timeout", 30)  # seconds to wait for IV data
SPOT_TIMEOUT = CONFIG.get("spot_timeout", 60)  # seconds to wait for spot update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deribit_pricer")

# Set matplotlib backend for macOS compatibility
plt.switch_backend('TkAgg')

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

# ---------- CSV Logging ----------
class CSVLogger:
    def __init__(self, asset: str, data_logs_dir: str = "data_logs"):
        self.asset = asset.upper()
        self.data_logs_dir = os.path.join(data_logs_dir, self.asset)
        self.csv_file = None
        self.csv_writer = None
        self._setup_csv_file()
    
    def _setup_csv_file(self):
        os.makedirs(self.data_logs_dir, exist_ok=True)
        filename = f"{self.asset}_probabilities.csv"
        filepath = os.path.join(self.data_logs_dir, filename)
        file_exists = os.path.exists(filepath)
        self.csv_file = open(filepath, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            header = [
                'timestamp', 'timestamp_iso', 'asset', 'spot', 'strike', 'option_type',
                'expiry_ts', 'expiry_iso', 'underlying_price', 'iv_pct', 'barrier_prob', 'n_points'
            ]
            self.csv_writer.writerow(header)
        logger.info(f"CSV logging to: {filepath} (append mode)")
    
    def log_data(self, output_data: Dict[str, Any]):
        if not self.csv_writer:
            return
        timestamp = output_data.get('timestamp', now_unix())
        timestamp_iso = iso_from_unix(timestamp)
        asset = output_data.get('asset', self.asset)
        spot = output_data.get('spot', 0.0)
        for result in output_data.get('results', []):
            row = [
                timestamp,
                timestamp_iso,
                asset,
                spot,
                result.get('strike', 0.0),
                result.get('option_type', ''),
                result.get('expiry_ts', 0),
                result.get('expiry_iso', ''),
                result.get('last_underlying', 0.0),
                result.get('last_iv_pct', 0.0),
                result.get('last_barrier_prob', 0.0),
                result.get('n_points', 0)
            ]
            self.csv_writer.writerow(row)
        self.csv_file.flush()
    
    def close(self):
        if self.csv_file:
            self.csv_file.close()
            logger.info(f"CSV logging stopped for {self.asset}")

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
        return res

# ---------- Expiry selection ----------
def pick_nearest_monthly_expiry(candidates: List[int], now_s: int) -> Optional[int]:
    if not candidates:
        logger.warning("No expiry candidates provided")
        return None
    
    now_dt = datetime.fromtimestamp(now_s, tz=timezone.utc)
    current_year = now_dt.year
    current_month = now_dt.month
    target_month = current_month if now_dt.day <= 15 else current_month + 1
    if target_month > 12:
        target_month = 1
        current_year += 1
    
    monthly = []
    for ts in candidates:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        days_to_expiry = (ts - now_s) / 86400.0
        if (dt.year == current_year and dt.month == target_month and dt.day >= MONTHLY_DAY_MIN) or \
           (dt.year == current_year + (1 if target_month == 1 else 0) and dt.month == target_month and dt.day >= MONTHLY_DAY_MIN):
            if 10 <= days_to_expiry <= 40:
                monthly.append(ts)
                logger.debug("Expiry %s (%.2f days) included in monthly filter", iso_from_unix(ts), days_to_expiry)
    
    logger.info("Available expiry timestamps: %s", [iso_from_unix(ts) for ts in candidates])
    logger.info("Filtered monthly expiries: %s", [iso_from_unix(ts) for ts in monthly])
    
    target_dt = datetime(current_year, target_month, min(29, calendar.monthrange(current_year, target_month)[1]), tzinfo=timezone.utc)
    
    if monthly:
        selected = min(monthly, key=lambda x: abs(x - int(target_dt.timestamp())))
        logger.info("Selected expiry: %s", iso_from_unix(selected))
        return selected
    logger.warning("No end-of-month expiry found, falling back to nearest expiry")
    selected = min(candidates, key=lambda x: abs(x - now_s)) if candidates else None
    if selected:
        logger.info("Fallback expiry: %s", iso_from_unix(selected))
    return selected

# ---------- Plotting ----------
fig, ax = plt.subplots(figsize=(12, 6))
def update_bar_chart(asset: str, results: List[Dict], expiry_ts: int):
    logger.debug("Updating chart with results: %s", results)
    ax.clear()
    strikes = [f"{int(r['strike']):,}" for r in results]
    probs = [r["last_barrier_prob"] if r["last_barrier_prob"] is not None else 0 for r in results]
    expiry_dt = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
    expiry_str = expiry_dt.strftime("%d%b").upper()
    bars = ax.bar(range(len(strikes)), probs, align='center', width=0.8, alpha=0.8, color='skyblue', label='Barrier Prob')
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Probability of Hitting Barrier")
    ax.set_title(f"Live Barrier Probabilities for {asset} (Expiry: {expiry_str})")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(strikes)))
    ax.set_xticklabels(strikes, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}' if height > 0 else '',
                ha='center', va='bottom' if height < 0.95 else 'top',
                fontsize=8)
    plt.tight_layout(pad=2.0)
    plt.draw()
    plt.pause(0.01)

# ---------- Formatted Terminal Output ----------
def print_formatted_output(output: Dict[str, Any]):
    """Prints a clean, formatted table to the terminal without clearing."""
    spot = output.get('spot')
    asset = output.get('asset')
    expiry_iso = output['results'][0]['expiry_iso'] if output['results'] else 'N/A'
   
    print("\n" + "=" * 70)
    print(f"DERIBIT BARRIER PROBABILITY PRICER".center(70))
    print("-" * 70)
    print(f"Asset: {asset:<15} Spot Price: ${spot:,.2f}")
    print(f"Expiry: {expiry_iso:<30}")
    print("-" * 70)
    print(f"{'Strike':>12} | {'Type':>6} | {'Mark IV (%)':>12} | {'Barrier Prob.':>15}")
    print("-" * 70)
    for r in output.get('results', []):
        strike_str = f"${r['strike']:,.0f}"
        type_str = r['option_type'].capitalize()
        iv_str = f"{r['last_iv_pct']:.2f}%" if r['last_iv_pct'] is not None else "N/A"
        prob_str = f"{r['last_barrier_prob']:.2%}" if r['last_barrier_prob'] is not None else "Calculating..."
        print(f"{strike_str:>12} | {type_str:>6} | {iv_str:>12} | {prob_str:>15}")
    print("-" * 70)
    print(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ---------- Live streaming ----------
async def live_stream(client: DeribitWS, asset: str, strikes: List[float], interval: int, minimal: bool = False, callback: Optional[callable] = None):
    asset = asset.upper()
    csv_logger = CSVLogger(asset)
    start_time = now_unix()
    last_spot_update = start_time  # Track last spot update time
    last_spot_value = None  # Track last logged spot price for deduplication
    pending_spot = None  # Track pending spot price change in current cycle
    
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
    last_spot_value = spot
    pending_spot = spot

    perp_ch = f"ticker.{asset}-PERPETUAL.100ms"
    await client.subscribe(perp_ch)

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
    all_expiries = []
    for s in strikes:
        key = int(round(s))
        cand = [i for i in by_strike.get(key, []) if i["option_type"] == decision[s]]
        if not cand:
            logger.warning(f"No live instruments for strike {s} and side {decision[s]}")
            continue
        candidate_ts = [c["expiration_ts"] for c in cand]
        all_expiries.extend(candidate_ts)
    
    logger.info("All available expiries: %s", [iso_from_unix(ts) for ts in all_expiries])
    
    picked_expiry = pick_nearest_monthly_expiry(all_expiries, now_s)
    if picked_expiry is None:
        raise RuntimeError("No monthly expiry found")

    chosen = {}
    missing_strikes = []
    for s in strikes:
        key = int(round(s))
        cand = [i for i in by_strike.get(key, []) if i["option_type"] == decision[s] and i["expiration_ts"] == picked_expiry]
        if not cand:
            missing_strikes.append(s)
            continue
        picked_inst = cand[0]
        chosen[s] = picked_inst
        await client.subscribe(f"ticker.{picked_inst['instrument_name']}.100ms")
    
    if missing_strikes:
        logger.warning("No instruments found for strikes %s with expiry %s", missing_strikes, iso_from_unix(picked_expiry))
        if not chosen:
            raise RuntimeError(f"No instruments available for any strikes with expiry {iso_from_unix(picked_expiry)}")

    storage = {s: {"unix": [], "underlying": [], "iv": []} for s in chosen}
    last_output = 0
    plt.ion()
    plt.show(block=False)
    reconnect_attempt = 0

    try:
        while True:
            try:
                # Check for stale spot price
                now_ts = now_unix()
                if now_ts - last_spot_update > SPOT_TIMEOUT:
                    logger.warning("No spot price update for %s seconds, polling API", SPOT_TIMEOUT)
                    index_resp = await client.get_index(asset)
                    new_spot = None
                    if isinstance(index_resp, dict):
                        if asset in index_resp:
                            new_spot = float(index_resp[asset])
                        elif "edp" in index_resp:
                            new_spot = float(index_resp["edp"])
                    if new_spot and new_spot != last_spot_value:
                        spot = new_spot
                        last_spot_update = now_ts
                        logger.info("Updated spot via API poll: %s", spot)
                        last_spot_value = spot
                        pending_spot = spot

                # Process recent messages
                msgs = client.last_messages
                client.last_messages = []
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    params = msg.get("params") or {}
                    ch = params.get("channel", "")
                    data = params.get("data") or {}
                    ts_ms = data.get("timestamp")
                    ts = ms_to_s_if_needed(int(ts_ms)) if ts_ms else now_unix()
                    logger.debug("Received message for channel %s: %s", ch, data)

                    if ch == perp_ch:
                        new_spot = data.get("index_price") or data.get("estimated_delivery_price")
                        if new_spot:
                            new_spot = float(new_spot)
                            if new_spot != pending_spot:
                                pending_spot = new_spot
                                last_spot_update = ts

                    for s, inst in chosen.items():
                        name = inst["instrument_name"]
                        if ch == f"ticker.{name}.100ms":
                            underlying_val = data.get("underlying_price")
                            iv_val = data.get("mark_iv")
                            if underlying_val and iv_val:
                                storage[s]["unix"].append(int(ts))
                                storage[s]["underlying"].append(float(underlying_val))
                                storage[s]["iv"].append(float(iv_val))

                current_time = time.time()
                if current_time - last_output >= interval:
                    # Log pending spot update if it differs from last logged value
                    if pending_spot is not None and pending_spot != last_spot_value:
                        spot = pending_spot
                        logger.info("Updated spot via WebSocket: %s", spot)
                        last_spot_value = spot
                    pending_spot = None  # Reset for next cycle

                    now_ts = now_unix()
                    results = []
                    for s in chosen:
                        inst = chosen[s]
                        expiry_ts = inst["expiration_ts"]
                        opt_type = inst["option_type"]
                        arr_iv = np.array(storage[s]["iv"], dtype=float)
                        arr_underlying = np.array([spot], dtype=float)
                        T = np.array([(expiry_ts - now_ts) / (365.25 * 86400.0)], dtype=float)
                        T = np.clip(T, 1e-12, None)
                        if arr_iv.size == 0 or (now_ts - start_time > IV_TIMEOUT and not np.all((arr_iv >= MIN_IV) & (arr_iv <= MAX_IV))):
                            logger.warning(f"No valid IV data for strike {s} after {IV_TIMEOUT} seconds")
                            rec = {
                                "strike": s,
                                "option_type": opt_type,
                                "expiry_ts": expiry_ts,
                                "expiry_iso": iso_from_unix(expiry_ts),
                                "last_unix": None,
                                "last_iso": None,
                                "last_underlying": None,
                                "last_iv_pct": None,
                                "last_barrier_prob": None,
                                "n_points": 0,
                            }
                        else:
                            sigma = np.array([arr_iv[-1] / 100.0], dtype=float)
                            barrier = prob_hit_upper(arr_underlying, s, T, sigma) if opt_type.lower().startswith("c") else prob_hit_lower(arr_underlying, s, T, sigma)
                            rec = {
                                "strike": s,
                                "option_type": opt_type,
                                "expiry_ts": int(expiry_ts),
                                "expiry_iso": iso_from_unix(int(expiry_ts)),
                                "last_unix": now_ts,
                                "last_iso": iso_from_unix(now_ts),
                                "last_underlying": float(arr_underlying[-1]),
                                "last_iv_pct": float(arr_iv[-1]),
                                "last_barrier_prob": float(barrier[-1]),
                                "n_points": 1,
                            }
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
                        print_formatted_output(output)
                        csv_logger.log_data(output)
                        if callback:
                            callback(output)
                    if results and any(r["last_barrier_prob"] is not None for r in results):
                        update_bar_chart(asset, results, picked_expiry)
                    last_output = current_time
                    reconnect_attempt = 0

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error("Error in live stream: %s", e)
                await client.disconnect()
                delay = RECONNECT_DELAYS[min(reconnect_attempt, len(RECONNECT_DELAYS) - 1)]
                await asyncio.sleep(delay)
                reconnect_attempt += 1
                await client.connect()
                await client.subscribe(perp_ch)
                for s, inst in chosen.items():
                    ch = f"ticker.{inst['instrument_name']}.100ms"
                    await client.subscribe(ch)
    finally:
        csv_logger.close()
        plt.ioff()
        plt.show()

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
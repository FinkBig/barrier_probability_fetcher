# thesis_project_simple/scripts/01_fetch_deribit.py

import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import logging
import math
import pathlib
import sys

# Assume config.py is in the parent directory of 'scripts/'
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent # Go up one level to thesis_project_simple
sys.path.append(str(PROJECT_ROOT)) # Add project root to sys.path

try:
    import config # Import config variables
except ModuleNotFoundError:
    logging.error("ERROR: config.py not found. Make sure it's in the project root directory.")
    sys.exit(1)

# --- Configuration from config.py ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DERIBIT_HISTORY_URL = config.DERIBIT_HISTORY_URL
TARGET_EXPIRY_DT = config.TARGET_EXPIRY_DT
START_DT = config.DERIBIT_FETCH_START_DT # Use specific fetch start date
END_DT = config.TARGET_EXPIRY_DT # Fetch up to expiry
CURRENCY = config.DERIBIT_CURRENCY
REQUEST_DELAY_S = config.REQUEST_DELAY_S
OUTPUT_DIR = config.INTERMEDIATE_DATA_DIR
OUTPUT_FILENAME = config.DERIBIT_RAW_AGG_FILENAME
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# Hardcoded here, could move to config
TRADES_COUNT_LIMIT = 10000
API_RETRY_ATTEMPTS = 3
API_RETRY_WAIT_S = 3
# --- End Configuration ---

# Ensure output directory exists
config.ensure_dir(OUTPUT_FILE_PATH)

# --- Helper Classes/Functions --- (Same as before)
class DeribitApiError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def make_api_request_with_retry(url, params, retries=API_RETRY_ATTEMPTS, wait=API_RETRY_WAIT_S):
    for attempt in range(retries):
        try:
            logging.debug(f"Attempt {attempt+1}: Requesting {url} with params: {params}")
            response = requests.get(url, params=params, timeout=20)
            try:
                data = response.json()
                if 'error' in data:
                    error_message = data['error'].get('message', 'Unknown error')
                    error_code = data['error'].get('code', 'N/A')
                    raise DeribitApiError(f"Deribit API Error (Code: {error_code}): {error_message}", code=error_code)
            except ValueError:
                 response.raise_for_status()
                 raise requests.exceptions.RequestException(f"Invalid JSON received: {response.text[:200]}")
            response.raise_for_status()
            return data
        except DeribitApiError as e:
             logging.error(f"Deribit API Error encountered: {e}. Not retrying.")
             raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
            logging.warning(f"Request failed (Attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s...")
            if attempt + 1 == retries:
                logging.error("Max retries reached. Request failed.")
                raise
            time.sleep(wait)
    return None

def get_historical_option_instruments(currency, target_expiry_ts_ms):
    endpoint = f"{DERIBIT_HISTORY_URL}/public/get_instruments"
    params = {'currency': currency, 'kind': 'option', 'expired': 'true'}
    logging.info(f"Fetching expired {currency} option instruments...")
    logging.info(f"Filtering for expiry timestamp: {target_expiry_ts_ms}")
    try:
        response = requests.get(endpoint, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        if 'result' not in data or not isinstance(data['result'], list):
            logging.error("Unexpected response format from get_instruments.")
            return []
        matching_instruments = [
            inst['instrument_name'] for inst in data['result']
            if isinstance(inst, dict) and inst.get('expiration_timestamp') == target_expiry_ts_ms and 'instrument_name' in inst
        ]
        logging.info(f"Found {len(matching_instruments)} instruments expiring at target.")
        return matching_instruments
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching instruments: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error processing instruments: {e}")
        return []

def get_historical_trades(instrument_name, start_ts, end_ts):
    endpoint = f"{DERIBIT_HISTORY_URL}/public/get_last_trades_by_instrument_and_time"
    all_trades = []
    current_start_ts = start_ts
    logging.info(f"Fetching trades for {instrument_name} from {datetime.fromtimestamp(start_ts/1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_ts/1000, tz=timezone.utc)}")
    while current_start_ts < end_ts:
        params = {'instrument_name': instrument_name, 'start_timestamp': current_start_ts, 'end_timestamp': end_ts, 'count': TRADES_COUNT_LIMIT, 'include_old': 'true'}
        logging.debug(f"Fetching trades batch starting from {current_start_ts}...")
        try:
            time.sleep(REQUEST_DELAY_S)
            data = make_api_request_with_retry(endpoint, params)
            if data is None: break
            if 'result' not in data or 'trades' not in data['result']: break
            trades = data['result']['trades']
            has_more = data['result'].get('has_more', False)
            if not trades: break
            all_trades.extend(trades)
            last_trade_ts_in_batch = trades[-1]['timestamp']
            logging.debug(f"Fetched {len(trades)} trades. Last ts: {last_trade_ts_in_batch}")
            if not has_more: break
            next_start_ts = last_trade_ts_in_batch + 1
            if next_start_ts == current_start_ts: break
            current_start_ts = next_start_ts
        except DeribitApiError as e: logging.error(f"Deribit API error for {instrument_name}: {e}. Stopping."); break
        except requests.exceptions.RequestException as e: logging.error(f"Request failed for {instrument_name} after retries: {e}. Stopping."); break
        except Exception as e: logging.error(f"Unexpected error fetching trades for {instrument_name}: {e}"); break
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df.drop_duplicates(subset=['trade_id'], keep='first')
        trades_df = trades_df[(trades_df['timestamp'] >= start_ts) & (trades_df['timestamp'] <= end_ts)]
        trades_df = trades_df.sort_values(by='timestamp')
        logging.info(f"Found {len(trades_df)} unique trades for {instrument_name} within time range.")
        return trades_df.to_dict('records')
    else:
        logging.info(f"No trades found for {instrument_name} in period.")
        return []

def aggregate_trades_to_hourly_klines(trades, instrument_name):
    if not trades: return pd.DataFrame()
    logging.debug(f"Aggregating {len(trades)} trades for {instrument_name}...")
    try:
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        aggregation_rules = {'price': ['first', 'max', 'min', 'last'], 'amount': 'sum'}
        hourly_klines = df.resample('h', label='left').agg(aggregation_rules)
        hourly_klines.columns = ['_'.join(col).strip() for col in hourly_klines.columns.values]
        hourly_klines = hourly_klines.rename(columns={'price_first': 'open', 'price_max': 'high', 'price_min': 'low', 'price_last': 'close', 'amount_sum': 'volume'})
        hourly_klines = hourly_klines.dropna(subset=['open'])
        hourly_klines = hourly_klines.reset_index()
        logging.info(f"Aggregated into {len(hourly_klines)} hourly kline bars for {instrument_name}.")
        return hourly_klines
    except Exception as e:
        logging.error(f"Failed to aggregate trades for {instrument_name}: {e}")
        return pd.DataFrame()

# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info(f"--- Running Script: {pathlib.Path(__file__).name} ---")
    # Check if dates were parsed correctly in config
    if TARGET_EXPIRY_DT is None or START_DT is None:
         logging.error("Start/Expiry dates not configured correctly in config.py. Exiting.")
         sys.exit(1)

    target_expiry_ts_ms = int(TARGET_EXPIRY_DT.timestamp() * 1000)
    start_ts_ms = int(START_DT.timestamp() * 1000)
    end_ts_ms = int(END_DT.timestamp() * 1000) # Use END_DT which is expiry time

    target_instrument_names = get_historical_option_instruments(CURRENCY, target_expiry_ts_ms)
    if not target_instrument_names:
        logging.warning(f"Could not find any instruments for expiry {TARGET_EXPIRY_DT}. Exiting.")
    else:
        logging.info(f"Found {len(target_instrument_names)} instruments. Fetching trades...")
        all_hourly_data = []
        processed_count = 0
        for name in target_instrument_names:
            logging.info(f"--- Processing instrument: {name} ({processed_count+1}/{len(target_instrument_names)}) ---")
            trades = get_historical_trades(name, start_ts_ms, end_ts_ms)
            if trades:
                hourly_klines_df = aggregate_trades_to_hourly_klines(trades, name)
                if not hourly_klines_df.empty:
                    hourly_klines_df['instrument'] = name
                    hourly_klines_df = hourly_klines_df[['instrument', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    all_hourly_data.append(hourly_klines_df)
            processed_count += 1
            if processed_count % 20 == 0: logging.info(f"--- Progress: Finished {processed_count}/{len(target_instrument_names)} ---")

        if all_hourly_data:
            final_df = pd.concat(all_hourly_data, ignore_index=True)
            final_df = final_df.sort_values(by=['instrument', 'timestamp']).reset_index(drop=True)
            logging.info(f"\n--- Aggregation Complete. Total hourly klines: {len(final_df)} ---")
            try:
                final_df.to_csv(OUTPUT_FILE_PATH, index=False, float_format="%.8f")
                logging.info(f"Aggregated data successfully saved to: {OUTPUT_FILE_PATH}")
            except Exception as e:
                 logging.error(f"Failed to save data to CSV '{OUTPUT_FILE_PATH}': {e}")
        else:
            logging.warning("\nNo historical trade data found/aggregated.")

    logging.info(f"--- Script Finished: {pathlib.Path(__file__).name} ---")


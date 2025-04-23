# thesis_project_simple/scripts/02_fetch_binance.py

import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
import logging
import pathlib
import math
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

BINANCE_API_URL = config.BINANCE_API_URL
KLINE_ENDPOINT = "/api/v3/klines" # Standard endpoint for klines
SYMBOL = config.BINANCE_SYMBOL
INTERVAL = config.BINANCE_INTERVAL
START_DT = config.ANALYSIS_START_DT # Use the main analysis start date
END_DT = config.SPOT_FETCH_END_DT # Fetch spot slightly longer than options
REQUEST_DELAY_S = config.REQUEST_DELAY_S
OUTPUT_DIR = config.INTERMEDIATE_DATA_DIR
OUTPUT_FILENAME = config.BINANCE_SPOT_FILENAME
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# Hardcoded here, could move to config
LIMIT_PER_REQUEST = 1000
# --- End Configuration ---

# Ensure output directory exists
# Check if config has the helper function, otherwise define locally
if hasattr(config, 'ensure_dir'):
    config.ensure_dir(OUTPUT_FILE_PATH)
else:
    # Define locally if not in config
    def ensure_dir(file_path):
        directory = pathlib.Path(file_path).parent
        directory.mkdir(parents=True, exist_ok=True)
    ensure_dir(OUTPUT_FILE_PATH)


# --- Helper Functions ---
def timestamp_to_milliseconds(dt_obj):
    """Converts a datetime object to milliseconds timestamp."""
    return int(dt_obj.timestamp() * 1000)

def fetch_binance_klines(symbol, interval, start_time_ms, end_time_ms, limit):
    """Fetches klines from Binance API for a specific time window."""
    url = f"{BINANCE_API_URL}{KLINE_ENDPOINT}"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time_ms,
        'endTime': end_time_ms,
        'limit': limit
    }
    logging.debug(f"Requesting URL: {url} with params: {params}")
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            # Handle potential API errors returned in JSON format
            if isinstance(data, dict) and 'code' in data:
                 logging.error(f"Binance API Error: Code={data.get('code')}, Msg={data.get('msg')}")
            else:
                 logging.error(f"Unexpected response format from Binance API: {data}")
            return None
    except requests.exceptions.Timeout:
        logging.error("Request timed out while fetching data from Binance.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from Binance: {e}")
        if e.response is not None:
            logging.error(f"Response status: {e.response.status_code}, Text: {e.response.text[:200]}...")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info(f"--- Running Script: {pathlib.Path(__file__).name} ---")
    # Check if dates were parsed correctly in config
    if START_DT is None or END_DT is None:
         logging.error("Start/End dates not configured correctly in config.py. Exiting.")
         sys.exit(1)

    # Convert to milliseconds for API
    start_time_ms = timestamp_to_milliseconds(START_DT)
    # Subtract 1ms as Binance endTime is inclusive
    end_time_ms = timestamp_to_milliseconds(END_DT) - 1

    logging.info(f"Fetching {SYMBOL} {INTERVAL} data from {START_DT} to {END_DT}")

    all_klines = []
    current_start_time_ms = start_time_ms

    while current_start_time_ms <= end_time_ms:
        logging.info(f"Fetching batch starting from: {datetime.fromtimestamp(current_start_time_ms/1000, tz=timezone.utc)}")
        klines_batch = fetch_binance_klines(
            SYMBOL, INTERVAL, current_start_time_ms, end_time_ms, LIMIT_PER_REQUEST
        )
        time.sleep(REQUEST_DELAY_S) # Delay after request

        if klines_batch is None:
            logging.error("Failed to fetch a batch, stopping.")
            break
        elif not klines_batch:
            logging.info("Received empty batch, assuming no more data in range.")
            break
        else:
            all_klines.extend(klines_batch)
            # Determine next start time based on the open time of the *next* candle
            last_kline_open_time_ms = klines_batch[-1][0]
            # Calculate interval duration in ms (simplistic for '1h')
            # TODO: Make interval duration calculation more robust for different intervals if needed
            interval_duration_ms = 60 * 60 * 1000 # For '1h'
            current_start_time_ms = last_kline_open_time_ms + interval_duration_ms

            # Safety check
            if current_start_time_ms > end_time_ms + interval_duration_ms : # Allow fetching one interval past end_time_ms
                 logging.info("Next start time exceeds overall end time range.")
                 break
            # Prevent infinite loop if timestamp doesn't advance (shouldn't happen with this logic)
            if len(klines_batch) > 0 and last_kline_open_time_ms >= current_start_time_ms :
                 logging.warning("Timestamp did not advance correctly. Breaking loop.")
                 break


    if not all_klines:
        logging.warning("No kline data fetched. Exiting.")
    else:
        logging.info(f"Successfully fetched {len(all_klines)} klines in total.")
        logging.info("Processing fetched data...")
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(all_klines, columns=columns)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        df = df.rename(columns={'open_time': 'timestamp'})
        output_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']
        df_output = df[output_cols].copy()
        df_output = df_output.sort_values('timestamp').reset_index(drop=True)
        initial_rows = len(df_output)
        df_output = df_output.drop_duplicates(subset=['timestamp'], keep='first')
        final_rows = len(df_output)
        if initial_rows > final_rows:
            logging.warning(f"Removed {initial_rows - final_rows} duplicate timestamp rows.")
        logging.info(f"Processed data has {final_rows} rows.")

        try:
            df_output.to_csv(OUTPUT_FILE_PATH, index=False, float_format="%.8f")
            logging.info(f"Spot price data successfully saved to: {OUTPUT_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save data to CSV '{OUTPUT_FILE_PATH}': {e}")

    logging.info(f"--- Script Finished: {pathlib.Path(__file__).name} ---")

# thesis_project_simple/scripts/03_clean_options.py

import pandas as pd
import numpy as np
import logging
import pathlib
import re
from datetime import datetime, timezone # Import timezone
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

INPUT_DIR = config.INTERMEDIATE_DATA_DIR
INPUT_FILENAME = config.DERIBIT_RAW_AGG_FILENAME # Input is the raw aggregated file
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

OUTPUT_DIR = config.INTERMEDIATE_DATA_DIR # Save cleaned file in the same intermediate dir
OUTPUT_FILENAME = config.DERIBIT_CLEANED_FILENAME
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# Cleaning Thresholds
MIN_VOLUME_THRESHOLD = config.MIN_VOLUME_THRESHOLD
MIN_PRICE_THRESHOLD = config.MIN_PRICE_THRESHOLD
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


# --- Helper Function ---
def parse_instrument(instrument_name):
    """Parses Deribit option instrument name."""
    # Format: CURRENCY-DDMMMYY-STRIKE-TYPE (e.g., BTC-28MAR25-80000-C)
    match = re.match(r"([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-(\d+)-([PC])", instrument_name)
    if match:
        currency, expiry_str, strike, option_type = match.groups()
        try:
            # Assume standard 8 AM UTC expiry if only date is present
            # Use the TARGET_EXPIRY_DT from config for consistency if needed,
            # but parsing from string is more general if file contains multiple expiries
            # For now, parse directly from string as before.
            expiry_dt = pd.to_datetime(expiry_str, format='%d%b%y').replace(hour=8, tzinfo=timezone.utc)
            return {
                'currency': currency,
                'expiry_str': expiry_str,
                'expiry_dt': expiry_dt,
                'strike': int(strike),
                'type': 'Call' if option_type == 'C' else 'Put'
            }
        except ValueError:
             logging.warning(f"Could not parse expiry date '{expiry_str}' from instrument '{instrument_name}'")
             return None
    else:
        logging.warning(f"Could not parse instrument name: {instrument_name}")
        return None

# --- Main Cleaning Logic ---
if __name__ == "__main__":
    logging.info(f"--- Running Script: {pathlib.Path(__file__).name} ---")

    # --- Load Data ---
    try:
        df = pd.read_csv(INPUT_FILE_PATH, parse_dates=['timestamp'])
        # Ensure timestamp is UTC (it should be from aggregation, but double-check)
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        logging.info(f"Loaded raw aggregated data from '{INPUT_FILE_PATH}': {len(df)} rows")
    except FileNotFoundError:
        logging.error(f"Input data file not found: {INPUT_FILE_PATH}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading input data: {e}")
        sys.exit(1)

    initial_rows = len(df)

    # --- Step 1: Parse Instrument Details ---
    # Assumes the aggregated file only has 'instrument' column, not strike/type/expiry
    logging.info("Parsing instrument names...")
    parsed_data = df['instrument'].apply(parse_instrument)
    df = pd.concat([df, pd.json_normalize(parsed_data)], axis=1)
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df['type'] = df['type'].astype(str)
    df['expiry_dt'] = pd.to_datetime(df['expiry_dt'], errors='coerce', utc=True)

    rows_after_parsing = len(df)
    essential_cols = ['timestamp', 'instrument', 'close', 'volume', 'strike', 'type', 'expiry_dt']
    df = df.dropna(subset=essential_cols)
    rows_after_dropna_parsing = len(df)
    if rows_after_parsing > rows_after_dropna_parsing:
         logging.warning(f"Dropped {rows_after_parsing - rows_after_dropna_parsing} rows due to missing essential data after parsing.")

    # --- Step 2: Filter by Volume ---
    rows_before_vol_filter = len(df)
    df = df[df['volume'] >= MIN_VOLUME_THRESHOLD]
    rows_after_vol_filter = len(df)
    logging.info(f"Filter by Volume (>= {MIN_VOLUME_THRESHOLD}): Removed {rows_before_vol_filter - rows_after_vol_filter} rows.")

    # --- Step 3: Filter by Price ---
    rows_before_price_filter = len(df)
    df = df[df['close'] >= MIN_PRICE_THRESHOLD]
    rows_after_price_filter = len(df)
    logging.info(f"Filter by Price (close >= {MIN_PRICE_THRESHOLD}): Removed {rows_before_price_filter - rows_after_price_filter} rows.")

    # --- Final Summary ---
    final_rows = len(df)
    total_removed = initial_rows - final_rows
    logging.info(f"\n--- Cleaning Summary ---")
    logging.info(f"Initial rows: {initial_rows}")
    logging.info(f"Rows remaining after basic filtering: {final_rows}")
    logging.info(f"Total rows removed: {total_removed}")

    # --- Save Cleaned Data ---
    if final_rows > 0:
        try:
            # Select and order columns for output
            output_columns = [
                'timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume',
                'strike', 'type', 'expiry_dt' # Include parsed columns
            ]
            # Filter for columns that actually exist in the dataframe
            output_columns = [col for col in output_columns if col in df.columns]
            df_output = df[output_columns]
            df_output.to_csv(OUTPUT_FILE_PATH, index=False, float_format="%.8f")
            logging.info(f"Cleaned data successfully saved to: {OUTPUT_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save cleaned data to CSV '{OUTPUT_FILE_PATH}': {e}")
    else:
        logging.warning("No data remaining after cleaning filters were applied. No output file saved.")

    logging.info(f"--- Script Finished: {pathlib.Path(__file__).name} ---")

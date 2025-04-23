# thesis_project_simple/config.py
import pathlib
from datetime import datetime, timezone, timedelta

# --- Project Root ---
# Assumes config.py is at the root of the project directory
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

# --- Data Folders ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
FINAL_DATA_DIR = DATA_DIR / "final"
PLOT_DIR = PROJECT_ROOT / "plots" # Changed from ANALYSIS_DIR / "plots" for simplicity

# --- Input Files ---
# <<< Updated Polymarket filename >>>
POLYMARKET_RAW_FILE = RAW_DATA_DIR / "polymarket-price-data-21-03-2025-28-03-2025-1745145393712.csv"

# --- File Names for Generated Data ---
# (Used by scripts to know what to save intermediate/final data as)
DERIBIT_RAW_AGG_FILENAME = "deribit_options_raw_agg_mar28.csv" # Added date hint
BINANCE_SPOT_FILENAME = "binance_spot_btc_mar21_mar28.csv" # Added date hint
DERIBIT_CLEANED_FILENAME = "deribit_options_cleaned_mar28.csv" # Added date hint
DERIBIT_SMOOTHED_FILENAME = "deribit_options_smoothed_mar28.csv" # Added date hint
HOURLY_COMPARISON_FILENAME = "rnd_comparison_smoothed_hourly_mar28.csv" # Added date hint
HOURLY_RND_FILENAME = "deribit_rnd_smoothed_hourly_mar28.csv" # Added date hint

# --- API & Fetching Parameters ---
DERIBIT_HISTORY_URL = "https://history.deribit.com/api/v2"
BINANCE_API_URL = "https://api.binance.com"
DERIBIT_CURRENCY = 'BTC'
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1h"
REQUEST_DELAY_S = 0.25 # Delay between API requests

# --- Date Range (Set for March 28 Analysis) ---
ANALYSIS_START_DT_STR = "2025-03-21 00:00:00"
ANALYSIS_END_DT_STR = "2025-03-28 08:00:00" # Deribit expiry time
SPOT_FETCH_END_DT_STR = "2025-03-28 09:00:00" # Fetch spot one hour longer

# Convert strings to datetime objects (can be done here or in scripts)
try:
    ANALYSIS_START_DT = datetime.strptime(ANALYSIS_START_DT_STR, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    ANALYSIS_END_DT = datetime.strptime(ANALYSIS_END_DT_STR, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    SPOT_FETCH_END_DT = datetime.strptime(SPOT_FETCH_END_DT_STR, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    # Define Deribit expiry based on end date
    TARGET_EXPIRY_DT = ANALYSIS_END_DT
    # Define Deribit fetch start (7 days prior)
    DERIBIT_FETCH_START_DT = TARGET_EXPIRY_DT - timedelta(days=7)

except ValueError as e:
    print(f"ERROR in config.py: Could not parse date strings - {e}")
    # Set to None or raise error to prevent scripts running with bad config
    ANALYSIS_START_DT = None
    ANALYSIS_END_DT = None
    SPOT_FETCH_END_DT = None
    TARGET_EXPIRY_DT = None
    DERIBIT_FETCH_START_DT = None


# --- Processing Parameters ---
# Cleaning
MIN_VOLUME_THRESHOLD = 0.1
MIN_PRICE_THRESHOLD = 0.0001
# IV/SVI/RND
RISK_FREE_RATE = 0.0 # Confirmed RFR
MIN_OTM_STRIKES_REQUIRED = 5 # For RND calculation
MIN_VALID_IVS_FOR_SVI = 5 # For SVI fitting
SVI_MAX_ITERATIONS = 100
# Comparison
MAX_TIME_DIFF_SECONDS = 30 * 60 # Max diff between Deribit/PM timestamps

# --- Plotting ---
# Timestamp to generate detailed snapshot plot for in the analysis script
TIMESTAMP_TO_PLOT = "2025-03-28 07:00:00+00:00" # Example, confirm this exists


# --- Helper to ensure directories exist ---
# (Optional, scripts can create their own output dirs)
def ensure_dir(file_path):
    """Ensures the directory for a given file path exists."""
    directory = pathlib.Path(file_path).parent
    directory.mkdir(parents=True, exist_ok=True)

# Example of ensuring plot dir exists when config is loaded
# ensure_dir(PLOT_DIR / "dummy.txt") # Creates PLOT_DIR


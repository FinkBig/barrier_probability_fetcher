# thesis_project_simple/scripts/04_smooth_options.py

import pandas as pd
import numpy as np
import logging
import pathlib
import re
from datetime import datetime, timezone
from scipy.stats import norm
from scipy.optimize import brentq, minimize
import warnings
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
warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore common numerical warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore numpy/scipy deprecation warnings

# Input Files
OPTIONS_INPUT_DIR = config.INTERMEDIATE_DATA_DIR
OPTIONS_FILENAME = config.DERIBIT_CLEANED_FILENAME # Input is the cleaned file
OPTIONS_INPUT_FILE = OPTIONS_INPUT_DIR / OPTIONS_FILENAME

SPOT_INPUT_DIR = config.INTERMEDIATE_DATA_DIR
SPOT_FILENAME = config.BINANCE_SPOT_FILENAME
SPOT_INPUT_FILE = SPOT_INPUT_DIR / SPOT_FILENAME

# Output Files
OUTPUT_DIR = config.INTERMEDIATE_DATA_DIR # Save smoothed file in intermediate dir
OUTPUT_FILENAME = config.DERIBIT_SMOOTHED_FILENAME
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# Parameters
RISK_FREE_RATE = config.RISK_FREE_RATE
MIN_VALID_IVS_FOR_SVI = config.MIN_VALID_IVS_FOR_SVI
SVI_MAX_ITERATIONS = config.SVI_MAX_ITERATIONS
# --- End Configuration ---

# Ensure output directory exists
if hasattr(config, 'ensure_dir'):
    config.ensure_dir(OUTPUT_FILE_PATH)
else:
    def ensure_dir(file_path):
        directory = pathlib.Path(file_path).parent
        directory.mkdir(parents=True, exist_ok=True)
    ensure_dir(OUTPUT_FILE_PATH)


# --- Black-Scholes and IV Functions --- (Same as before)
def black_scholes(S, K, T, r, sigma, option_type='Call'):
    if sigma <= 1e-6 or T <= 1e-9:
        if option_type == 'Call': return np.maximum(0.0, S - K)
        else: return np.maximum(0.0, K - S)
    sigma = np.maximum(sigma, 1e-6); T = np.maximum(T, 1e-9)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call': price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'Put': price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else: raise ValueError("option_type must be 'Call' or 'Put'")
    return np.maximum(price, 0.0)

def implied_volatility(target_price, S, K, T, r, option_type='Call'):
    intrinsic_value = 0
    if option_type == 'Call': intrinsic_value = np.maximum(0.0, S - K * np.exp(-r * T))
    else: intrinsic_value = np.maximum(0.0, K * np.exp(-r * T) - S)
    if target_price < intrinsic_value - 1e-6: return np.nan
    if abs(target_price - intrinsic_value) < 1e-6: return 0.0001
    def objective(sigma):
        try:
             price = black_scholes(S, K, T, r, sigma, option_type)
             if np.isnan(price): return 1e12
             return price - target_price
        except Exception: return 1e12
    try: iv = brentq(objective, 1e-4, 5.0, xtol=1e-6, rtol=1e-6, maxiter=100)
    except ValueError:
        try:
            val_low = objective(1e-4); val_high = objective(5.0)
            if abs(val_low) < 1e-4: iv = 1e-4
            elif abs(val_high) < 1e-4: iv = 5.0
            elif val_low * val_high > 0 and abs(val_low) > 1e-3: iv = np.nan
            else: iv = np.nan
        except Exception: iv = np.nan
    except Exception as e: logging.error(f"Unexpected error in IV calc: {e}"); iv = np.nan
    if pd.isna(iv) or iv <= 0: return np.nan
    return iv

# --- SVI Model Functions --- (Same as before)
def svi_raw_variance(k, params):
    a, b, rho, m, sigma_tot = params
    if b < 0 or sigma_tot < 0 or abs(rho) >= 1: return np.inf
    k_m = k - m
    sqrt_term = np.sqrt(k_m**2 + sigma_tot**2)
    w = a + b * (rho * k_m + sqrt_term)
    return np.maximum(w, 1e-9)

def svi_objective_function(params, k, T, market_total_variance):
    model_total_variance = svi_raw_variance(k, params)
    if np.any(np.isinf(model_total_variance)): return 1e12
    error = np.sum((model_total_variance - market_total_variance)**2)
    return error

# --- Main Smoothing Logic ---
if __name__ == "__main__":
    logging.info(f"--- Running Script: {pathlib.Path(__file__).name} ---")

    # --- Load Cleaned Options Data ---
    try:
        df_options = pd.read_csv(OPTIONS_INPUT_FILE, parse_dates=['timestamp', 'expiry_dt'])
        df_options['timestamp'] = df_options['timestamp'].dt.tz_convert('UTC')
        df_options['expiry_dt'] = df_options['expiry_dt'].dt.tz_convert('UTC')
        logging.info(f"Loaded cleaned options data from '{OPTIONS_INPUT_FILE}': {len(df_options)} rows")
    except FileNotFoundError:
        logging.error(f"Input options data file not found: {OPTIONS_INPUT_FILE}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading cleaned options data: {e}")
        sys.exit(1)

    # --- Load Spot Price Data ---
    try:
        df_spot = pd.read_csv(SPOT_INPUT_FILE, parse_dates=['timestamp'])
        df_spot['timestamp'] = df_spot['timestamp'].dt.tz_convert('UTC')
        df_spot = df_spot.set_index('timestamp')
        logging.info(f"Loaded spot price data from '{SPOT_INPUT_FILE}': {len(df_spot)} rows")
    except FileNotFoundError:
        logging.error(f"Spot price data file not found: {SPOT_INPUT_FILE}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading spot price data: {e}")
        sys.exit(1)

    # Get the single expiry date
    # Ensure expiry_dt column was loaded correctly
    if 'expiry_dt' not in df_options.columns or df_options['expiry_dt'].isnull().all():
         logging.error("Expiry date information missing or invalid in options data.")
         sys.exit(1)
    target_expiry_dt = df_options['expiry_dt'].iloc[0]
    logging.info(f"Target Expiry Date: {target_expiry_dt}")

    # --- Process Data Timestamp by Timestamp ---
    all_smoothed_results = []
    processed_timestamps = 0
    grouped = df_options.groupby('timestamp')
    total_timestamps = len(grouped)
    logging.info(f"Processing {total_timestamps} unique timestamps...")

    for timestamp, group_df in grouped:
        processed_timestamps += 1
        logging.info(f"--- Processing timestamp {processed_timestamps}/{total_timestamps}: {timestamp} ---")

        # Calculate Time to Expiry (T)
        time_to_expiry = (target_expiry_dt - timestamp).total_seconds() / (365.25 * 24 * 60 * 60)
        if time_to_expiry <= 1e-9:
            logging.warning(f"Skipping timestamp {timestamp}: Time to expiry too small ({time_to_expiry:.6f}).")
            continue

        # Get Underlying Price S from Spot Data
        try:
            underlying_price = df_spot.loc[timestamp, 'close']
            if pd.isna(underlying_price): raise ValueError("Spot price is NaN")
            logging.info(f"Using Underlying S = {underlying_price:.2f} from spot data")
        except KeyError: logging.warning(f"Skipping {timestamp}: No matching spot price found."); continue
        except Exception as e: logging.warning(f"Skipping {timestamp}: Error looking up spot price - {e}."); continue

        forward_price = underlying_price # Assume F=S as r=0

        # Calculate Implied Volatilities
        iv_list = [implied_volatility(r['close'], underlying_price, r['strike'], time_to_expiry, RISK_FREE_RATE, r['type']) for _, r in group_df.iterrows()]
        group_df['iv'] = iv_list
        valid_iv_df = group_df.dropna(subset=['iv'])
        valid_iv_df = valid_iv_df[(valid_iv_df['iv'] > 1e-3) & (valid_iv_df['iv'] < 5.0)]
        num_valid_ivs = len(valid_iv_df)
        logging.info(f"Calculated {num_valid_ivs} valid IVs out of {len(group_df)} prices.")

        if num_valid_ivs < MIN_VALID_IVS_FOR_SVI:
             logging.warning(f"Skipping {timestamp}: Insufficient valid IVs ({num_valid_ivs}) for SVI fitting.")
             continue

        # Prepare Data for SVI Fitting
        valid_iv_df['log_moneyness'] = np.log(valid_iv_df['strike'] / forward_price)
        valid_iv_df['total_variance'] = valid_iv_df['iv']**2 * time_to_expiry
        k_fit = valid_iv_df['log_moneyness'].values
        w_fit = valid_iv_df['total_variance'].values

        # Fit SVI Model
        logging.info("Fitting SVI model...")
        atm_variance_approx = np.median(w_fit)
        initial_guess = [atm_variance_approx * 0.9, 0.1, -0.5, 0.0, 0.1]
        bounds = [(1e-6, 1.0), (1e-6, 1.0), (-0.999, 0.999), (-1.0, 1.0), (1e-6, 1.0)]
        try:
            result = minimize(svi_objective_function, initial_guess, args=(k_fit, time_to_expiry, w_fit), method='L-BFGS-B', bounds=bounds, options={'maxiter': SVI_MAX_ITERATIONS, 'disp': False})
            if result.success:
                fitted_params = result.x
                logging.info(f"SVI Fit OK. Params (a,b,rho,m,sigma): {np.round(fitted_params, 4)}")
            else: logging.warning(f"SVI Fit Fail: {result.message}"); continue
        except Exception as e: logging.error(f"SVI Opt Error: {e}"); continue

        # Generate Smoothed Prices
        all_strikes = np.sort(group_df['strike'].unique())
        all_k = np.log(all_strikes / forward_price)
        smoothed_total_variance = svi_raw_variance(all_k, fitted_params)
        smoothed_total_variance = np.maximum(smoothed_total_variance, 1e-9)
        smoothed_iv = np.sqrt(smoothed_total_variance / time_to_expiry)

        df_smoothed_k = pd.DataFrame({'strike': all_strikes, 'log_moneyness': all_k, 'smoothed_iv': smoothed_iv})
        df_smoothed_k['smoothed_price_call'] = [black_scholes(underlying_price, k, time_to_expiry, RISK_FREE_RATE, iv, 'Call') for k, iv in zip(df_smoothed_k['strike'], df_smoothed_k['smoothed_iv'])]
        df_smoothed_k['smoothed_price_put'] = [black_scholes(underlying_price, k, time_to_expiry, RISK_FREE_RATE, iv, 'Put') for k, iv in zip(df_smoothed_k['strike'], df_smoothed_k['smoothed_iv'])]
        df_smoothed_k['timestamp'] = timestamp
        df_smoothed_k['underlying_price'] = underlying_price
        df_smoothed_k['time_to_expiry'] = time_to_expiry
        df_smoothed_k['svi_a'], df_smoothed_k['svi_b'], df_smoothed_k['svi_rho'], df_smoothed_k['svi_m'], df_smoothed_k['svi_sigma_tot'] = fitted_params

        all_smoothed_results.append(df_smoothed_k)

    # Combine and Save Results
    if all_smoothed_results:
        final_smoothed_df = pd.concat(all_smoothed_results, ignore_index=True)
        logging.info(f"\n--- Smoothing Complete ---")
        logging.info(f"Generated smoothed prices for {final_smoothed_df['timestamp'].nunique()} timestamps.")
        try:
            final_smoothed_df.to_csv(OUTPUT_FILE_PATH, index=False, float_format="%.8f")
            logging.info(f"Smoothed data successfully saved to: {OUTPUT_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save smoothed data: {e}")
    else:
        logging.warning("No smoothed data was generated.")

    logging.info(f"--- Script Finished: {pathlib.Path(__file__).name} ---")

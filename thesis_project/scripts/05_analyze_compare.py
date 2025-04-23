# thesis_project_simple/scripts/05_analyze_compare.py

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import re
import logging
import pathlib
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid # Use newer trapezoid function
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
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
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Input Files
SMOOTHED_OPTIONS_INPUT_DIR = config.INTERMEDIATE_DATA_DIR
SMOOTHED_OPTIONS_FILENAME = config.DERIBIT_SMOOTHED_FILENAME
SMOOTHED_OPTIONS_INPUT_FILE = SMOOTHED_OPTIONS_INPUT_DIR / SMOOTHED_OPTIONS_FILENAME

POLYMARKET_INPUT_DIR = config.RAW_DATA_DIR # Polymarket data is raw input
# Ensure POLYMARKET_RAW_FILE is defined in config.py
if not hasattr(config, 'POLYMARKET_RAW_FILE'):
     logging.error("POLYMARKET_RAW_FILE not defined in config.py")
     sys.exit(1)
POLYMARKET_INPUT_FILE = config.POLYMARKET_RAW_FILE # Use the full Path object from config

# Output Files
FINAL_DATA_DIR = config.FINAL_DATA_DIR
PLOT_DIR = config.PLOT_DIR
HOURLY_COMPARISON_FILENAME = config.HOURLY_COMPARISON_FILENAME
HOURLY_RND_FILENAME = config.HOURLY_RND_FILENAME
COMPARISON_OUTPUT_FILE = FINAL_DATA_DIR / HOURLY_COMPARISON_FILENAME
RND_OUTPUT_FILE = FINAL_DATA_DIR / HOURLY_RND_FILENAME

# Parameters
RISK_FREE_RATE = config.RISK_FREE_RATE
MIN_OTM_STRIKES_REQUIRED = config.MIN_OTM_STRIKES_REQUIRED
MAX_TIME_DIFF_SECONDS = config.MAX_TIME_DIFF_SECONDS
TIMESTAMP_TO_PLOT = config.TIMESTAMP_TO_PLOT # For snapshot plots
# --- End Configuration ---

# Ensure output directories exist
# Define ensure_dir locally in case it's not in config
def ensure_dir(file_path):
    directory = pathlib.Path(file_path).parent
    directory.mkdir(parents=True, exist_ok=True)

ensure_dir(COMPARISON_OUTPUT_FILE)
ensure_dir(RND_OUTPUT_FILE)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---
def parse_polymarket_buckets(pm_df):
    """Parses Polymarket column headers into price buckets."""
    buckets = {}
    ignore_cols = {'timestamp', 'pm_timestamp_actual', 'bucket_label',
                   'price_range', 'deribit_prob', 'polymarket_prob',
                   'difference', 'Date (UTC)', 'Timestamp (UTC)'}
    for col in pm_df.columns:
        if col in ignore_cols: continue
        col_lower = col.lower().replace(' ', '')
        match = re.match(r'>(\d+)k', col_lower);
        if match: buckets[col] = (int(match.group(1)) * 1000, np.inf); continue
        match = re.match(r'<(\d+)k', col_lower);
        if match: buckets[col] = (-np.inf, int(match.group(1)) * 1000); continue
        match = re.match(r'(\d+)-(\d+)k', col_lower);
        if match: buckets[col] = (int(match.group(1)) * 1000, int(match.group(2)) * 1000); continue
        logging.debug(f"Could not parse Polymarket column header into bucket: {col}")
    unique_buckets = {}
    ranges_seen = set()
    for label, range_val in buckets.items():
        if range_val not in ranges_seen: unique_buckets[label] = range_val; ranges_seen.add(range_val)
    # <<< Use the get_sort_key function defined globally below >>>
    sorted_items = sorted(unique_buckets.items(), key=lambda item: get_sort_key(item[0]))
    return dict(sorted_items)

# <<< FIX: Define get_sort_key globally BEFORE it's used >>>
def get_sort_key(label):
    """Helper function to sort bucket labels numerically."""
    label_lower = label.lower().replace(' ', '')
    if '<' in label_lower:
        match = re.match(r'<(\d+)k', label_lower)
        return -np.inf if match else 0
    if '>' in label_lower:
        match = re.match(r'>(\d+)k', label_lower)
        return np.inf if match else 0
    match = re.match(r'(\d+)-(\d+)k', label_lower)
    if match:
        return int(match.group(1)) * 1000
    return 0

def calculate_rnd_breeden_litzenberger_smoothed(options_df_ts, risk_free_rate):
    """Calculates RND using Breeden-Litzenberger on smoothed OTM prices."""
    if options_df_ts.empty: return pd.DataFrame(columns=['strike', 'rnd'])
    underlying_price = options_df_ts['underlying_price'].iloc[0]; time_to_expiry = options_df_ts['time_to_expiry'].iloc[0]; analysis_dt = options_df_ts['timestamp'].iloc[0]
    logging.debug(f"Calculating RND for {analysis_dt} S={underlying_price:.2f}, T={time_to_expiry:.6f}")
    if time_to_expiry <= 1e-9: return pd.DataFrame(columns=['strike', 'rnd'])
    forward_price = underlying_price # r=0
    otm_puts = options_df_ts[options_df_ts['strike'] < forward_price][['strike', 'smoothed_price_put']].rename(columns={'smoothed_price_put': 'price'})
    otm_calls = options_df_ts[options_df_ts['strike'] >= forward_price][['strike', 'smoothed_price_call']].rename(columns={'smoothed_price_call': 'price'})
    otm_prices_df = pd.concat([otm_puts, otm_calls]).sort_values('strike').drop_duplicates(subset=['strike']).reset_index(drop=True)
    if len(otm_prices_df) < MIN_OTM_STRIKES_REQUIRED: logging.debug(f"Insufficient OTM strikes ({len(otm_prices_df)})"); return pd.DataFrame(columns=['strike', 'rnd'])
    strikes = otm_prices_df['strike'].values; prices = otm_prices_df['price'].values; rnd_values = np.zeros_like(strikes, dtype=float)
    for i in range(1, len(otm_prices_df) - 1):
        k_prev, k_curr, k_next = strikes[i-1], strikes[i], strikes[i+1]; o_prev, o_curr, o_next = prices[i-1], prices[i], prices[i+1]
        dk_fwd = k_next - k_curr; dk_bwd = k_curr - k_prev; dk_total = k_next - k_prev
        if dk_fwd <= 0 or dk_bwd <= 0 or dk_total <= 0: continue
        dO_dK_fwd = (o_next - o_curr) / dk_fwd; dO_dK_bwd = (o_curr - o_prev) / dk_bwd
        d2O_dK2 = 2 * (dO_dK_fwd - dO_dK_bwd) / dk_total
        discount_factor = np.exp(risk_free_rate * time_to_expiry); rnd_values[i] = discount_factor * d2O_dK2
    rnd_values[0], rnd_values[-1] = 0, 0
    rnd_df = pd.DataFrame({'strike': strikes, 'raw_rnd': rnd_values}); rnd_df['rnd'] = np.maximum(rnd_df['raw_rnd'], 0)
    negative_count = (rnd_df['raw_rnd'] < 0).sum()
    if negative_count > 0: logging.debug(f"Zeroed {negative_count} negative raw RND values.")
    if len(rnd_df) > 1:
        total_probability = trapezoid(y=rnd_df['rnd'].values, x=rnd_df['strike'].values)
        logging.debug(f"Pre-norm mass: {total_probability:.4f}")
        if total_probability > 1e-8:
            rnd_df['rnd'] = rnd_df['rnd'] / total_probability; post_total_prob = trapezoid(y=rnd_df['rnd'].values, x=rnd_df['strike'].values); logging.debug(f"Post-norm mass: {post_total_prob:.4f}")
        else: rnd_df['rnd'] = 0.0
    else: rnd_df['rnd'] = 0.0
    return rnd_df[['strike', 'rnd']].reset_index(drop=True)

def integrate_rnd_for_buckets(rnd_df, price_buckets):
    """Integrates the discrete RND over specified price buckets."""
    if rnd_df.empty or len(rnd_df) < 2: return {label: 0.0 for label in price_buckets.keys()}
    probabilities = {}; strikes = rnd_df['strike'].values; rnd_values = rnd_df['rnd'].values
    sort_idx = np.argsort(strikes); strikes, rnd_values = strikes[sort_idx], rnd_values[sort_idx]
    rnd_interpolator = interp1d(strikes, rnd_values, kind='linear', bounds_error=False, fill_value=0.0)
    logging.debug("Integrating RND over Polymarket buckets...")
    for label, (k_low, k_high) in price_buckets.items():
        integration_strikes = np.unique(np.sort(np.concatenate([[k_low if np.isfinite(k_low) else strikes.min()], strikes[(strikes >= k_low) & (strikes <= k_high)], [k_high if np.isfinite(k_high) else strikes.max()]])))
        integration_strikes = integration_strikes[(integration_strikes >= k_low) & (integration_strikes <= k_high)]
        if len(integration_strikes) < 2: prob = 0.0
        else: interpolated_rnd = rnd_interpolator(integration_strikes); interpolated_rnd = np.maximum(interpolated_rnd, 0); prob = trapezoid(y=interpolated_rnd, x=integration_strikes)
        probabilities[label] = np.maximum(prob, 0)
    total_integrated_prob = sum(probabilities.values()); logging.debug(f"Total integrated probability: {total_integrated_prob:.4f}")
    if not np.isclose(total_integrated_prob, 1.0, atol=0.02): logging.warning(f"Total integrated probability ({total_integrated_prob:.4f}) differs significantly from 1.")
    return probabilities

def plot_hourly_comparison(timestamp_to_plot, comparison_df, price_buckets, output_dir):
    """Plots Deribit vs Polymarket probabilities for a specific hour."""
    ts_dt = pd.to_datetime(timestamp_to_plot).tz_convert('UTC'); data_for_plot = comparison_df[comparison_df['timestamp'] == ts_dt].copy()
    if data_for_plot.empty: logging.warning(f"No comparison data for snapshot plot at {ts_dt}"); return
    data_for_plot = data_for_plot.dropna(subset=['deribit_prob', 'polymarket_prob'])
    bucket_labels_ordered = list(price_buckets.keys()); data_for_plot['bucket_label'] = pd.Categorical(data_for_plot['bucket_label'], categories=bucket_labels_ordered, ordered=True); data_for_plot = data_for_plot.sort_values('bucket_label')
    if data_for_plot.empty: logging.warning(f"Plot data empty after filtering NaNs for {ts_dt}"); return
    labels = data_for_plot['bucket_label']; deribit_probs = data_for_plot['deribit_prob']; polymarket_probs = data_for_plot['polymarket_prob']
    x = np.arange(len(labels)); width = 0.35; fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, deribit_probs, width, label='Deribit (Smoothed RND)', color='red', alpha=0.8); rects2 = ax.bar(x + width/2, polymarket_probs, width, label='Polymarket', color='blue', alpha=0.8)
    ax.set_ylabel('Probability'); ax.set_xlabel('Price Buckets'); ax.set_title(f'Deribit vs Polymarket Implied Probabilities\n{ts_dt.strftime("%Y-%m-%d %H:%M %Z")}'); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.7); ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)); fig.tight_layout()
    plot_filename = output_dir / f"prob_comparison_{ts_dt.strftime('%Y%m%d_%H%M%S')}.png"
    try: plt.savefig(plot_filename, dpi=120); logging.info(f"Snapshot plot saved to: {plot_filename}")
    except Exception as e: logging.error(f"Failed to save snapshot plot: {e}")
    finally: plt.close(fig)

def plot_rnd_density(timestamp_to_plot, rnd_df, comparison_df, price_buckets, output_dir):
    """Plots Deribit RND density and compares with Polymarket buckets."""
    if rnd_df is None: logging.warning("RND data not loaded, skipping density plot."); return
    ts_dt = pd.to_datetime(timestamp_to_plot).tz_convert('UTC'); rnd_data_ts = rnd_df[rnd_df['timestamp'] == ts_dt].sort_values('strike'); comp_data_ts = comparison_df[comparison_df['timestamp'] == ts_dt]
    if rnd_data_ts.empty: logging.warning(f"Missing RND data for {ts_dt}, skipping density plot."); return
    strikes = rnd_data_ts['strike'].values; rnd_values = rnd_data_ts['rnd'].values
    if len(strikes) < 2: logging.warning(f"Not enough RND strikes for density plot {ts_dt}."); return
    interp_strikes = np.linspace(strikes.min(), strikes.max(), 500); rnd_interpolator = interp1d(strikes, rnd_values, kind='linear', bounds_error=False, fill_value=0.0); interp_rnd = np.maximum(rnd_interpolator(interp_strikes), 0)
    fig, ax = plt.subplots(figsize=(12, 6)); ax.plot(interp_strikes, interp_rnd, color='red', label='Deribit Smoothed RND'); ax.fill_between(interp_strikes, interp_rnd, color='red', alpha=0.2)
    if not comp_data_ts.empty:
        plot_labels_ordered = list(price_buckets.keys()); pm_plot_df = pd.DataFrame({'bucket_label': plot_labels_ordered}); pm_plot_data = pd.merge(pm_plot_df, comp_data_ts, on='bucket_label', how='left'); pm_plot_data = pm_plot_data.dropna(subset=['polymarket_prob'])
        if pm_plot_data.empty: logging.warning(f"No matching Polymarket bucket data for density plot at {ts_dt}")
        else:
            total_pm_prob = pm_plot_data['polymarket_prob'].sum(); pm_plot_data['pm_prob_norm'] = pm_plot_data['polymarket_prob'] / total_pm_prob if total_pm_prob > 1e-9 else 0
            for idx, row in pm_plot_data.iterrows():
                label = row['bucket_label'];
                if pd.isna(label) or label not in price_buckets: continue
                k_low, k_high = price_buckets[label]; plot_low = k_low if np.isfinite(k_low) else strikes.min() - 1000; plot_high = k_high if np.isfinite(k_high) else strikes.max() + 1000; width = plot_high - plot_low
                if width <= 0 or np.isinf(width): continue
                density = row['pm_prob_norm'] / width if width > 0 else 0; bar_label = 'Polymarket Implied Density' if idx == 0 else ""; ax.bar(plot_low + width/2, density, width=width, color='blue', alpha=0.5, label=bar_label)
    ax.set_xlabel('BTC Strike Price (K)'); ax.set_ylabel('Probability Density f(K)'); ax.set_title(f'Implied Probability Density Comparison\n{ts_dt.strftime("%Y-%m-%d %H:%M %Z")}'); handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles)); ax.legend(by_label.values(), by_label.keys()); ax.grid(True, linestyle='--', alpha=0.6); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k')); fig.tight_layout()
    plot_filename = output_dir / f"rnd_density_comparison_{ts_dt.strftime('%Y%m%d_%H%M%S')}.png"
    try: plt.savefig(plot_filename, dpi=150); logging.info(f"RND Density plot saved to: {plot_filename}")
    except Exception as e: logging.error(f"Failed to save density plot: {e}")
    finally: plt.close(fig)

# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info(f"--- Running Script: {pathlib.Path(__file__).name} ---")

    # --- Load Smoothed Options Data ---
    try:
        df_options_smoothed = pd.read_csv(SMOOTHED_OPTIONS_INPUT_FILE, parse_dates=['timestamp'])
        df_options_smoothed['timestamp'] = df_options_smoothed['timestamp'].dt.tz_convert('UTC')
        logging.info(f"Loaded smoothed options data: {len(df_options_smoothed)} rows from {SMOOTHED_OPTIONS_INPUT_FILE}")
    except FileNotFoundError: logging.error(f"Input file not found: {SMOOTHED_OPTIONS_INPUT_FILE}"); sys.exit(1)
    except Exception as e: logging.error(f"Error loading smoothed options data: {e}"); sys.exit(1)

    # --- Load Polymarket Data ---
    try:
        polymarket_timestamp_col = 'Timestamp (UTC)' # Assume this is correct
        df_polymarket = pd.read_csv(POLYMARKET_INPUT_FILE)
        if polymarket_timestamp_col not in df_polymarket.columns: raise KeyError(f"Column '{polymarket_timestamp_col}' not found.")
        df_polymarket[polymarket_timestamp_col] = pd.to_datetime(df_polymarket[polymarket_timestamp_col], unit='s', errors='coerce')
        df_polymarket = df_polymarket.dropna(subset=[polymarket_timestamp_col])
        df_polymarket = df_polymarket.rename(columns={polymarket_timestamp_col: 'timestamp'})
        df_polymarket['timestamp'] = df_polymarket['timestamp'].dt.tz_localize('UTC')
        logging.info(f"Loaded and parsed Polymarket data: {len(df_polymarket)} rows from {POLYMARKET_INPUT_FILE}")
        df_polymarket = df_polymarket.sort_values('timestamp').reset_index(drop=True)
    except KeyError as e: logging.error(f"Error loading Polymarket data: {e}"); sys.exit(1)
    except FileNotFoundError: logging.error(f"Polymarket data file not found: {POLYMARKET_INPUT_FILE}"); sys.exit(1)
    except Exception as e: logging.error(f"Error loading or processing Polymarket data: {e}"); sys.exit(1)

    # --- Prepare Polymarket Buckets ---
    price_buckets = parse_polymarket_buckets(df_polymarket)
    if not price_buckets: logging.error("Could not parse any price buckets."); sys.exit(1)
    logging.info(f"Parsed Polymarket Price Buckets: {price_buckets}")

    # --- Find Common Timestamps for Analysis ---
    options_grouped = df_options_smoothed.groupby('timestamp')
    available_option_timestamps = set(options_grouped.groups.keys())
    pm_lookup = df_polymarket.set_index('timestamp')
    timestamps_to_process = []
    for ts_opt in sorted(list(available_option_timestamps)):
        try:
            closest_pm_index = pm_lookup.index.get_indexer([ts_opt], method='nearest', tolerance=pd.Timedelta(seconds=MAX_TIME_DIFF_SECONDS))
            if closest_pm_index[0] != -1: timestamps_to_process.append(ts_opt)
        except Exception as e: logging.error(f"Error finding closest Polymarket timestamp for {ts_opt}: {e}")
    logging.info(f"Found {len(timestamps_to_process)} common timestamps for hourly analysis.")

    # --- Loop Through Timestamps for Analysis ---
    all_hourly_comparisons = []
    all_hourly_rnds = []
    processed_count = 0
    for analysis_dt in timestamps_to_process:
        processed_count += 1
        logging.info(f"--- Analyzing Timestamp {processed_count}/{len(timestamps_to_process)}: {analysis_dt} ---")
        options_at_timestamp = options_grouped.get_group(analysis_dt)
        closest_pm_idx_val = pm_lookup.index.get_indexer([analysis_dt], method='nearest', tolerance=pd.Timedelta(seconds=MAX_TIME_DIFF_SECONDS))
        if closest_pm_idx_val[0] == -1: logging.warning(f"Skipping {analysis_dt}: No PM data in tolerance."); continue
        pm_row = pm_lookup.iloc[closest_pm_idx_val[0]]
        pm_actual_ts = pm_row.name

        df_rnd_hour = calculate_rnd_breeden_litzenberger_smoothed(options_at_timestamp, RISK_FREE_RATE) # Renamed variable
        if not df_rnd_hour.empty:
            df_rnd_hour['timestamp'] = analysis_dt
            all_hourly_rnds.append(df_rnd_hour)
            deribit_probabilities = integrate_rnd_for_buckets(df_rnd_hour, price_buckets) # Use df_rnd_hour
            comparison_data = []
            for label, price_range_tuple in price_buckets.items():
                prob_deribit = deribit_probabilities.get(label, 0.0)
                if label in pm_row.index:
                    prob_polymarket = pm_row[label]
                    comparison_data.append({'timestamp': analysis_dt, 'pm_timestamp_actual': pm_actual_ts, 'bucket_label': label, 'price_range': price_range_tuple, 'deribit_prob': prob_deribit, 'polymarket_prob': prob_polymarket})
                else: logging.warning(f"Bucket label '{label}' not found in PM data for {pm_actual_ts}.")
            if comparison_data:
                df_comparison_hour = pd.DataFrame(comparison_data); df_comparison_hour['difference'] = df_comparison_hour['deribit_prob'] - df_comparison_hour['polymarket_prob']; all_hourly_comparisons.append(df_comparison_hour)
            else: logging.warning(f"Could not generate comparison data for {analysis_dt}.")
        else: logging.warning(f"RND calculation failed for {analysis_dt}.")

    # --- Combine and Save Aggregated Results ---
    if all_hourly_comparisons:
        final_comparison_df = pd.concat(all_hourly_comparisons, ignore_index=True)
        logging.info(f"\n--- Aggregated Hourly Comparison Complete ({final_comparison_df['timestamp'].nunique()} timestamps) ---")
        try: final_comparison_df.to_csv(COMPARISON_OUTPUT_FILE, index=False, float_format="%.6f"); logging.info(f"Aggregated comparison results saved to: {COMPARISON_OUTPUT_FILE}")
        except Exception as e: logging.error(f"Failed to save aggregated comparison results: {e}")

        # --- Perform and Plot Overall Analysis ---
        logging.info("Calculating overall differences per bucket...")
        # <<< FIX: Use get_sort_key defined globally >>>
        overall_diff_stats = final_comparison_df.groupby('bucket_label')['difference'].agg(['mean', 'std', 'min', 'max']).sort_index(key=lambda x: x.map(get_sort_key))
        print("\n--- Overall Difference Statistics (Deribit Prob - Polymarket Prob) ---")
        print(overall_diff_stats.to_string(float_format="%.4f"))

        logging.info("Analyzing time evolution...")
        # <<< FIX: Define bucket_order using overall_diff_stats >>>
        bucket_order = list(overall_diff_stats.index)
        df_pivot_deribit = final_comparison_df.pivot(index='timestamp', columns='bucket_label', values='deribit_prob').reindex(columns=bucket_order)
        df_pivot_pm = final_comparison_df.pivot(index='timestamp', columns='bucket_label', values='polymarket_prob').reindex(columns=bucket_order)
        df_pivot_diff = final_comparison_df.pivot(index='timestamp', columns='bucket_label', values='difference').reindex(columns=bucket_order)

        # Plot Time Series
        fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10), sharex=True); plot_buckets = ['82-84k', '84-86k', '86-88k']
        for i, bucket in enumerate(plot_buckets):
            if bucket in df_pivot_deribit.columns: axes1[i].plot(df_pivot_deribit.index, df_pivot_deribit[bucket], label=f'Deribit ({bucket})', color='red', alpha=0.8)
            if bucket in df_pivot_pm.columns: axes1[i].plot(df_pivot_pm.index, df_pivot_pm[bucket], label=f'Polymarket ({bucket})', color='blue', alpha=0.8)
            axes1[i].set_ylabel('Probability'); axes1[i].legend(); axes1[i].grid(True); axes1[i].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        axes1[-1].set_xlabel('Timestamp (UTC)'); axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M')); plt.xticks(rotation=45); fig1.suptitle('Probability Evolution for Key Buckets', fontsize=16); fig1.tight_layout(rect=[0, 0.03, 1, 0.97]); plot1_filename = PLOT_DIR / "prob_evolution_buckets.png"; plt.savefig(plot1_filename, dpi=150); logging.info(f"Prob evolution plot saved: {plot1_filename}"); plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for bucket in plot_buckets:
             if bucket in df_pivot_diff.columns: ax2.plot(df_pivot_diff.index, df_pivot_diff[bucket], label=f'Difference ({bucket})', alpha=0.9)
        ax2.set_ylabel('Prob Difference (Deribit - PM)'); ax2.set_xlabel('Timestamp (UTC)'); ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Zero Diff'); ax2.legend(); ax2.grid(True); ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M')); plt.xticks(rotation=45); fig2.suptitle('Difference Evolution for Key Buckets', fontsize=16); fig2.tight_layout(); plot2_filename = PLOT_DIR / "prob_difference_evolution.png"; plt.savefig(plot2_filename, dpi=150); logging.info(f"Prob difference plot saved: {plot2_filename}"); plt.close(fig2)

        # Plot Snapshots
        logging.info("Generating distribution snapshot plots...")
        # Ensure final_comparison_df is not empty before getting timestamps
        if not final_comparison_df.empty:
             snapshot_timestamps = [final_comparison_df['timestamp'].min(), final_comparison_df['timestamp'].iloc[len(final_comparison_df)//2], final_comparison_df['timestamp'].max()]
             # Parse buckets from the original Polymarket data for labels
             parsed_buckets = parse_polymarket_buckets(pd.read_csv(POLYMARKET_INPUT_FILE)) # Read PM file again just for headers/buckets
             for ts in snapshot_timestamps: plot_hourly_comparison(ts, final_comparison_df, parsed_buckets, PLOT_DIR)
        else:
             logging.warning("Aggregated comparison data is empty, cannot generate snapshots.")


    else: logging.warning("No hourly comparison data was generated.")

    if all_hourly_rnds:
         final_rnd_df = pd.concat(all_hourly_rnds, ignore_index=True)
         try: final_rnd_df.to_csv(RND_OUTPUT_FILE, index=False, float_format="%.8f"); logging.info(f"Aggregated RND results saved to: {RND_OUTPUT_FILE}")
         except Exception as e: logging.error(f"Failed to save aggregated RND results: {e}")

         # Plot last density snapshot
         if not final_comparison_df.empty and snapshot_timestamps: # Check snapshot_timestamps exists
              last_ts = snapshot_timestamps[-1]
              rnd_data_exists = (final_rnd_df['timestamp'] == last_ts).any()
              comp_data_exists = (final_comparison_df['timestamp'] == last_ts).any()
              if rnd_data_exists and comp_data_exists:
                   # Ensure parsed_buckets exists before calling
                   if 'parsed_buckets' in locals():
                        plot_rnd_density(last_ts, final_rnd_df, final_comparison_df, parsed_buckets, PLOT_DIR)
                   else:
                        logging.warning("Parsed buckets not available for density plot.")
              else: logging.warning(f"Data missing for last snapshot {last_ts}, skipping density plot.")
         elif final_comparison_df.empty: logging.warning("Comparison data empty, skipping density plot.")

    else: logging.warning("No hourly RND data was generated.")

    logging.info(f"--- Script Finished: {pathlib.Path(__file__).name} ---")

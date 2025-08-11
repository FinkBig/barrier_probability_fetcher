#!/usr/bin/env python3
"""
compare_markets.py

A script to load, process, and compare barrier probability data from a
deribit_pricer log file with market data from Polymarket.

It aligns the two datasets on a minute-by-minute timeframe and generates
a separate expressive plot for each strike, focusing solely on probabilities
to highlight small changes. Charts are saved into a dedicated asset folder
with timestamped filenames.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import logging
from datetime import datetime

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def process_deribit_data(filepath: str) -> pd.DataFrame:
    """Loads and processes the Deribit CSV data."""
    logging.info(f"Processing Deribit data from: {filepath}")
    df = pd.read_csv(filepath)
    # Select necessary columns and copy to avoid SettingWithCopyWarning
    processed_df = df[['timestamp', 'strike', 'barrier_prob', 'asset', 'option_type']].copy()
    # Convert Unix timestamp to datetime, then round to the nearest minute
    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], unit='s').dt.round('1T')
    # Add direction based on option_type
    processed_df['direction'] = processed_df['option_type'].apply(lambda x: 'down' if x == 'put' else 'up')
    # Group by the minute, strike, and direction, taking the last recorded probability
    agg_df = processed_df.groupby(['timestamp', 'strike', 'direction']).last().reset_index()
    agg_df = agg_df[['timestamp', 'strike', 'direction', 'barrier_prob', 'asset']]
    return agg_df

def process_polymarket_data(filepath: str) -> pd.DataFrame:
    """Loads and processes the Polymarket CSV data."""
    logging.info(f"Processing Polymarket data from: {filepath}")
    df = pd.read_csv(filepath)
    # Rename the timestamp column for consistency
    df.rename(columns={'Timestamp (UTC)': 'timestamp'}, inplace=True)
    # Find all columns that represent a market (containing ↑ or ↓)
    market_columns = [col for col in df.columns if '↑' in col or '↓' in col]
    # Unpivot the DataFrame from a wide to a long format
    long_df = pd.melt(df, id_vars=['timestamp'], value_vars=market_columns, var_name='market', value_name='poly_prob')
    # Remove rows where there was no trade/price data
    long_df.dropna(subset=['poly_prob'], inplace=True)
    # Extract the numerical strike price from the market name (e.g., '↑ 150k' -> 150000)
    def extract_strike(market_name: str) -> int:
        num_match = re.search(r'(\d+)', market_name)
        if num_match:
            num = int(num_match.group(1))
            if 'k' in market_name.lower():
                return num * 1000
            return num
        return 0
    long_df['strike'] = long_df['market'].apply(extract_strike)
    # Extract direction: 'up' for ↑ (prob > strike), 'down' for ↓ (prob < strike)
    long_df['direction'] = long_df['market'].apply(lambda x: 'up' if '↑' in x else 'down')
    # Convert Unix timestamp to datetime and round to the nearest minute
    long_df['timestamp'] = pd.to_datetime(long_df['timestamp'], unit='s').dt.round('1T')
    # Group by the minute, strike, and direction, taking the mean probability
    agg_df = long_df.groupby(['timestamp', 'strike', 'direction'])['poly_prob'].mean().reset_index()
    return agg_df

def generate_comparison_charts(deribit_filepath: str, polymarket_filepath: str, output_dir: str):
    """Main function to generate expressive comparison charts for each strike."""
    # Process both datasets
    deribit_agg = process_deribit_data(deribit_filepath)
    poly_agg = process_polymarket_data(polymarket_filepath)

    if deribit_agg.empty or poly_agg.empty:
        logging.error("Could not process data from one or both files. Aborting.")
        return

    # Dynamically get asset name from the data
    asset_name = deribit_agg['asset'].iloc[0]
    
    # Create the asset-specific output directory
    asset_output_dir = os.path.join(output_dir, asset_name)
    os.makedirs(asset_output_dir, exist_ok=True)
    logging.info(f"Charts will be saved in: {asset_output_dir}")

    # Merge the two aggregated datasets on timestamp, strike, and direction
    logging.info("Merging processed dataframes...")
    comparison_df = pd.merge(deribit_agg, poly_agg, on=['timestamp', 'strike', 'direction'], how='outer')
    comparison_df.sort_values(by=['strike', 'timestamp'], inplace=True)

    # Clean the merged data by forward-filling missing values for continuous lines
    comparison_df['barrier_prob'] = comparison_df.groupby('strike')['barrier_prob'].ffill()
    comparison_df['poly_prob'] = comparison_df.groupby('strike')['poly_prob'].ffill()
    comparison_df.dropna(subset=['barrier_prob', 'poly_prob'], inplace=True)

    if comparison_df.empty:
        logging.warning("No overlapping data found between the two files. Cannot generate charts.")
        return

    # --- Plotting Loop ---
    strikes = sorted(comparison_df['strike'].unique())
    logging.info(f"Found {len(strikes)} unique strikes to plot.")

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for strike in strikes:
        strike_df = comparison_df[comparison_df['strike'] == strike]
        if strike_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(18, 10))  # Increased figure size
        
        # --- Probability Plot ---
        color_deribit = '#1f77b4'  # Blue for Deribit
        color_poly = '#ff4d4d'  # Red for Polymarket
        ax.set_xlabel('Time (UTC)', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.plot(strike_df['timestamp'], strike_df['barrier_prob'], 
                color=color_deribit, linestyle='-', linewidth=2, marker='o', markersize=6, 
                label='Deribit (Model)')
        ax.plot(strike_df['timestamp'], strike_df['poly_prob'], 
                color=color_poly, linestyle='--', linewidth=2, marker='^', markersize=6, 
                label='Polymarket (Crowd)')
        ax.tick_params(axis='y', labelsize=12)
        
        # Dynamic y-axis range to magnify small changes
        prob_min = min(strike_df['barrier_prob'].min(), strike_df['poly_prob'].min())
        prob_max = max(strike_df['barrier_prob'].max(), strike_df['poly_prob'].max())
        margin = (prob_max - prob_min) * 0.1  # 10% margin
        ax.set_ylim(max(0, prob_min - margin), min(1, prob_max + margin))
        
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # --- General Formatting ---
        direction = strike_df['direction'].iloc[0] if not strike_df['direction'].empty else ''
        plt.title(f'{asset_name} Strike {int(strike)} ({direction.upper()}): Deribit vs. Polymarket', fontsize=16, pad=15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))  # Show every 5 minutes
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=12)
        fig.tight_layout()

        # Save the chart
        output_filename = f"{asset_name}_strike_{int(strike)}_comparison_{current_timestamp}.png"
        output_filepath = os.path.join(asset_output_dir, output_filename)
        plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
        logging.info(f"Successfully generated chart: {output_filepath}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Deribit and Polymarket barrier probabilities for each strike.")
    parser.add_argument(
        "--deribit-csv",
        required=True,
        help="Path to the Deribit pricer CSV log file."
    )
    parser.add_argument(
        "--polymarket-csv",
        required=True,
        help="Path to the Polymarket price data CSV file."
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_outputs",
        help="Base directory to save the output comparison charts."
    )
    args = parser.parse_args()

    generate_comparison_charts(
        deribit_filepath=args.deribit_csv,
        polymarket_filepath=args.polymarket_csv,
        output_dir=args.output_dir
    )
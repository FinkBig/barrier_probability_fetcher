# Deribit Barrier Probability Fetcher

A real-time WebSocket-based barrier probability calculator for Deribit options. This tool continuously monitors option prices and calculates barrier hit probabilities for specified strikes, providing live streaming updates and visual charts.

## Features

- **Real-time WebSocket streaming** from Deribit API
- **Barrier probability calculations** for both call and put options
- **Live bar chart visualization** showing probabilities across strikes
- **Clean formatted terminal output** with real-time table display
- **JSON output** for easy integration with other systems (use `--minimal` flag)
- **Automatic monthly expiry selection** targeting end-of-month expiries (e.g., Aug 29, 2025)
- **Robust error handling** with automatic reconnection and spot price fallback
- **YAML configuration** for easy customization
- **Comparison tools** for Deribit vs Polymarket analysis
- **Persistent CSV logging** with append mode for historical data

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

- `numpy` - Numerical computations
- `websockets` - WebSocket client for Deribit API
- `scipy` - Statistical functions for probability calculations
- `matplotlib` - Real-time chart visualization
- `pyyaml` - YAML configuration file support

## Usage

### Basic Usage

```bash
python deribit_pricer.py --asset BTC --strike 100000 110000 120000
```

### Command Line Options

- `--asset`: Asset symbol (e.g., BTC, ETH) - **Required**
- `--strike`: List of strike prices - **Required**
- `--interval`: Update interval in seconds (default: 5)
- `--minimal`: Enable minimal output mode (JSON format for data processing)

### Configuration

The script uses `config.yaml` for configuration. Key settings:

```yaml
websocket_uri: "wss://www.deribit.com/ws/api/v2"
rpc_timeout: 8.0
min_iv: 0.5
max_iv: 1000.0
interval: 5
spot_timeout: 60  # seconds to wait for spot price update
iv_timeout: 30    # seconds to wait for IV data
```

### Examples

**Monitor BTC options with multiple strikes:**

```bash
python deribit_pricer.py --asset BTC --strike 100000 110000 120000
```

**Minimal output mode for data processing:**

```bash
python deribit_pricer.py --asset ETH --strike 3300 4000 4200 --minimal
```

**Compare Deribit vs Polymarket data:**

```bash
python compare_markets.py --asset BTC --strike 100000 110000 125000 135000
```

## Output

### Terminal Display

By default, the script displays a clean formatted table in the terminal:

```
======================================================================
                    DERIBIT BARRIER PROBABILITY PRICER
----------------------------------------------------------------------
Asset: BTC           Spot Price: $121,208.53
Expiry: 2025-08-29T08:00:00+00:00
----------------------------------------------------------------------
      Strike |   Type | Mark IV (%) |   Barrier Prob.
----------------------------------------------------------------------
   $100,000 |   Put |      50.19% |         9.36%
   $110,000 |   Put |      38.50% |        26.89%
   $125,000 |  Call |      32.64% |        65.93%
   $135,000 |  Call |      35.44% |        16.03%
----------------------------------------------------------------------
Last updated: 2025-08-11 10:59:10 UTC
```

### JSON Output Format (with --minimal flag)

When using `--minimal`, the script outputs JSON lines to stdout:

```json
{
  "asset": "BTC",
  "spot": 52000.0,
  "timestamp": 1640995200,
  "results": [
    {
      "strike": 50000.0,
      "option_type": "call",
      "expiry_ts": 1640995200,
      "expiry_iso": "2022-01-01T00:00:00+00:00",
      "last_unix": 1640995200,
      "last_iso": "2022-01-01T00:00:00+00:00",
      "last_underlying": 52000.0,
      "last_iv_pct": 85.5,
      "last_barrier_prob": 0.75,
      "n_points": 100
    }
  ]
}
```

### Minimal Output Format

When using `--minimal` flag:

```json
{
  "asset": "BTC",
  "timestamp": 1640995200,
  "probabilities": [
    {"strike": 50000.0, "barrier_prob": 0.75},
    {"strike": 60000.0, "barrier_prob": 0.45}
  ]
}
```

## Features Explained

### Barrier Probability Calculation

The script calculates the probability that the underlying asset will hit the strike price (barrier) before expiration:

- **For calls**: Probability of hitting the upper barrier (strike price)
- **For puts**: Probability of hitting the lower barrier (strike price)

The calculation uses the Black-Scholes framework with:
- Risk-free rate (r) = 0
- Dividend yield (q) = 0
- Current implied volatility from market data

### Monthly Expiry Selection

The tool automatically selects the nearest monthly expiry for each strike, specifically targeting end-of-month expiries (e.g., August 29, 2025) to align with Polymarket's monthly markets. This ensures optimal liquidity and consistent pricing across platforms.

### Real-time Visualization

A live bar chart displays:
- Strike prices on the x-axis
- Barrier probabilities on the y-axis (0-100% scale)
- Automatic updates every interval seconds
- Asset name and expiry date in the title
- Percentage labels on bars for easy reading

## Technical Details

- **WebSocket Connection**: Connects to Deribit's public WebSocket API
- **Data Sources**: 
  - Perpetual futures for spot price updates
  - Option tickers for implied volatility
- **Error Handling**: Automatic reconnection on connection loss with exponential backoff
- **Spot Price Fallback**: API polling when WebSocket updates are stale
- **Data Storage**: Maintains rolling window of last 100 data points per instrument
- **Volatility Filtering**: Only processes data with implied volatility between 0.5% and 1000%
- **CSV Logging**: Persistent logging to `data_logs/<asset>/<asset>_probabilities.csv` in append mode

## System Requirements

- Python 3.7+
- macOS/Linux/Windows
- Internet connection for Deribit API access
- Display capability for matplotlib charts (or use headless mode)

## Additional Scripts

### compare_markets.py

A comparison tool that analyzes Deribit barrier probabilities against Polymarket prediction market data:

- **Real-time Deribit data** via WebSocket
- **Polymarket data** from CSV files
- **Side-by-side comparison** with visual charts
- **Statistical analysis** of probability differences
- **Export functionality** for further analysis

Usage:
```bash
python compare_markets.py --asset BTC --strike 100000 110000 125000 135000
```

## Troubleshooting

**Connection Issues:**
- Check internet connectivity
- Verify Deribit API status
- Script will automatically reconnect after 5 seconds

**Chart Display Issues:**
- On macOS, the script uses TkAgg backend for matplotlib
- For headless servers, consider using a different matplotlib backend

**Memory Usage:**
- The script maintains a rolling window of 100 data points per instrument
- Memory usage scales with the number of strikes monitored

**Spot Price Issues:**
- If spot price becomes static, the script will automatically poll the API
- Check logs for "Updated spot via API poll" messages
- Ensure perpetual futures are trading for real-time updates

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with Deribit's API terms of service when using this tool.

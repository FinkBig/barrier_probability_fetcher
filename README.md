# Deribit Barrier Probability Fetcher

A real-time WebSocket-based barrier probability calculator for Deribit options. This tool continuously monitors option prices and calculates barrier hit probabilities for specified strikes, providing live streaming updates and visual charts.

## Features

- **Real-time WebSocket streaming** from Deribit API
- **Barrier probability calculations** for both call and put options
- **Live bar chart visualization** showing probabilities across strikes
- **JSON output** for easy integration with other systems
- **Automatic monthly expiry selection** for optimal liquidity
- **Robust error handling** with automatic reconnection
- **Minimal output mode** for streamlined data processing

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

## Usage

### Basic Usage

```bash
python deribit_pricer.py --asset BTC --strike 50000 60000 70000
```

### Command Line Options

- `--asset`: Asset symbol (e.g., BTC, ETH) - **Required**
- `--strike`: List of strike prices - **Required**
- `--interval`: Update interval in seconds (default: 5)
- `--minimal`: Enable minimal output mode (asset, strike, probability only)

### Examples

**Monitor BTC options with multiple strikes:**

```bash
python deribit_pricer.py --asset BTC --strike 45000 50000 55000 60000
```

**Monitor ETH options with custom interval:**

```bash
python deribit_pricer.py --asset ETH --strike 3000 3500 4000 --interval 10
```

**Minimal output mode for data processing:**

```bash
python deribit_pricer.py --asset BTC --strike 50000 60000 --minimal
```

## Output

### Standard Output Format

The script outputs JSON lines to stdout with the following structure:

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

The tool automatically selects the nearest monthly expiry for each strike to ensure optimal liquidity and consistent pricing.

### Real-time Visualization

A live bar chart displays:
- Strike prices on the x-axis
- Barrier probabilities on the y-axis (0-1 scale)
- Automatic updates every interval seconds
- Asset name and expiry date in the title

## Technical Details

- **WebSocket Connection**: Connects to Deribit's public WebSocket API
- **Data Sources**: 
  - Perpetual futures for spot price updates
  - Option tickers for implied volatility
- **Error Handling**: Automatic reconnection on connection loss
- **Data Storage**: Maintains rolling window of last 100 data points per instrument
- **Volatility Filtering**: Only processes data with implied volatility between 0.5% and 1000%

## System Requirements

- Python 3.7+
- macOS/Linux/Windows
- Internet connection for Deribit API access
- Display capability for matplotlib charts (or use headless mode)

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

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with Deribit's API terms of service when using this tool.

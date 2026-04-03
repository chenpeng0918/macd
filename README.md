# MACD Market Backtest

This project is a multi-market backtesting study built around `MACD`, Bollinger Bands, and a combined signal strategy. It currently covers:

- SSE Composite Index
- Nikkei 225
- Dalian Commodity Exchange continuous futures for soybean meal, corn, coke, and iron ore

The repository keeps both the research notebook and a reusable Python script, which makes it suitable for coursework, strategy prototyping, and GitHub presentation.

## Project Structure

```text
macd/
├── macd.ipynb
├── macd_backtest.py
├── requirements.txt
├── README.md
├── outputs/
│   └── .gitkeep
└── tests/
    └── test_macd_backtest.py
```

## Features

- Computes `MACD` and Bollinger Bands indicators
- Generates three strategy types:
  - `MACD` crossover
  - Bollinger Bands breakout
  - Combined `MACD + Bollinger Bands` strategy
- Runs backtests across index and futures markets
- Reports:
  - Total return
  - CAGR
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Number of trades
- Exports summary tables and return descriptive statistics

## Environment Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Notebook

```bash
jupyter notebook macd.ipynb
```

### Option 2: Run the Script

By default, the script downloads market data online, prints results, and saves CSV reports.

```bash
python macd_backtest.py --no-plots
```

If you want charts to be displayed:

```bash
python macd_backtest.py
```

You can also pass custom parameters:

```bash
python macd_backtest.py --start-date 2015-01-01 --cost-bps 2 --output-dir outputs --no-plots
```

## Tests

The repository includes a small unit test suite covering:

- No premature entries during the MACD warm-up period
- Proper alignment between executed positions and transaction costs
- Trade count and win rate measured by actual trade segments

Run the tests with:

```bash
python -m unittest discover -s tests
```

## Output Files

After running the script or notebook, reports are saved to `outputs/`:

- `outputs/strategy_performance_summary.csv`
- `outputs/return_descriptive_statistics.csv`

## Data Sources

This project uses the following online data sources:

- `AkShare`
- `Yahoo Finance`

An internet connection is required, and downloaded market data may vary slightly over time.

## Before Uploading to GitHub

- Keep notebook outputs empty or minimal to avoid committing large execution artifacts
- Do not commit local virtual environment directories
- Add a `LICENSE` file if you plan to make the repository public

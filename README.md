# Stock Price Analysis using Markov Chains

This project analyzes stock price movements using Markov chains. It includes data fetching, preprocessing, Markov modeling, simulations, and backtesting. It also supports multi-ticker comparisons.

## Features
- Fetch stock data from Yahoo Finance.
- Assign states based on daily returns.
- Build and analyze transition matrices.
- Simulate future price paths.
- Backtest the model and evaluate performance.
- Compare multiple tickers.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/EmmaAdjei007/stock-markov-analysis.git

## Dependencies:
2. ```bash
    pip install -r requiremnts.txt

## Usage
```bash
    python src/cli.py --tickers AAPL MSFT --start_date 2020-01-01 --end_date 2023-01-01



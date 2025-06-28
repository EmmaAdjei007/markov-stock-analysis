import yfinance as yf
import os
import pandas as pd
from typing import List, Dict

def safe_date_parser(x):
    """
    Parse dates using pd.to_datetime with errors='coerce' so that any invalid dates become NaT.
    """
    dt = pd.to_datetime(x, errors='coerce')
    return dt

def get_stock_data(tickers: List[str], start_date, end_date, cache_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with caching and robust error handling.
    start_date and end_date can be datetime objects; they will be formatted safely for filenames.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data = {}

    # Convert dates to safe strings for filenames
    if hasattr(start_date, "strftime"):
        start_str = start_date.strftime("%Y-%m-%d")
    else:
        start_str = str(start_date)
    if hasattr(end_date, "strftime"):
        end_str = end_date.strftime("%Y-%m-%d")
    else:
        end_str = str(end_date)

    for ticker in tickers:
        cache_file = os.path.join(cache_dir, f"{ticker}_{start_str}_{end_str}.csv")
        
        if os.path.exists(cache_file):
            print(f"Loading cached data for {ticker} from {cache_file}...")
            try:
                # Use safe_date_parser to convert dates; invalid dates become NaT
                df = pd.read_csv(
                    cache_file, 
                    index_col='Date', 
                    parse_dates=True, 
                    date_parser=safe_date_parser
                )
                # Drop rows where the index could not be parsed
                df = df[df.index.notna()]
                data[ticker] = df
            except Exception as e:
                print(f"Error reading cached data for {ticker} with custom parser: {e}")
                try:
                    df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                    # Ensure the index is a proper DatetimeIndex
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[df.index.notna()]
                    data[ticker] = df
                except Exception as e2:
                    print(f"Fallback error reading cached data for {ticker}: {e2}")
        else:
            try:
                print(f"Fetching data for {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_str, end=end_str)
                if df.empty:
                    print(f"No data found for {ticker} in the specified date range.")
                else:
                    try:
                        df.to_csv(cache_file)
                    except Exception as e:
                        print(f"Error writing cache file for {ticker}: {e}")
                    data[ticker] = df
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
    
    return data

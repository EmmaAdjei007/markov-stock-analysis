"""
Enhanced stock data fetcher with caching, error handling, and data validation.
"""

import yfinance as yf
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import time
import requests
import pytz

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality checks."""
    ticker: str
    total_records: int
    missing_records: int
    data_completeness: float
    date_range: tuple
    has_gaps: bool
    gap_details: Optional[List[dict]] = None


class StockDataFetcher:
    """
    Enhanced stock data fetcher with intelligent caching and validation.

    Features:
        - Automatic caching to avoid redundant API calls
        - Robust error handling and retry logic
        - Data quality validation and gap detection
        - Multiple ticker batch processing
        - Configurable cache expiry

    Example:
        >>> fetcher = StockDataFetcher(cache_dir="data/raw")
        >>> data = fetcher.fetch(["AAPL", "MSFT"], start_date="2020-01-01")
        >>> quality = fetcher.validate_data(data["AAPL"])
    """

    def __init__(
        self,
        cache_dir: str = "data/raw",
        cache_expiry_days: int = 7,
        auto_cache: bool = True
    ):
        """
        Initialize the stock data fetcher.

        Args:
            cache_dir: Directory for caching downloaded data
            cache_expiry_days: Number of days before cache is considered stale
            auto_cache: Automatically cache fetched data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_days = cache_expiry_days
        self.auto_cache = auto_cache

        # Setup custom session with headers to avoid rate limiting
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

    def _safe_date_parser(self, date_str: str) -> Optional[pd.Timestamp]:
        """
        Safely parse date strings with error handling.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime or None if invalid
        """
        try:
            return pd.to_datetime(date_str, errors='coerce')
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            return None

    def _format_date(self, date: Union[str, datetime, pd.Timestamp]) -> str:
        """
        Convert various date formats to standardized string.

        Args:
            date: Date in various formats

        Returns:
            Date string in YYYY-MM-DD format
        """
        if isinstance(date, str):
            return date
        elif hasattr(date, "strftime"):
            return date.strftime("%Y-%m-%d")
        else:
            return str(date)

    def _get_cache_path(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Path:
        """Get cache file path for specific ticker and date range."""
        return self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cache file exists and is not expired.

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid and not expired
        """
        if not cache_path.exists():
            return False

        # Check cache age
        file_age_days = (
            datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        ).days

        return file_age_days < self.cache_expiry_days

    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from cache with robust error handling.

        Args:
            cache_path: Path to cache file

        Returns:
            DataFrame or None if loading fails
        """
        try:
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_csv(
                cache_path,
                index_col='Date',
                parse_dates=True,
                date_parser=self._safe_date_parser
            )

            # Remove rows with invalid dates
            df = df[df.index.notna()]

            if df.empty:
                logger.warning(f"Cached data is empty: {cache_path}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {e}")
            return None

    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> bool:
        """
        Save DataFrame to cache.

        Args:
            df: DataFrame to save
            cache_path: Destination path

        Returns:
            True if successful
        """
        try:
            df.to_csv(cache_path)
            logger.info(f"Saved data to cache: {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
            return False

    def fetch(
        self,
        tickers: Union[str, List[str]],
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for one or multiple tickers using bulk download.

        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for data (default: 2 years ago)
            end_date: End date for data (default: today)
            use_cache: Use cached data if available

        Returns:
            Dictionary mapping tickers to DataFrames

        Example:
            >>> fetcher = StockDataFetcher()
            >>> data = fetcher.fetch(["AAPL", "MSFT"], start_date="2020-01-01")
            >>> aapl_df = data["AAPL"]
        """
        # Handle single ticker
        if isinstance(tickers, str):
            tickers = [tickers]

        # Default date range with timezone awareness
        tz = pytz.timezone("America/New_York")  # NYSE timezone

        # Convert end_date to timezone-aware datetime
        if end_date is None:
            end_date = tz.localize(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
        elif isinstance(end_date, str):
            end_date = tz.localize(pd.to_datetime(end_date).to_pydatetime())
        else:
            # Check if it's a date object (not datetime)
            if hasattr(end_date, 'date') and not hasattr(end_date, 'hour'):
                # It's a date object, convert to datetime
                from datetime import date as date_type
                if isinstance(end_date, date_type) and not isinstance(end_date, datetime):
                    end_date = tz.localize(datetime.combine(end_date, datetime.min.time()))
            elif isinstance(end_date, datetime):
                if end_date.tzinfo is None:
                    end_date = tz.localize(end_date)

        # Convert start_date to timezone-aware datetime
        if start_date is None:
            start_date = tz.localize((datetime.now() - timedelta(days=730)).replace(hour=0, minute=0, second=0, microsecond=0))
        elif isinstance(start_date, str):
            start_date = tz.localize(pd.to_datetime(start_date).to_pydatetime())
        else:
            # Check if it's a date object (not datetime)
            if hasattr(start_date, 'date') and not hasattr(start_date, 'hour'):
                # It's a date object, convert to datetime
                from datetime import date as date_type
                if isinstance(start_date, date_type) and not isinstance(start_date, datetime):
                    start_date = tz.localize(datetime.combine(start_date, datetime.min.time()))
            elif isinstance(start_date, datetime):
                if start_date.tzinfo is None:
                    start_date = tz.localize(start_date)

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)

        data = {}
        tickers_to_fetch = []

        # Check cache for each ticker
        for ticker in tickers:
            cache_path = self._get_cache_path(ticker, start_str, end_str)

            # Try to load from cache
            if use_cache and self._is_cache_valid(cache_path):
                df = self._load_from_cache(cache_path)
                if df is not None:
                    data[ticker] = df
                    continue

            tickers_to_fetch.append(ticker)

        # Fetch remaining tickers using bulk download (avoids rate limiting)
        if tickers_to_fetch:
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching data for {tickers_to_fetch} from {start_str} to {end_str} (attempt {attempt + 1}/{max_retries})")

                    # Use bulk download with progress disabled to avoid 429 errors
                    # Pass timezone-aware datetime objects instead of strings
                    downloaded = yf.download(
                        tickers_to_fetch,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        group_by='ticker' if len(tickers_to_fetch) > 1 else None,
                        threads=False,  # Disable threading to reduce rate limit issues
                        ignore_tz=False  # Keep timezone information
                    )

                    if downloaded.empty:
                        logger.warning(f"No data downloaded for {tickers_to_fetch}")
                        break

                    # Process downloaded data
                    if len(tickers_to_fetch) == 1:
                        ticker = tickers_to_fetch[0]
                        if not downloaded.empty:
                            data[ticker] = downloaded
                            cache_path = self._get_cache_path(ticker, start_str, end_str)
                            if self.auto_cache:
                                self._save_to_cache(downloaded, cache_path)
                    else:
                        # Multiple tickers
                        for ticker in tickers_to_fetch:
                            try:
                                if ticker in downloaded.columns.levels[0]:
                                    df = downloaded[ticker]
                                    if not df.empty:
                                        data[ticker] = df
                                        cache_path = self._get_cache_path(ticker, start_str, end_str)
                                        if self.auto_cache:
                                            self._save_to_cache(df, cache_path)
                            except Exception as e:
                                logger.warning(f"Error processing {ticker}: {e}")

                    # Success - break retry loop
                    break

                except Exception as e:
                    logger.error(f"Error fetching data (attempt {attempt + 1}/{max_retries}): {e}")

                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to fetch data after {max_retries} attempts")

        return data

    def validate_data(
        self,
        df: pd.DataFrame,
        ticker: str = "Unknown"
    ) -> DataQualityReport:
        """
        Validate data quality and detect gaps.

        Args:
            df: DataFrame to validate
            ticker: Ticker symbol for reporting

        Returns:
            DataQualityReport with validation results
        """
        total_records = len(df)
        missing_records = df.isnull().sum().sum()
        completeness = 1.0 - (missing_records / (total_records * len(df.columns)))

        date_range = (df.index.min(), df.index.max())

        # Check for gaps in trading days
        expected_days = pd.date_range(start=date_range[0], end=date_range[1], freq='B')
        actual_days = df.index
        missing_days = expected_days.difference(actual_days)

        has_gaps = len(missing_days) > 0
        gap_details = None

        if has_gaps:
            gap_details = [
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "day_of_week": day.strftime("%A")
                }
                for day in missing_days[:10]  # Limit to first 10
            ]

        return DataQualityReport(
            ticker=ticker,
            total_records=total_records,
            missing_records=missing_records,
            data_completeness=completeness,
            date_range=date_range,
            has_gaps=has_gaps,
            gap_details=gap_details
        )

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the latest available price for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Latest closing price or None
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error fetching latest price for {ticker}: {e}")
            return None

    def clear_cache(self, ticker: Optional[str] = None) -> int:
        """
        Clear cached data files.

        Args:
            ticker: Specific ticker to clear, or None to clear all

        Returns:
            Number of files removed
        """
        removed_count = 0

        if ticker:
            # Clear specific ticker
            pattern = f"{ticker}_*.csv"
        else:
            # Clear all
            pattern = "*.csv"

        for file_path in self.cache_dir.glob(pattern):
            try:
                file_path.unlink()
                removed_count += 1
                logger.info(f"Removed cache file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")

        return removed_count


# Backward compatibility function
def get_stock_data(
    tickers: List[str],
    start_date,
    end_date,
    cache_dir: str = "data/raw"
) -> Dict[str, pd.DataFrame]:
    """
    Legacy function for backward compatibility.

    Use StockDataFetcher class for new code.
    """
    fetcher = StockDataFetcher(cache_dir=cache_dir)
    return fetcher.fetch(tickers, start_date, end_date)

import pytest
from src.data_fetcher import get_stock_data

def test_get_stock_data_single_ticker():
    """
    Test fetching data for a single ticker.
    """
    data = get_stock_data(["AAPL"], "2020-01-01", "2023-01-01")
    assert "AAPL" in data, "AAPL data should be fetched"
    assert not data["AAPL"].empty, "AAPL data should not be empty"

def test_get_stock_data_multiple_tickers():
    """
    Test fetching data for multiple tickers.
    """
    data = get_stock_data(["AAPL", "MSFT"], "2020-01-01", "2023-01-01")
    assert len(data) == 2, "Data should contain two tickers"
    assert "AAPL" in data, "AAPL data should be fetched"
    assert "MSFT" in data, "MSFT data should be fetched"
    assert not data["AAPL"].empty, "AAPL data should not be empty"
    assert not data["MSFT"].empty, "MSFT data should not be empty"

def test_get_stock_data_invalid_ticker():
    """
    Test fetching data for an invalid ticker.
    """
    data = get_stock_data(["INVALID_TICKER"], "2020-01-01", "2023-01-01")
    assert "INVALID_TICKER" not in data, "Invalid ticker should not be in the data"
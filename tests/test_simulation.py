import pytest
import numpy as np
from src.simulation import simulate_price_path

def test_simulate_price_path():
    """
    Test the price path simulation.
    """
    initial_price = 100
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])  # 2 states
    returns_by_state = [np.array([-1, 1]), np.array([-2, 2])]  # Example returns
    days = 10
    volatility = 0.01
    
    prices = simulate_price_path(initial_price, transition_matrix, returns_by_state, days, volatility)
    
    # Check output shape
    assert len(prices) == days + 1, "Price path should have length days + 1"
    
    # Check initial price
    assert prices[0] == initial_price, "Initial price should match input"
    
    # Check price changes
    assert np.all(prices > 0), "Prices should be positive"

def test_simulate_price_path_zero_days():
    """
    Test price path simulation with zero days.
    """
    initial_price = 100
    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    returns_by_state = [np.array([-1, 1]), np.array([-2, 2])]
    days = 0
    
    prices = simulate_price_path(initial_price, transition_matrix, returns_by_state, days)
    
    # Check output shape
    assert len(prices) == 1, "Price path should have length 1 for zero days"
    assert prices[0] == initial_price, "Initial price should match input"
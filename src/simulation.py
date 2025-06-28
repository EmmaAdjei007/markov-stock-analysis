import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tabulate import tabulate
import os

def simulate_price_path(initial_price: float, transition_matrix: np.ndarray, returns_by_state: List[np.ndarray], days: int, volatility: float = 0.01) -> np.ndarray:
    """
    Simulate price path with added noise for volatility.
    """
    n_states = transition_matrix.shape[0]
    prices = [initial_price]
    current_state = np.random.choice(n_states)
    
    for _ in range(days):
        next_state = np.random.choice(n_states, p=transition_matrix[current_state])
        return_pct = np.random.choice(returns_by_state[next_state])
        noise = np.random.normal(0, volatility)  # Add noise for volatility
        next_price = prices[-1] * (1 + (return_pct + noise) / 100)
        prices.append(next_price)
        current_state = next_state
        
    return np.array(prices)

def calculate_performance_metrics(actual_prices: np.ndarray = None,
                                 predicted_prices: np.ndarray = None,
                                 simulations: np.ndarray = None,
                                 df: pd.DataFrame = None,
                                 current_price: float = None,
                                 forecast_days: int = 100) -> Tuple[Dict[str, float], List]:
    """
    Calculate comprehensive performance metrics and statistics.
    
    Returns:
        tuple: (metrics_dict, summary_table)
    """
    metrics = {}
    summary_stats = []

    # Backtest metrics
    if actual_prices is not None and predicted_prices is not None:
        if len(actual_prices) != len(predicted_prices):
            raise ValueError("Actual and predicted prices must have same length")
            
        metrics['RMSE'] = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
        metrics['Hit Rate'] = np.mean(np.sign(actual_prices[1:] - actual_prices[:-1]) == 
                             np.sign(predicted_prices[1:] - predicted_prices[:-1]))
        metrics['Sharpe Ratio'] = np.mean(actual_prices) / np.std(actual_prices)

    # Simulation statistics
    if simulations is not None and df is not None and current_price is not None:
        if simulations.size == 0:
            raise ValueError("Simulations array cannot be empty")
        if 'Daily_Return' not in df.columns:
            raise ValueError("DataFrame must contain 'Daily_Return' column")

        final_prices = simulations[:, -1]
        returns = (final_prices - current_price) / current_price
        historical_vol = df['Daily_Return'].std() * np.sqrt(252) * 100 # Annualized historical volatility
        sim_vol = np.std(simulations) / np.mean(simulations) * 100

        # Expected Return
        expected_return = np.mean(returns) * 100  # Convert to percentage
        metrics['Simulated Mean Return'] = expected_return

        # Volatility (Standard Deviation of returns)
        volatility = np.std(returns) * 100  # Convert to percentage
        metrics['Simulated Volatility'] = volatility

        # Value at Risk (95% Confidence Level)
        var_95 = np.percentile(returns, 5) * 100  # Convert to percentage
        metrics['Simulated VaR (95%)'] = var_95

        # Best Case (5%)
        best_case_5 = np.percentile(returns, 95) * 100  # Convert to percentage
        metrics['Simulated Best Case (5%)'] = best_case_5


        summary_stats = [
            ['Current Price', f"${current_price:.2f}"],
            ['Forecast Horizon', f"{forecast_days} days"],
            ['Predicted Mean Price', f"${np.mean(final_prices):.2f}"],
            ['Price Range', f"${np.min(final_prices):.2f} - ${np.max(final_prices):.2f}"],
            ['Historical Volatility', f"{historical_vol:.2f}%"],
            ['Simulated Volatility', f"{sim_vol:.2f}%"],
            ['90% CI', f"${np.percentile(final_prices, 5):.2f} - ${np.percentile(final_prices, 95):.2f}"],
            ['95% CI', f"${np.percentile(final_prices, 2.5):.2f} - ${np.percentile(final_prices, 97.5):.2f}"]
        ]

        # Add simulation metrics to main dict
        metrics.update({
            'Simulated Mean Price': np.mean(final_prices),
            'Simulated Price STD': np.std(final_prices),
            'Simulated Min Price': np.min(final_prices),
            'Simulated Max Price': np.max(final_prices)
        })

    # Print formatted table if simulation stats exist
    if summary_stats:
        print(f"\n📊 Performance Summary ({forecast_days} days)")
        print(tabulate(summary_stats, 
                     headers=['Metric', 'Value'], 
                     tablefmt='psql',
                     colalign=("left", "right")))

    return metrics, summary_stats


def save_performance_metrics(metrics: Dict[str, float], 
                            filename: str = "reports/performance_metrics.md"):
    """Save metrics to markdown file with proper directory creation."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("# Performance Metrics\n\n")
        for key, value in metrics.items():
            f.write(f"- **{key}**: {value:.4f}\n")
        f.write("\n![State Distribution](figures/state_distribution.png)")



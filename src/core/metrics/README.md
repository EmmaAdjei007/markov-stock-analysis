# Performance Metrics Module

Comprehensive performance metrics module for financial analysis, providing a complete suite of metrics to evaluate trading strategies, portfolio performance, and prediction accuracy.

## Features

### Risk-Adjusted Returns
- **Sharpe Ratio**: Risk-adjusted return metric (Mean Return - Risk Free Rate) / Standard Deviation
- **Sortino Ratio**: Downside risk-adjusted return (penalizes only downside volatility)
- **Calmar Ratio**: Return/Maximum Drawdown ratio
- **Information Ratio**: Risk-adjusted active return vs benchmark

### Drawdown Analysis
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Maximum Drawdown Duration**: Longest period in drawdown (days)
- **Current Drawdown**: Current decline from peak

### Risk Metrics
- **Value at Risk (VaR)**: Maximum expected loss at confidence levels (90%, 95%, 99%)
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold (Expected Shortfall)
- **Annualized Volatility**: Standard deviation of returns (annualized)

### Prediction Accuracy
- **Hit Rate**: Directional prediction accuracy (0-1)
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Portfolio Metrics
- **Alpha**: Excess return not explained by market movements
- **Beta**: Systematic risk (sensitivity to market)
- **Cumulative Return**: Total return over period

### Distribution Statistics
- **Skewness**: Return distribution asymmetry
- **Kurtosis**: Return distribution tail behavior
- **Win Rate**: Percentage of positive return days
- **Win/Loss Ratio**: Average win to average loss ratio

## Installation

The module requires the following dependencies:

```bash
pip install numpy pandas scipy tabulate
```

## Quick Start

### Basic Usage

```python
from src.core.metrics.performance import PerformanceMetrics
import numpy as np

# Create metrics calculator
metrics = PerformanceMetrics(risk_free_rate=0.02)

# Sample returns data
returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])

# Calculate individual metrics
sharpe = metrics.sharpe_ratio(returns)
print(f"Sharpe Ratio: {sharpe:.4f}")

sortino = metrics.sortino_ratio(returns)
print(f"Sortino Ratio: {sortino:.4f}")

# Calculate prices from returns
prices = (1 + returns).cumprod() * 100

max_dd = metrics.max_drawdown(prices)
print(f"Maximum Drawdown: {max_dd:.2f}%")
```

### Comprehensive Analysis

```python
from src.core.metrics.performance import PerformanceMetrics
import pandas as pd

# Your return series
returns = pd.Series([...])  # Your data here
prices = pd.Series([...])   # Your price data

# Calculate all metrics at once
metrics = PerformanceMetrics(risk_free_rate=0.02)

report = metrics.calculate_all(
    returns=returns,
    prices=prices,
    ticker="AAPL"
)

# Access metrics
print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
print(f"Max Drawdown: {report.max_drawdown:.2f}%")
print(f"VaR (95%): {report.var_95:.2f}%")
```

### Generate Reports

```python
from src.core.metrics.performance import generate_summary_table

# Generate formatted table
table = generate_summary_table(report)
print(table)

# Export to JSON
report.to_json("reports/performance.json")

# Export to Markdown
report.to_markdown("reports/performance.md")
```

### Compare Strategies

```python
from src.core.metrics.performance import PerformanceMetrics, compare_strategies

metrics = PerformanceMetrics()

# Calculate reports for multiple strategies
report1 = metrics.calculate_all(returns=strategy1_returns, ticker="Strategy A")
report2 = metrics.calculate_all(returns=strategy2_returns, ticker="Strategy B")
report3 = metrics.calculate_all(returns=strategy3_returns, ticker="Strategy C")

# Compare side-by-side
comparison = compare_strategies([report1, report2, report3])
print(comparison)
```

### Prediction Accuracy

```python
# With predicted prices for accuracy metrics
report = metrics.calculate_all(
    returns=returns,
    prices=actual_prices,
    predicted_prices=predicted_prices,
    ticker="AAPL"
)

print(f"Hit Rate: {report.hit_rate:.2%}")
print(f"RMSE: {report.rmse:.4f}")
print(f"MAE: {report.mae:.4f}")
```

### Portfolio Analysis

```python
# With benchmark for portfolio metrics
report = metrics.calculate_all(
    returns=portfolio_returns,
    benchmark_returns=sp500_returns,
    ticker="My Portfolio"
)

print(f"Alpha: {report.alpha:.4f}")
print(f"Beta: {report.beta:.4f}")
print(f"Information Ratio: {report.information_ratio:.4f}")
```

## API Reference

### PerformanceMetrics Class

```python
PerformanceMetrics(
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
)
```

**Methods:**

- `sharpe_ratio(returns, annualize=True)` → float
- `sortino_ratio(returns, target_return=0.0, annualize=True)` → float
- `calmar_ratio(returns, prices=None)` → float
- `max_drawdown(prices)` → float
- `max_drawdown_duration(prices)` → int
- `current_drawdown(prices)` → float
- `value_at_risk(returns, confidence_level=0.95)` → float
- `conditional_var(returns, confidence_level=0.95)` → float
- `hit_rate(actual, predicted)` → float
- `rmse(actual, predicted)` → float
- `mae(actual, predicted)` → float
- `mape(actual, predicted)` → float
- `annualized_return(returns, periods_per_year=None)` → float
- `annualized_volatility(returns, periods_per_year=None)` → float
- `information_ratio(returns, benchmark_returns, annualize=True)` → float
- `alpha_beta(returns, benchmark_returns, annualize=True)` → Tuple[float, float]
- `calculate_all(...)` → MetricsReport

### MetricsReport Dataclass

Contains all calculated performance metrics with the following attributes:

```python
@dataclass
class MetricsReport:
    ticker: str
    calculation_date: str

    # Risk-Adjusted Returns
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    information_ratio: Optional[float]

    # Drawdown Metrics
    max_drawdown: Optional[float]
    max_drawdown_duration: Optional[int]
    current_drawdown: Optional[float]

    # Value at Risk
    var_90: Optional[float]
    var_95: Optional[float]
    var_99: Optional[float]
    cvar_95: Optional[float]

    # Prediction Accuracy
    hit_rate: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    mape: Optional[float]

    # Return Metrics
    annualized_return: Optional[float]
    annualized_volatility: Optional[float]
    cumulative_return: Optional[float]

    # Portfolio Metrics
    alpha: Optional[float]
    beta: Optional[float]

    # Distribution Stats
    skewness: Optional[float]
    kurtosis: Optional[float]
    winning_days: Optional[float]
    avg_win: Optional[float]
    avg_loss: Optional[float]
    win_loss_ratio: Optional[float]

    # Metadata
    n_observations: int
    period_start: Optional[str]
    period_end: Optional[str]
    risk_free_rate: float
```

**Methods:**
- `to_dict()` → Dict
- `to_json(filepath=None, indent=2)` → str
- `to_markdown(filepath=None)` → str

### Utility Functions

```python
generate_summary_table(
    report: MetricsReport,
    sections: Optional[List[str]] = None
) → str
```

Generates a formatted table of metrics. Sections can be: `'risk_adjusted'`, `'drawdown'`, `'var'`, `'returns'`, `'accuracy'`, `'portfolio'`

```python
compare_strategies(
    reports: List[MetricsReport],
    metrics_to_compare: Optional[List[str]] = None
) → pd.DataFrame
```

Compare multiple strategies side-by-side in a DataFrame.

## Examples

See the `examples/performance_metrics_demo.py` file for comprehensive demonstrations including:

1. Basic metric calculations
2. Comprehensive reports
3. Prediction accuracy analysis
4. Strategy comparison
5. Risk analysis across market conditions
6. Portfolio metrics (Alpha, Beta, IR)
7. Drawdown analysis
8. Report export (JSON, Markdown)

Run the demo:

```bash
python examples/performance_metrics_demo.py
```

## Metric Interpretations

### Sharpe Ratio
- **> 3.0**: Excellent
- **2.0 - 3.0**: Very Good
- **1.0 - 2.0**: Good
- **0.5 - 1.0**: Acceptable
- **< 0.5**: Poor

### Sortino Ratio
- Similar to Sharpe, but typically higher as it only penalizes downside volatility
- Preferred when upside volatility is desirable

### Maximum Drawdown
- Lower (more negative) is worse
- **> -10%**: Low risk
- **-10% to -20%**: Moderate risk
- **-20% to -30%**: High risk
- **< -30%**: Very high risk

### VaR (95%)
- Represents the worst expected loss 95% of the time
- Example: VaR of -3% means there's a 5% chance of losing more than 3%

### Beta
- **β = 1**: Moves in line with market
- **β > 1**: More volatile than market (amplifies movements)
- **β < 1**: Less volatile than market (dampens movements)
- **β < 0**: Moves opposite to market

### Alpha
- Positive alpha indicates outperformance after risk adjustment
- Negative alpha indicates underperformance

## Error Handling

The module includes comprehensive error handling:

- Empty arrays return `np.nan`
- Mismatched array lengths raise `ValueError`
- NaN values are automatically filtered
- Division by zero is handled gracefully
- All exceptions are logged

## Testing

Run the standalone test suite:

```bash
python test_performance_standalone.py
```

Run pytest tests (requires all dependencies):

```bash
pytest tests/test_performance_metrics.py -v
```

## Performance Considerations

- Calculations are optimized using NumPy vectorization
- Large datasets (>100k data points) are handled efficiently
- Memory usage scales linearly with data size
- All metrics are computed in O(n) or O(n log n) time

## Integration with Stock Markov Analysis

This module integrates seamlessly with the Markov chain simulation:

```python
from src.core.simulation.simulator import MonteCarloSimulator
from src.core.metrics.performance import PerformanceMetrics

# Run simulation
simulator = MonteCarloSimulator(n_simulations=1000, n_days=30)
result = simulator.run(initial_price, transition_matrix, returns_by_state)

# Analyze simulation results
metrics = PerformanceMetrics()

# Calculate metrics on simulated paths
for i, path in enumerate(result.paths[:10]):  # Analyze first 10 paths
    path_returns = np.diff(path) / path[:-1]
    report = metrics.calculate_all(returns=path_returns, ticker=f"Sim_{i}")
    print(f"Sharpe: {report.sharpe_ratio:.2f}")
```

## License

This module is part of the Stock Markov Analysis project.

## Contributing

When adding new metrics:

1. Add the calculation method to `PerformanceMetrics` class
2. Add corresponding field to `MetricsReport` dataclass
3. Update the `calculate_all()` method to include the new metric
4. Add tests in `tests/test_performance_metrics.py`
5. Update this README with usage examples

## Support

For issues or questions, please open an issue on the project repository.

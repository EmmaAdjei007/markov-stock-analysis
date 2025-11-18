"""
Integrated Example: Performance Metrics with Markov Chain Analysis

This example demonstrates how to use the performance metrics module
alongside the Markov chain simulation for comprehensive stock analysis.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load performance module directly to avoid dependency issues
spec = importlib.util.spec_from_file_location(
    "performance",
    Path(__file__).parent.parent / "src" / "core" / "metrics" / "performance.py"
)
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)

PerformanceMetrics = performance.PerformanceMetrics
MetricsReport = performance.MetricsReport
generate_summary_table = performance.generate_summary_table


def simulate_markov_returns(n_days=252, n_states=5):
    """
    Simulate returns using a simplified Markov chain approach.

    This is a simplified version for demonstration purposes.
    In production, use the full MarkovChain class from src.core.models.markov
    """
    np.random.seed(42)

    # Create transition matrix (simplified)
    transition_matrix = np.random.dirichlet(np.ones(n_states), n_states)

    # Define return distributions for each state
    state_means = np.linspace(-0.02, 0.02, n_states)  # -2% to +2%
    state_stds = np.linspace(0.01, 0.03, n_states)    # 1% to 3%

    returns = []
    current_state = np.random.randint(0, n_states)

    for _ in range(n_days):
        # Generate return based on current state
        ret = np.random.normal(state_means[current_state], state_stds[current_state])
        returns.append(ret)

        # Transition to next state
        current_state = np.random.choice(n_states, p=transition_matrix[current_state])

    return np.array(returns)


def example_1_basic_analysis():
    """Example 1: Basic performance analysis of simulated strategy."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Performance Analysis")
    print("=" * 80 + "\n")

    # Simulate returns
    returns = simulate_markov_returns(n_days=252)  # 1 year
    prices = (1 + returns).cumprod() * 100  # Start at 100

    # Initialize metrics
    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Calculate comprehensive report
    report = metrics.calculate_all(
        returns=returns,
        prices=prices,
        ticker="Markov Strategy"
    )

    # Display results
    print("Performance Summary:")
    print(f"  Ticker: {report.ticker}")
    print(f"  Observations: {report.n_observations}")
    print(f"  Sharpe Ratio: {report.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio: {report.sortino_ratio:.4f}")
    print(f"  Max Drawdown: {report.max_drawdown:.2f}%")
    print(f"  Annualized Return: {report.annualized_return:.2f}%")
    print(f"  Annualized Volatility: {report.annualized_volatility:.2f}%")
    print(f"  VaR (95%): {report.var_95:.2f}%")
    print(f"  CVaR (95%): {report.cvar_95:.2f}%")

    return report


def example_2_compare_states():
    """Example 2: Compare performance across different market states."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Compare Different Market States")
    print("=" * 80 + "\n")

    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Simulate different scenarios
    scenarios = {
        'Bull Market (5 states)': simulate_markov_returns(252, n_states=5),
        'Bear Market (3 states)': simulate_markov_returns(252, n_states=3) - 0.005,
        'Volatile Market (7 states)': simulate_markov_returns(252, n_states=7) * 1.5,
    }

    reports = []
    for name, returns in scenarios.items():
        report = metrics.calculate_all(returns=returns, ticker=name)
        reports.append(report)

        print(f"{name}:")
        print(f"  Sharpe: {report.sharpe_ratio:.4f}")
        print(f"  Max DD: {report.max_drawdown:.2f}%")
        print(f"  Ann. Return: {report.annualized_return:.2f}%")
        print(f"  Ann. Vol: {report.annualized_volatility:.2f}%")
        print()

    return reports


def example_3_simulation_ensemble():
    """Example 3: Analyze ensemble of Monte Carlo simulations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Monte Carlo Simulation Ensemble Analysis")
    print("=" * 80 + "\n")

    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Run multiple simulations
    n_simulations = 100
    n_days = 60  # ~3 months

    print(f"Running {n_simulations} simulations of {n_days} days each...")

    all_sharpes = []
    all_max_dds = []
    all_returns = []

    for i in range(n_simulations):
        np.random.seed(i)
        returns = simulate_markov_returns(n_days)
        prices = (1 + returns).cumprod() * 100

        sharpe = metrics.sharpe_ratio(returns)
        max_dd = metrics.max_drawdown(prices)
        ann_return = metrics.annualized_return(returns)

        all_sharpes.append(sharpe)
        all_max_dds.append(max_dd)
        all_returns.append(ann_return)

    # Aggregate statistics
    print(f"\nSimulation Results (n={n_simulations}):")
    print(f"  Sharpe Ratio:")
    print(f"    Mean: {np.mean(all_sharpes):.4f}")
    print(f"    Std: {np.std(all_sharpes):.4f}")
    print(f"    Range: [{np.min(all_sharpes):.4f}, {np.max(all_sharpes):.4f}]")

    print(f"\n  Maximum Drawdown:")
    print(f"    Mean: {np.mean(all_max_dds):.2f}%")
    print(f"    Std: {np.std(all_max_dds):.2f}%")
    print(f"    Range: [{np.min(all_max_dds):.2f}%, {np.max(all_max_dds):.2f}%]")

    print(f"\n  Annualized Return:")
    print(f"    Mean: {np.mean(all_returns):.2f}%")
    print(f"    Std: {np.std(all_returns):.2f}%")
    print(f"    Range: [{np.min(all_returns):.2f}%, {np.max(all_returns):.2f}%]")

    # Create report for average simulation
    avg_returns = simulate_markov_returns(n_days)
    avg_report = metrics.calculate_all(
        returns=avg_returns,
        ticker="Average Simulation"
    )

    return avg_report


def example_4_risk_analysis():
    """Example 4: Detailed risk analysis with VaR at multiple levels."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Comprehensive Risk Analysis")
    print("=" * 80 + "\n")

    # Simulate returns with fat tails (more extreme events)
    np.random.seed(42)
    normal_returns = np.random.normal(0.001, 0.02, 252)

    # Add some extreme events
    extreme_indices = np.random.choice(252, size=10, replace=False)
    normal_returns[extreme_indices] *= 3

    returns = normal_returns
    prices = (1 + returns).cumprod() * 100

    metrics = PerformanceMetrics(risk_free_rate=0.02)

    print("Risk Metrics Analysis:")
    print("\nValue at Risk (VaR):")
    for confidence in [0.90, 0.95, 0.99]:
        var = metrics.value_at_risk(returns, confidence)
        cvar = metrics.conditional_var(returns, confidence)
        print(f"  {confidence*100:.0f}% Confidence:")
        print(f"    VaR: {var:.2f}%")
        print(f"    CVaR: {cvar:.2f}%")
        print(f"    Difference: {cvar - var:.2f}%")

    print("\nDrawdown Analysis:")
    max_dd = metrics.max_drawdown(prices)
    dd_duration = metrics.max_drawdown_duration(prices)
    current_dd = metrics.current_drawdown(prices)

    print(f"  Maximum Drawdown: {max_dd:.2f}%")
    print(f"  Max DD Duration: {dd_duration} days")
    print(f"  Current Drawdown: {current_dd:.2f}%")

    # Calculate drawdown series
    running_max = np.maximum.accumulate(prices)
    drawdown_series = (prices - running_max) / running_max * 100
    underwater_days = np.sum(drawdown_series < 0)

    print(f"  Days Underwater: {underwater_days} ({underwater_days/len(prices)*100:.1f}%)")

    print("\nVolatility Analysis:")
    print(f"  Daily Volatility: {np.std(returns)*100:.2f}%")
    print(f"  Annualized Volatility: {metrics.annualized_volatility(returns):.2f}%")

    # Downside volatility
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(252) * 100
        print(f"  Downside Volatility: {downside_vol:.2f}%")


def example_5_prediction_backtest():
    """Example 5: Backtest with prediction accuracy metrics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Prediction Accuracy Analysis")
    print("=" * 80 + "\n")

    # Generate actual prices
    np.random.seed(42)
    actual_returns = simulate_markov_returns(100)
    actual_prices = (1 + actual_returns).cumprod() * 100

    # Generate predictions (simulate model predictions with some error)
    prediction_error = np.random.normal(0, 2, 100)
    predicted_prices = actual_prices + prediction_error

    metrics = PerformanceMetrics()

    # Calculate prediction metrics
    hit_rate = metrics.hit_rate(actual_prices, predicted_prices)
    rmse = metrics.rmse(actual_prices, predicted_prices)
    mae = metrics.mae(actual_prices, predicted_prices)
    mape = metrics.mape(actual_prices, predicted_prices)

    print("Prediction Accuracy Metrics:")
    print(f"  Hit Rate (Directional): {hit_rate:.2%}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Comprehensive report
    report = metrics.calculate_all(
        returns=actual_returns,
        prices=actual_prices,
        predicted_prices=predicted_prices,
        ticker="Backtest"
    )

    print("\nStrategy Performance (Backtest):")
    print(f"  Sharpe Ratio: {report.sharpe_ratio:.4f}")
    print(f"  Max Drawdown: {report.max_drawdown:.2f}%")
    print(f"  Win Rate: {report.winning_days:.1f}%")
    if report.win_loss_ratio:
        print(f"  Win/Loss Ratio: {report.win_loss_ratio:.2f}")

    return report


def example_6_export_reports():
    """Example 6: Export reports to various formats."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Export Performance Reports")
    print("=" * 80 + "\n")

    # Generate sample data and report
    returns = simulate_markov_returns(252)
    prices = (1 + returns).cumprod() * 100

    metrics = PerformanceMetrics(risk_free_rate=0.02)
    report = metrics.calculate_all(
        returns=returns,
        prices=prices,
        ticker="DEMO-STOCK"
    )

    # Create output directory
    output_dir = Path(__file__).parent.parent / "reports" / "performance_examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    json_path = output_dir / "performance_report.json"
    report.to_json(str(json_path))
    print(f"Exported JSON report to: {json_path}")

    # Export to Markdown
    md_path = output_dir / "performance_report.md"
    report.to_markdown(str(md_path))
    print(f"Exported Markdown report to: {md_path}")

    # Display markdown preview
    print("\nMarkdown Preview:")
    print("-" * 80)
    markdown = report.to_markdown()
    lines = markdown.split('\n')[:20]
    print('\n'.join(lines))
    print("...")
    print("-" * 80)


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("=" + " " * 10 + "INTEGRATED PERFORMANCE METRICS ANALYSIS EXAMPLES" + " " * 19 + "=")
    print("=" * 80)

    try:
        example_1_basic_analysis()
        example_2_compare_states()
        example_3_simulation_ensemble()
        example_4_risk_analysis()
        example_5_prediction_backtest()
        example_6_export_reports()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Performance metrics provide comprehensive strategy evaluation")
        print("  2. Multiple metrics give different perspectives on risk/return")
        print("  3. Ensemble analysis reveals strategy robustness")
        print("  4. Risk metrics (VaR, CVaR, DD) are crucial for risk management")
        print("  5. Prediction accuracy complements return-based metrics")
        print("  6. Reports can be exported for documentation and sharing")
        print("\nFor more details, see:")
        print("  - src/core/metrics/README.md")
        print("  - src/core/metrics/QUICK_REFERENCE.md")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

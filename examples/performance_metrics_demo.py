"""
Demonstration of the Performance Metrics module.

This script shows how to use the comprehensive performance metrics module
to analyze stock returns, calculate risk metrics, and generate reports.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.metrics.performance import (
    PerformanceMetrics,
    MetricsReport,
    generate_summary_table,
    compare_strategies
)


def generate_sample_data():
    """Generate sample stock data for demonstration."""
    np.random.seed(42)

    # Generate 2 years of daily returns
    n_days = 504  # ~2 years
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Strategy 1: Moderate risk, moderate return
    returns_1 = pd.Series(
        np.random.normal(0.0005, 0.015, n_days),  # 12.6% annual return, 24% vol
        index=dates,
        name='Strategy 1'
    )

    # Strategy 2: Higher risk, higher return
    returns_2 = pd.Series(
        np.random.normal(0.0008, 0.025, n_days),  # 20% annual return, 40% vol
        index=dates,
        name='Strategy 2'
    )

    # Benchmark: Market index
    benchmark = pd.Series(
        np.random.normal(0.0004, 0.012, n_days),  # 10% annual return, 19% vol
        index=dates,
        name='Benchmark'
    )

    # Calculate prices
    prices_1 = (1 + returns_1).cumprod() * 100
    prices_2 = (1 + returns_2).cumprod() * 100
    benchmark_prices = (1 + benchmark).cumprod() * 100

    return {
        'strategy_1': {'returns': returns_1, 'prices': prices_1},
        'strategy_2': {'returns': returns_2, 'prices': prices_2},
        'benchmark': {'returns': benchmark, 'prices': benchmark_prices}
    }


def demo_basic_metrics():
    """Demonstrate basic metric calculations."""
    print("=" * 80)
    print("DEMO 1: Basic Metric Calculations")
    print("=" * 80 + "\n")

    # Generate sample data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns

    # Create metrics calculator
    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Calculate individual metrics
    print("Individual Metric Calculations:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio(returns):.4f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio(returns):.4f}")
    print(f"  Annualized Return: {metrics.annualized_return(returns):.2f}%")
    print(f"  Annualized Volatility: {metrics.annualized_volatility(returns):.2f}%")

    # Calculate prices from returns
    prices = (1 + returns).cumprod() * 100

    print(f"  Maximum Drawdown: {metrics.max_drawdown(prices):.2f}%")
    print(f"  Max Drawdown Duration: {metrics.max_drawdown_duration(prices)} days")

    # Value at Risk
    print(f"  VaR (95%): {metrics.value_at_risk(returns, 0.95):.2f}%")
    print(f"  CVaR (95%): {metrics.conditional_var(returns, 0.95):.2f}%")

    print("\n")


def demo_comprehensive_report():
    """Demonstrate comprehensive metrics report."""
    print("=" * 80)
    print("DEMO 2: Comprehensive Metrics Report")
    print("=" * 80 + "\n")

    data = generate_sample_data()

    # Calculate metrics for Strategy 1
    metrics = PerformanceMetrics(risk_free_rate=0.02)

    report = metrics.calculate_all(
        returns=data['strategy_1']['returns'],
        prices=data['strategy_1']['prices'],
        benchmark_returns=data['benchmark']['returns'],
        ticker="Strategy 1"
    )

    # Display summary table
    print(generate_summary_table(report))

    print("\n")


def demo_prediction_accuracy():
    """Demonstrate prediction accuracy metrics."""
    print("=" * 80)
    print("DEMO 3: Prediction Accuracy Metrics")
    print("=" * 80 + "\n")

    # Generate actual and predicted prices
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
    actual_returns = np.random.normal(0.001, 0.02, 100)
    actual_prices = (1 + actual_returns).cumprod() * 100

    # Simulated predictions with some error
    predicted_prices = actual_prices + np.random.normal(0, 1.5, 100)

    # Calculate metrics
    metrics = PerformanceMetrics()

    report = metrics.calculate_all(
        returns=actual_returns,
        prices=actual_prices,
        predicted_prices=predicted_prices,
        ticker="AAPL"
    )

    print("Prediction Accuracy Metrics:")
    print(f"  Hit Rate: {report.hit_rate:.2%}")
    print(f"  RMSE: {report.rmse:.4f}")
    print(f"  MAE: {report.mae:.4f}")
    print(f"  MAPE: {report.mape:.2f}%")

    print("\n")


def demo_strategy_comparison():
    """Demonstrate comparing multiple strategies."""
    print("=" * 80)
    print("DEMO 4: Strategy Comparison")
    print("=" * 80 + "\n")

    data = generate_sample_data()

    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Calculate reports for both strategies
    report_1 = metrics.calculate_all(
        returns=data['strategy_1']['returns'],
        prices=data['strategy_1']['prices'],
        benchmark_returns=data['benchmark']['returns'],
        ticker="Moderate Risk"
    )

    report_2 = metrics.calculate_all(
        returns=data['strategy_2']['returns'],
        prices=data['strategy_2']['prices'],
        benchmark_returns=data['benchmark']['returns'],
        ticker="Aggressive"
    )

    benchmark_report = metrics.calculate_all(
        returns=data['benchmark']['returns'],
        prices=data['benchmark']['prices'],
        ticker="Benchmark"
    )

    # Compare strategies
    comparison = compare_strategies([report_1, report_2, benchmark_report])

    print("Strategy Comparison:")
    print(comparison.to_string())

    print("\n")


def demo_risk_analysis():
    """Demonstrate risk analysis across different market conditions."""
    print("=" * 80)
    print("DEMO 5: Risk Analysis - Different Market Conditions")
    print("=" * 80 + "\n")

    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Simulate different market conditions
    scenarios = {
        'Bull Market': np.random.normal(0.002, 0.015, 252),
        'Bear Market': np.random.normal(-0.001, 0.025, 252),
        'Volatile Market': np.random.normal(0.0005, 0.035, 252),
        'Stable Market': np.random.normal(0.0008, 0.008, 252)
    }

    results = []

    for scenario_name, returns in scenarios.items():
        prices = (1 + returns).cumprod() * 100

        print(f"{scenario_name}:")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio(returns):.4f}")
        print(f"  Max Drawdown: {metrics.max_drawdown(prices):.2f}%")
        print(f"  VaR (95%): {metrics.value_at_risk(returns, 0.95):.2f}%")
        print(f"  CVaR (95%): {metrics.conditional_var(returns, 0.95):.2f}%")
        print(f"  Annualized Return: {metrics.annualized_return(returns):.2f}%")
        print(f"  Annualized Volatility: {metrics.annualized_volatility(returns):.2f}%")
        print()


def demo_export_reports():
    """Demonstrate exporting reports to different formats."""
    print("=" * 80)
    print("DEMO 6: Exporting Reports")
    print("=" * 80 + "\n")

    # Generate sample data and report
    data = generate_sample_data()
    metrics = PerformanceMetrics(risk_free_rate=0.02)

    report = metrics.calculate_all(
        returns=data['strategy_1']['returns'],
        prices=data['strategy_1']['prices'],
        benchmark_returns=data['benchmark']['returns'],
        ticker="AAPL"
    )

    # Create output directory
    output_dir = Path(__file__).parent.parent / "reports" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    json_path = output_dir / "performance_metrics.json"
    report.to_json(str(json_path))
    print(f"Exported JSON report to: {json_path}")

    # Export to Markdown
    md_path = output_dir / "performance_metrics.md"
    report.to_markdown(str(md_path))
    print(f"Exported Markdown report to: {md_path}")

    # Display markdown preview
    print("\nMarkdown Preview (first 500 chars):")
    print("-" * 80)
    markdown = report.to_markdown()
    print(markdown[:500] + "...")
    print("-" * 80)

    print("\n")


def demo_portfolio_metrics():
    """Demonstrate portfolio-specific metrics (Alpha, Beta, IR)."""
    print("=" * 80)
    print("DEMO 7: Portfolio Metrics (Alpha, Beta, Information Ratio)")
    print("=" * 80 + "\n")

    data = generate_sample_data()
    metrics = PerformanceMetrics(risk_free_rate=0.02)

    # Calculate alpha and beta
    alpha, beta = metrics.alpha_beta(
        data['strategy_1']['returns'],
        data['benchmark']['returns']
    )

    print(f"Strategy 1 vs Benchmark:")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  Beta: {beta:.4f}")

    # Information Ratio
    ir = metrics.information_ratio(
        data['strategy_1']['returns'],
        data['benchmark']['returns']
    )

    print(f"  Information Ratio: {ir:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if beta > 1:
        print(f"  - Beta > 1: Strategy is more volatile than benchmark ({beta:.2f}x)")
    elif beta < 1:
        print(f"  - Beta < 1: Strategy is less volatile than benchmark ({beta:.2f}x)")
    else:
        print(f"  - Beta ≈ 1: Strategy has similar volatility to benchmark")

    if alpha > 0:
        print(f"  - Positive Alpha: Strategy outperforms after adjusting for risk")
    else:
        print(f"  - Negative Alpha: Strategy underperforms after adjusting for risk")

    print("\n")


def demo_drawdown_analysis():
    """Demonstrate detailed drawdown analysis."""
    print("=" * 80)
    print("DEMO 8: Detailed Drawdown Analysis")
    print("=" * 80 + "\n")

    # Generate data with specific drawdown pattern
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    # Introduce a significant drawdown
    returns[100:150] = np.random.normal(-0.005, 0.03, 50)  # Bad period

    prices = (1 + returns).cumprod() * 100

    metrics = PerformanceMetrics()

    print("Drawdown Analysis:")
    print(f"  Maximum Drawdown: {metrics.max_drawdown(prices):.2f}%")
    print(f"  Max DD Duration: {metrics.max_drawdown_duration(prices)} days")
    print(f"  Current Drawdown: {metrics.current_drawdown(prices):.2f}%")

    # Calculate drawdown series
    running_max = np.maximum.accumulate(prices)
    drawdown_series = (prices - running_max) / running_max * 100

    # Find underwater periods
    underwater_days = np.sum(drawdown_series < 0)
    print(f"  Days Underwater: {underwater_days} ({underwater_days/len(prices)*100:.1f}%)")

    # Recovery analysis
    in_recovery = drawdown_series[-1] < 0
    if in_recovery:
        print(f"  Status: Currently in drawdown")
    else:
        print(f"  Status: At or near peak")

    print("\n")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "PERFORMANCE METRICS MODULE DEMONSTRATION" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    try:
        demo_basic_metrics()
        demo_comprehensive_report()
        demo_prediction_accuracy()
        demo_strategy_comparison()
        demo_risk_analysis()
        demo_portfolio_metrics()
        demo_drawdown_analysis()
        demo_export_reports()

        print("=" * 80)
        print("All demonstrations completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

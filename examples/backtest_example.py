"""
Example script demonstrating the backtesting framework.

This script shows how to use the advanced backtesting capabilities
with MarkovChain and HigherOrderMarkov models.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models.markov import MarkovChain, HigherOrderMarkov, EnsembleMarkov
from src.core.metrics.backtesting import (
    Backtester,
    BacktestResult,
    plot_backtest_results
)
from src.data_preprocessor import DataPreprocessor


def generate_synthetic_stock_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic stock price data for testing.

    Args:
        n_samples: Number of data points
        seed: Random seed

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)

    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Generate price series with trend and volatility
    base_price = 100.0
    returns = np.random.randn(n_samples) * 0.02 + 0.0005  # 2% daily vol, slight upward drift
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    high = prices * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    low = prices * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    open_prices = prices + np.random.randn(n_samples) * 0.5
    close_prices = prices

    # Generate volume
    volume = np.random.randint(1000000, 5000000, n_samples)

    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close_prices,
        'Volume': volume
    })

    return df


def state_to_price(state: int, base_price: float = 100.0, n_states: int = 5) -> float:
    """
    Convert state back to approximate price.

    This is a simple linear mapping for demonstration purposes.

    Args:
        state: State label (0 to n_states-1)
        base_price: Base price level
        n_states: Total number of states

    Returns:
        Approximate price
    """
    # Map state to percentage change: -10% to +10%
    pct_change = (state - (n_states // 2)) / n_states * 0.2
    return base_price * (1 + pct_change)


def example_1_basic_backtest():
    """Example 1: Basic single train/test split backtest."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Train/Test Split Backtest")
    print("="*70)

    # Generate data
    df = generate_synthetic_stock_data(1000)

    # Discretize returns into states
    preprocessor = DataPreprocessor(df)
    returns = preprocessor.calculate_returns()

    # Create states (5-state discretization)
    states = pd.qcut(returns.dropna(), q=5, labels=False, duplicates='drop')
    states = states.values

    # Create model
    model = MarkovChain(n_states=5, alpha=0.001)

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.7,
        min_train_size=100
    )

    # Run backtest
    result = backtester.run_single_backtest(
        states=states,
        prices=df['Close'].values[1:len(states)+1]  # Align with returns
    )

    # Print results
    print(f"\nTrain size: {result.metadata['train_size']}")
    print(f"Test size: {result.metadata['test_size']}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {result.overall_metrics['accuracy']:.4f}")
    print(f"  Precision: {result.overall_metrics['precision']:.4f}")
    print(f"  Recall:    {result.overall_metrics['recall']:.4f}")
    print(f"  F1 Score:  {result.overall_metrics['f1_score']:.4f}")

    if result.overall_metrics.get('direction_accuracy'):
        print(f"  Direction Accuracy: {result.overall_metrics['direction_accuracy']:.4f}")

    return result


def example_2_walk_forward():
    """Example 2: Walk-forward analysis with rolling windows."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Walk-Forward Analysis")
    print("="*70)

    # Generate data
    df = generate_synthetic_stock_data(1500)

    # Discretize
    preprocessor = DataPreprocessor(df)
    returns = preprocessor.calculate_returns()
    states = pd.qcut(returns.dropna(), q=5, labels=False, duplicates='drop').values

    # Create second-order model for more sophistication
    model = HigherOrderMarkov(n_states=5, order=2, alpha=0.001)

    # Create backtester with rolling window
    backtester = Backtester(
        model=model,
        train_size=0.5,
        window_size=100,
        step_size=50,
        expanding_window=False,  # Rolling window
        retrain_interval=2  # Retrain every 2 windows
    )

    # Run walk-forward backtest
    result = backtester.run_walk_forward(
        states=states,
        prices=df['Close'].values[1:len(states)+1],
        verbose=True
    )

    # Print summary
    print(f"\n{result.metadata['n_windows']} windows completed")
    print(f"\nOverall Performance:")
    print(result.summary())

    print(f"\nTiming:")
    print(f"  Total train time: {result.metadata['total_train_time']:.2f}s")
    print(f"  Total predict time: {result.metadata['total_predict_time']:.2f}s")

    return result


def example_3_cross_validation():
    """Example 3: Time-series cross-validation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Time-Series Cross-Validation")
    print("="*70)

    # Generate data
    df = generate_synthetic_stock_data(1200)

    # Discretize
    preprocessor = DataPreprocessor(df)
    returns = preprocessor.calculate_returns()
    states = pd.qcut(returns.dropna(), q=5, labels=False, duplicates='drop').values

    # Create ensemble model
    model = EnsembleMarkov(
        n_states=5,
        orders=[1, 2],
        weights=[0.3, 0.7],  # Give more weight to second-order
        alpha=0.001
    )

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.6,
        min_train_size=200
    )

    # Run CV
    result = backtester.run_time_series_cv(
        states=states,
        n_splits=5,
        prices=df['Close'].values[1:len(states)+1],
        verbose=True
    )

    # Print results
    print(f"\nCross-Validation Results ({len(result.windows)} folds):")
    print(result.summary())

    # Show per-fold results
    print(f"\nPer-Fold Accuracy:")
    for window in result.windows:
        print(f"  Fold {window.window_id}: {window.accuracy:.4f}")

    return result


def example_4_model_comparison():
    """Example 4: Compare multiple models."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Model Comparison")
    print("="*70)

    # Generate data
    df = generate_synthetic_stock_data(1000)

    # Discretize
    preprocessor = DataPreprocessor(df)
    returns = preprocessor.calculate_returns()
    states = pd.qcut(returns.dropna(), q=5, labels=False, duplicates='drop').values

    # Create multiple models
    models = [
        MarkovChain(n_states=5, alpha=0.001),
        HigherOrderMarkov(n_states=5, order=2, alpha=0.001),
        EnsembleMarkov(n_states=5, orders=[1, 2], alpha=0.001)
    ]

    model_names = [
        'First-Order Markov',
        'Second-Order Markov',
        'Ensemble (1st + 2nd)'
    ]

    # Create backtester
    backtester = Backtester(
        model=models[0],  # Placeholder
        train_size=0.7,
        window_size=80,
        step_size=40
    )

    # Compare models
    comparison = backtester.compare_models(
        models=models,
        model_names=model_names,
        states=states,
        prices=df['Close'].values[1:len(states)+1],
        method='walk_forward',
        verbose=False
    )

    print("\nModel Comparison Results:")
    print(comparison[['accuracy', 'precision', 'recall', 'f1_score']])

    # Find best model
    best_model = comparison['accuracy'].idxmax()
    print(f"\nBest Model: {best_model}")
    print(f"Accuracy: {comparison.loc[best_model, 'accuracy']:.4f}")

    return comparison


def example_5_statistical_testing():
    """Example 5: Statistical significance testing."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Statistical Significance Testing")
    print("="*70)

    # Generate data
    df = generate_synthetic_stock_data(1000)

    # Discretize
    preprocessor = DataPreprocessor(df)
    returns = preprocessor.calculate_returns()
    states = pd.qcut(returns.dropna(), q=5, labels=False, duplicates='drop').values

    # Create two models
    model1 = MarkovChain(n_states=5, alpha=0.001)
    model2 = HigherOrderMarkov(n_states=5, order=2, alpha=0.001)

    # Create backtester
    backtester1 = Backtester(
        model=model1,
        train_size=0.6,
        window_size=80,
        step_size=40
    )

    backtester2 = Backtester(
        model=model2,
        train_size=0.6,
        window_size=80,
        step_size=40
    )

    # Run backtests
    result1 = backtester1.run_walk_forward(states, verbose=False)
    result2 = backtester2.run_walk_forward(states, verbose=False)

    # Statistical test
    test_result = backtester1.statistical_significance_test(
        result1, result2,
        metric='accuracy',
        alpha=0.05
    )

    # Print results
    print(f"\nModel 1 (First-Order) Accuracy: {test_result['mean_1']:.4f}")
    print(f"Model 2 (Second-Order) Accuracy: {test_result['mean_2']:.4f}")
    print(f"Mean Difference: {test_result['mean_diff']:.4f}")
    print(f"\nStatistical Test:")
    print(f"  t-statistic: {test_result['t_statistic']:.4f}")
    print(f"  p-value: {test_result['p_value']:.4f}")
    print(f"  Significant (α=0.05): {test_result['significant']}")
    print(f"  Effect Size (Cohen's d): {test_result['cohens_d']:.4f}")
    print(f"  Interpretation: {test_result['interpretation']}")

    if test_result['significant']:
        winner = 'Model 1' if test_result['winner'] == 1 else 'Model 2'
        print(f"\n✓ {winner} performs significantly better!")
    else:
        print("\n○ No significant difference between models")

    return test_result


def example_6_expanding_window():
    """Example 6: Expanding window backtest."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Expanding Window Backtest")
    print("="*70)

    # Generate data
    df = generate_synthetic_stock_data(1200)

    # Discretize
    preprocessor = DataPreprocessor(df)
    returns = preprocessor.calculate_returns()
    states = pd.qcut(returns.dropna(), q=5, labels=False, duplicates='drop').values

    # Create model
    model = MarkovChain(n_states=5, alpha=0.001)

    # Create backtester with expanding window
    backtester = Backtester(
        model=model,
        train_size=0.5,
        window_size=100,
        step_size=50,
        expanding_window=True,  # Key difference: expanding window
        retrain_interval=1
    )

    # Run backtest
    result = backtester.run_walk_forward(
        states=states,
        verbose=True
    )

    print(f"\nExpanding Window Results:")
    print(f"  Windows: {len(result.windows)}")
    print(f"  Mean Accuracy: {result.overall_metrics['accuracy']:.4f} ± {result.overall_metrics['accuracy_std']:.4f}")

    # Show how training set grows
    print(f"\nTraining Set Growth:")
    for i, window in enumerate(result.windows[:5]):  # First 5 windows
        train_size = window.train_end - window.train_start
        test_size = window.test_end - window.test_start
        print(f"  Window {i}: train={train_size}, test={test_size}, accuracy={window.accuracy:.4f}")

    return result


def main():
    """Run all examples."""
    print("\n")
    print("="*70)
    print(" BACKTESTING FRAMEWORK EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates various backtesting capabilities:")
    print("  1. Basic train/test split")
    print("  2. Walk-forward analysis with rolling windows")
    print("  3. Time-series cross-validation")
    print("  4. Model comparison")
    print("  5. Statistical significance testing")
    print("  6. Expanding window backtest")

    # Run examples
    try:
        result1 = example_1_basic_backtest()
        result2 = example_2_walk_forward()
        result3 = example_3_cross_validation()
        result4 = example_4_model_comparison()
        result5 = example_5_statistical_testing()
        result6 = example_6_expanding_window()

        print("\n" + "="*70)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)

        # Optionally plot results from example 2
        try:
            print("\nGenerating visualization for walk-forward results...")
            plot_backtest_results(result2, save_path='backtest_results.png')
            print("Visualization saved to: backtest_results.png")
        except Exception as e:
            print(f"Visualization skipped: {e}")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

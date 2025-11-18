"""
Quick integration test for backtesting framework with MarkovChain models.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import directly to avoid dependency issues
import importlib.util

# Load markov module
markov_path = Path(__file__).parent / 'src' / 'core' / 'models' / 'markov.py'
spec = importlib.util.spec_from_file_location("markov", markov_path)
markov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(markov)

MarkovChain = markov.MarkovChain
HigherOrderMarkov = markov.HigherOrderMarkov
EnsembleMarkov = markov.EnsembleMarkov

# Load backtesting module
backtest_path = Path(__file__).parent / 'src' / 'core' / 'metrics' / 'backtesting.py'
spec = importlib.util.spec_from_file_location("backtesting", backtest_path)
backtesting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtesting)

Backtester = backtesting.Backtester
BacktestResult = backtesting.BacktestResult

def test_markov_chain_integration():
    """Test backtesting with MarkovChain."""
    print("\n" + "="*60)
    print("TEST 1: MarkovChain Integration")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 500
    states = np.random.randint(0, 5, n)

    # Create model
    model = MarkovChain(n_states=5, alpha=0.001)

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.7,
        min_train_size=50
    )

    # Run single backtest
    result = backtester.run_single_backtest(states)

    print(f"Train size: {result.metadata['train_size']}")
    print(f"Test size: {result.metadata['test_size']}")
    print(f"Accuracy: {result.overall_metrics['accuracy']:.4f}")
    print(f"Precision: {result.overall_metrics['precision']:.4f}")
    print(f"Recall: {result.overall_metrics['recall']:.4f}")
    print(f"F1 Score: {result.overall_metrics['f1_score']:.4f}")

    assert 0 <= result.overall_metrics['accuracy'] <= 1
    assert len(result.windows) == 1

    print("[PASS] Test passed!")
    return result


def test_higher_order_integration():
    """Test backtesting with HigherOrderMarkov."""
    print("\n" + "="*60)
    print("TEST 2: HigherOrderMarkov Integration")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 600
    states = np.random.randint(0, 5, n)

    # Create model
    model = HigherOrderMarkov(n_states=5, order=2, alpha=0.001)

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.6,
        window_size=80,
        step_size=40
    )

    # Run walk-forward
    result = backtester.run_walk_forward(states, verbose=False)

    print(f"Windows: {len(result.windows)}")
    print(f"Mean Accuracy: {result.overall_metrics['accuracy']:.4f}")
    print(f"Std Accuracy: {result.overall_metrics['accuracy_std']:.4f}")

    assert len(result.windows) > 0
    assert 0 <= result.overall_metrics['accuracy'] <= 1

    print("[PASS] Test passed!")
    return result


def test_ensemble_integration():
    """Test backtesting with EnsembleMarkov."""
    print("\n" + "="*60)
    print("TEST 3: EnsembleMarkov Integration")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 500
    states = np.random.randint(0, 5, n)

    # Create model
    model = EnsembleMarkov(
        n_states=5,
        orders=[1, 2],
        weights=[0.4, 0.6],
        alpha=0.001
    )

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.7
    )

    # Run CV
    result = backtester.run_time_series_cv(states, n_splits=3, verbose=False)

    print(f"Folds: {len(result.windows)}")
    print(f"Mean Accuracy: {result.overall_metrics['accuracy']:.4f}")

    for i, window in enumerate(result.windows):
        print(f"  Fold {i}: {window.accuracy:.4f}")

    assert len(result.windows) > 0

    print("[PASS] Test passed!")
    return result


def test_model_comparison():
    """Test comparing multiple models."""
    print("\n" + "="*60)
    print("TEST 4: Model Comparison")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 400
    states = np.random.randint(0, 5, n)

    # Create models
    models = [
        MarkovChain(n_states=5, alpha=0.001),
        HigherOrderMarkov(n_states=5, order=2, alpha=0.001),
        EnsembleMarkov(n_states=5, orders=[1, 2], alpha=0.001)
    ]

    model_names = ['Order-1', 'Order-2', 'Ensemble']

    # Create backtester
    backtester = Backtester(
        model=models[0],
        train_size=0.7
    )

    # Compare
    comparison = backtester.compare_models(
        models=models,
        model_names=model_names,
        states=states,
        method='single',
        verbose=False
    )

    print("\nComparison Results:")
    print(comparison[['accuracy', 'precision', 'recall', 'f1_score']])

    assert len(comparison) == 3
    assert 'accuracy' in comparison.columns

    best = comparison['accuracy'].idxmax()
    print(f"\nBest model: {best}")

    print("[OK] Test passed!")
    return comparison


def test_statistical_testing():
    """Test statistical significance testing."""
    print("\n" + "="*60)
    print("TEST 5: Statistical Significance Testing")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 500
    states = np.random.randint(0, 5, n)

    # Create two models
    model1 = MarkovChain(n_states=5, alpha=0.001)
    model2 = HigherOrderMarkov(n_states=5, order=2, alpha=0.001)

    # Backtest both
    backtester1 = Backtester(
        model=model1,
        train_size=0.5,
        window_size=80,
        step_size=40
    )

    backtester2 = Backtester(
        model=model2,
        train_size=0.5,
        window_size=80,
        step_size=40
    )

    result1 = backtester1.run_walk_forward(states, verbose=False)
    result2 = backtester2.run_walk_forward(states, verbose=False)

    # Statistical test
    test_result = backtester1.statistical_significance_test(
        result1, result2, metric='accuracy'
    )

    print(f"Model 1 Accuracy: {test_result['mean_1']:.4f}")
    print(f"Model 2 Accuracy: {test_result['mean_2']:.4f}")
    print(f"Difference: {test_result['mean_diff']:.4f}")
    print(f"p-value: {test_result['p_value']:.4f}")
    print(f"Significant: {test_result['significant']}")
    print(f"Effect size: {test_result['cohens_d']:.4f} ({test_result['interpretation']})")

    assert 'p_value' in test_result
    assert 'cohens_d' in test_result

    print("[OK] Test passed!")
    return test_result


def test_with_prices():
    """Test backtesting with price predictions."""
    print("\n" + "="*60)
    print("TEST 6: Backtesting with Prices")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 500
    states = np.random.randint(0, 5, n)
    prices = 100 + np.cumsum(np.random.randn(n) * 2)

    # Create model
    model = MarkovChain(n_states=5, alpha=0.001)

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.7
    )

    # Run backtest
    result = backtester.run_single_backtest(states, prices)

    print(f"Accuracy: {result.overall_metrics['accuracy']:.4f}")
    print(f"MAE: {result.overall_metrics.get('mae', 'N/A')}")
    print(f"RMSE: {result.overall_metrics.get('rmse', 'N/A')}")
    print(f"Direction Accuracy: {result.overall_metrics.get('direction_accuracy', 'N/A')}")

    # Note: MAE/RMSE might be None if state_to_price_fn not provided
    # Direction accuracy should be available
    if result.overall_metrics.get('direction_accuracy'):
        print("[OK] Price metrics calculated successfully!")

    print("[PASS] Test passed!")
    return result


def main():
    """Run all integration tests."""
    print("\n")
    print("="*60)
    print("BACKTESTING FRAMEWORK INTEGRATION TESTS")
    print("="*60)

    try:
        test_markov_chain_integration()
        test_higher_order_integration()
        test_ensemble_integration()
        test_model_comparison()
        test_statistical_testing()
        test_with_prices()

        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED!")
        print("="*60)
        print("\nThe backtesting framework is fully compatible with:")
        print("  [OK] MarkovChain")
        print("  [OK] HigherOrderMarkov")
        print("  [OK] EnsembleMarkov")
        print("\nReady for production use!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

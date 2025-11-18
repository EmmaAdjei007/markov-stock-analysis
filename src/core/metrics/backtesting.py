"""
Advanced backtesting framework for Markov chain models.

This module provides comprehensive backtesting capabilities including:
- Walk-forward analysis with rolling windows
- Time-series cross-validation
- Out-of-sample validation
- Model retraining at specified intervals
- Statistical significance testing
- Performance metric tracking

Example:
    >>> from src.core.models.markov import MarkovChain
    >>> from src.core.metrics.backtesting import Backtester
    >>>
    >>> model = MarkovChain(n_states=5, alpha=0.001)
    >>> backtester = Backtester(model, train_size=0.7, window_size=100)
    >>> results = backtester.run_walk_forward(prices, states)
    >>> backtester.plot_results(results)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from scipy import stats
from collections import defaultdict
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class BacktestWindow:
    """Results from a single backtesting window."""

    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    # Predictions
    predicted_states: np.ndarray
    actual_states: np.ndarray
    predicted_prices: Optional[np.ndarray] = None
    actual_prices: Optional[np.ndarray] = None

    # Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Regression metrics (if prices provided)
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    direction_accuracy: Optional[float] = None

    # Model metadata
    model_params: Dict[str, Any] = field(default_factory=dict)
    train_time: float = 0.0
    predict_time: float = 0.0


@dataclass
class BacktestResult:
    """
    Comprehensive results from backtesting analysis.

    Attributes:
        windows: List of individual window results
        overall_metrics: Aggregated metrics across all windows
        statistical_tests: Results from significance tests
        metadata: Additional information about the backtest
    """

    windows: List[BacktestWindow]
    overall_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric_series(self, metric_name: str) -> np.ndarray:
        """
        Extract a specific metric across all windows.

        Args:
            metric_name: Name of the metric to extract

        Returns:
            Array of metric values
        """
        values = []
        for window in self.windows:
            if hasattr(window, metric_name):
                value = getattr(window, metric_name)
                if value is not None:
                    values.append(value)
        return np.array(values)

    def summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all metrics.

        Returns:
            DataFrame with metric statistics
        """
        metrics = {}

        # Extract all numeric metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score',
                      'mae', 'rmse', 'mape', 'direction_accuracy']:
            series = self.get_metric_series(metric)
            if len(series) > 0:
                metrics[metric] = {
                    'mean': np.mean(series),
                    'std': np.std(series),
                    'min': np.min(series),
                    'max': np.max(series),
                    'median': np.median(series)
                }

        return pd.DataFrame(metrics).T

    def to_dict(self) -> Dict:
        """Convert results to dictionary for serialization."""
        return {
            'n_windows': len(self.windows),
            'overall_metrics': self.overall_metrics,
            'statistical_tests': self.statistical_tests,
            'metadata': self.metadata,
            'window_metrics': [
                {
                    'window_id': w.window_id,
                    'accuracy': w.accuracy,
                    'mae': w.mae,
                    'rmse': w.rmse,
                    'direction_accuracy': w.direction_accuracy
                }
                for w in self.windows
            ]
        }


class Backtester:
    """
    Advanced backtesting framework for time series models.

    Features:
        - Walk-forward analysis with rolling/expanding windows
        - Time-series cross-validation
        - Out-of-sample validation
        - Automatic model retraining
        - Comprehensive performance metrics
        - Statistical significance testing

    Example:
        >>> model = MarkovChain(n_states=5)
        >>> backtester = Backtester(
        ...     model=model,
        ...     train_size=0.7,
        ...     window_size=100,
        ...     step_size=20
        ... )
        >>> results = backtester.run_walk_forward(prices, states)
        >>> print(results.summary())
    """

    def __init__(
        self,
        model: Any,
        train_size: float = 0.7,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        expanding_window: bool = False,
        retrain_interval: int = 1,
        min_train_size: int = 50,
        random_state: Optional[int] = None
    ):
        """
        Initialize backtester.

        Args:
            model: Markov model instance (MarkovChain, HigherOrderMarkov, etc.)
            train_size: Initial training set size (fraction or absolute count)
            window_size: Size of rolling window (None for expanding window)
            step_size: Number of steps to move forward (default: 10% of window)
            expanding_window: If True, use expanding window instead of rolling
            retrain_interval: Retrain model every N windows
            min_train_size: Minimum training samples required
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.train_size = train_size
        self.window_size = window_size
        self.step_size = step_size
        self.expanding_window = expanding_window
        self.retrain_interval = retrain_interval
        self.min_train_size = min_train_size
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Validate parameters
        if train_size <= 0 or train_size >= 1:
            if not isinstance(train_size, int) or train_size < min_train_size:
                raise ValueError(f"train_size must be in (0, 1) or >= {min_train_size}")

    def train_test_split(
        self,
        data: np.ndarray,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets (time-series aware).

        Args:
            data: Input array
            shuffle: Whether to shuffle (generally False for time series)

        Returns:
            Tuple of (train_data, test_data)
        """
        n = len(data)

        if 0 < self.train_size < 1:
            train_end = int(n * self.train_size)
        else:
            train_end = int(self.train_size)

        train_end = max(train_end, self.min_train_size)

        if shuffle:
            logger.warning("Shuffling time series data - this may break temporal dependencies")
            indices = np.random.permutation(n)
            train_idx = indices[:train_end]
            test_idx = indices[train_end:]
            return data[train_idx], data[test_idx]
        else:
            return data[:train_end], data[train_end:]

    def _calculate_window_metrics(
        self,
        window: BacktestWindow,
        predicted_states: np.ndarray,
        actual_states: np.ndarray,
        predicted_prices: Optional[np.ndarray] = None,
        actual_prices: Optional[np.ndarray] = None
    ) -> None:
        """
        Calculate performance metrics for a window.

        Args:
            window: BacktestWindow to populate
            predicted_states: Predicted state sequence
            actual_states: Actual state sequence
            predicted_prices: Predicted prices (optional)
            actual_prices: Actual prices (optional)
        """
        # Classification metrics
        window.accuracy = np.mean(predicted_states == actual_states)

        # Per-class precision/recall/f1
        n_states = max(np.max(predicted_states), np.max(actual_states)) + 1

        precisions = []
        recalls = []
        f1_scores = []

        for state in range(n_states):
            # True positives
            tp = np.sum((predicted_states == state) & (actual_states == state))
            # False positives
            fp = np.sum((predicted_states == state) & (actual_states != state))
            # False negatives
            fn = np.sum((predicted_states != state) & (actual_states == state))

            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)

            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)

            # F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        window.precision = np.mean(precisions)
        window.recall = np.mean(recalls)
        window.f1_score = np.mean(f1_scores)

        # Regression metrics (if prices provided)
        if predicted_prices is not None and actual_prices is not None:
            # MAE
            window.mae = np.mean(np.abs(predicted_prices - actual_prices))

            # RMSE
            window.rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))

            # MAPE (avoid division by zero)
            mask = actual_prices != 0
            if np.any(mask):
                window.mape = np.mean(np.abs((actual_prices[mask] - predicted_prices[mask]) / actual_prices[mask])) * 100

            # Direction accuracy
            if len(actual_prices) > 1:
                actual_direction = np.sign(np.diff(actual_prices))
                predicted_direction = np.sign(np.diff(predicted_prices))
                window.direction_accuracy = np.mean(actual_direction == predicted_direction)

    def _get_prediction(
        self,
        history: Union[int, List[int]],
        n_steps: int = 1
    ) -> int:
        """
        Get prediction from model based on its type.

        Args:
            history: Current state or state history
            n_steps: Number of steps ahead

        Returns:
            Predicted state
        """
        # Check model type and call appropriate method
        model_class = self.model.__class__.__name__

        if model_class == 'MarkovChain':
            # First-order model
            current_state = history if isinstance(history, int) else history[-1]
            probs = self.model.predict(current_state, n_steps=n_steps)
            return int(np.argmax(probs))

        elif model_class in ['HigherOrderMarkov', 'EnsembleMarkov']:
            # Higher-order models need history
            if isinstance(history, int):
                history = [history]
            probs = self.model.predict(history)
            return int(np.argmax(probs))

        else:
            # Generic fallback
            try:
                if hasattr(self.model, 'predict'):
                    if isinstance(history, int):
                        probs = self.model.predict(history, n_steps=n_steps)
                    else:
                        probs = self.model.predict(history)
                    return int(np.argmax(probs))
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise

    def run_single_backtest(
        self,
        states: np.ndarray,
        prices: Optional[np.ndarray] = None,
        state_to_price_fn: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Run a single train/test split backtest.

        Args:
            states: State sequence
            prices: Price sequence (optional)
            state_to_price_fn: Function to convert states to prices

        Returns:
            BacktestResult with single window
        """
        states = np.asarray(states)
        if prices is not None:
            prices = np.asarray(prices)
            if len(prices) != len(states):
                raise ValueError("States and prices must have same length")

        # Split data
        train_states, test_states = self.train_test_split(states)

        if prices is not None:
            train_prices, test_prices = self.train_test_split(prices)
        else:
            test_prices = None

        # Train model
        start_time = datetime.now()
        self.model.fit(train_states)
        train_time = (datetime.now() - start_time).total_seconds()

        # Make predictions
        start_time = datetime.now()
        predicted_states = []

        # Get model order
        order = getattr(self.model, 'order', 1)

        # Initialize history
        if order == 1:
            history = int(train_states[-1])
        else:
            history = list(train_states[-order:])

        for i in range(len(test_states)):
            pred = self._get_prediction(history)
            predicted_states.append(pred)

            # Update history with actual state for next prediction
            if order == 1:
                history = int(test_states[i])
            else:
                history = list(history[1:]) + [int(test_states[i])]

        predicted_states = np.array(predicted_states)
        predict_time = (datetime.now() - start_time).total_seconds()

        # Convert to prices if needed
        predicted_prices = None
        if state_to_price_fn is not None:
            predicted_prices = np.array([state_to_price_fn(s) for s in predicted_states])

        # Create window
        window = BacktestWindow(
            window_id=0,
            train_start=0,
            train_end=len(train_states),
            test_start=len(train_states),
            test_end=len(states),
            predicted_states=predicted_states,
            actual_states=test_states,
            predicted_prices=predicted_prices,
            actual_prices=test_prices,
            model_params={
                'n_states': getattr(self.model, 'n_states', None),
                'order': order,
                'alpha': getattr(self.model, 'alpha', None)
            },
            train_time=train_time,
            predict_time=predict_time
        )

        # Calculate metrics
        self._calculate_window_metrics(
            window,
            predicted_states,
            test_states,
            predicted_prices,
            test_prices
        )

        # Create result
        result = BacktestResult(
            windows=[window],
            overall_metrics={
                'accuracy': window.accuracy,
                'precision': window.precision,
                'recall': window.recall,
                'f1_score': window.f1_score,
                'mae': window.mae,
                'rmse': window.rmse,
                'mape': window.mape,
                'direction_accuracy': window.direction_accuracy
            },
            metadata={
                'method': 'single_split',
                'train_size': len(train_states),
                'test_size': len(test_states),
                'total_train_time': train_time,
                'total_predict_time': predict_time
            }
        )

        return result

    def run_walk_forward(
        self,
        states: np.ndarray,
        prices: Optional[np.ndarray] = None,
        state_to_price_fn: Optional[Callable] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run walk-forward backtesting with rolling/expanding windows.

        This method simulates real-world trading where the model is trained
        on historical data and tested on future unseen data, then retrained
        as new data becomes available.

        Args:
            states: State sequence
            prices: Price sequence (optional)
            state_to_price_fn: Function to convert states to prices
            verbose: Print progress messages

        Returns:
            BacktestResult with multiple windows
        """
        states = np.asarray(states)
        n = len(states)

        if prices is not None:
            prices = np.asarray(prices)
            if len(prices) != n:
                raise ValueError("States and prices must have same length")

        # Determine initial split
        if 0 < self.train_size < 1:
            initial_train_size = int(n * self.train_size)
        else:
            initial_train_size = int(self.train_size)

        initial_train_size = max(initial_train_size, self.min_train_size)

        # Determine step size
        if self.step_size is None:
            # Default to 10% of window or 20, whichever is larger
            self.step_size = max(20, initial_train_size // 10)

        # Determine test window size
        if self.window_size is None:
            test_size = min(self.step_size, (n - initial_train_size) // 5)
        else:
            test_size = self.window_size

        windows = []
        train_start = 0
        train_end = initial_train_size
        window_id = 0

        total_train_time = 0.0
        total_predict_time = 0.0

        if verbose:
            logger.info(f"Starting walk-forward backtest: {n} samples, initial train={initial_train_size}")

        while train_end + test_size <= n:
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if verbose:
                logger.info(f"Window {window_id}: train=[{train_start}:{train_end}], test=[{test_start}:{test_end}]")

            # Extract train/test data
            train_states = states[train_start:train_end]
            test_states = states[test_start:test_end]

            if prices is not None:
                test_prices = prices[test_start:test_end]
            else:
                test_prices = None

            # Retrain model if needed
            if window_id % self.retrain_interval == 0:
                start_time = datetime.now()
                self.model.fit(train_states)
                train_time = (datetime.now() - start_time).total_seconds()
                total_train_time += train_time
            else:
                train_time = 0.0

            # Make predictions
            start_time = datetime.now()
            predicted_states = []

            order = getattr(self.model, 'order', 1)

            # Initialize history from training data
            if order == 1:
                history = int(train_states[-1])
            else:
                history = list(train_states[-order:])

            for i in range(len(test_states)):
                pred = self._get_prediction(history)
                predicted_states.append(pred)

                # Update history with actual state
                if order == 1:
                    history = int(test_states[i])
                else:
                    history = list(history[1:]) + [int(test_states[i])]

            predicted_states = np.array(predicted_states)
            predict_time = (datetime.now() - start_time).total_seconds()
            total_predict_time += predict_time

            # Convert to prices if needed
            predicted_prices = None
            if state_to_price_fn is not None:
                predicted_prices = np.array([state_to_price_fn(s) for s in predicted_states])

            # Create window
            window = BacktestWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                predicted_states=predicted_states,
                actual_states=test_states,
                predicted_prices=predicted_prices,
                actual_prices=test_prices,
                model_params={
                    'n_states': getattr(self.model, 'n_states', None),
                    'order': order,
                    'alpha': getattr(self.model, 'alpha', None)
                },
                train_time=train_time,
                predict_time=predict_time
            )

            # Calculate metrics
            self._calculate_window_metrics(
                window,
                predicted_states,
                test_states,
                predicted_prices,
                test_prices
            )

            windows.append(window)

            # Move to next window
            if self.expanding_window:
                # Expanding window: keep same start, extend end
                train_end += self.step_size
            else:
                # Rolling window: move both start and end
                if self.window_size is not None:
                    train_start += self.step_size
                train_end += self.step_size

            window_id += 1

        if verbose:
            logger.info(f"Completed {len(windows)} windows")

        # Calculate overall metrics
        overall_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score',
                      'mae', 'rmse', 'mape', 'direction_accuracy']:
            values = []
            for w in windows:
                val = getattr(w, metric)
                if val is not None:
                    values.append(val)
            if values:
                overall_metrics[metric] = float(np.mean(values))
                overall_metrics[f'{metric}_std'] = float(np.std(values))

        # Create result
        result = BacktestResult(
            windows=windows,
            overall_metrics=overall_metrics,
            metadata={
                'method': 'walk_forward',
                'n_windows': len(windows),
                'expanding_window': self.expanding_window,
                'retrain_interval': self.retrain_interval,
                'total_train_time': total_train_time,
                'total_predict_time': total_predict_time,
                'window_size': test_size,
                'step_size': self.step_size
            }
        )

        return result

    def run_time_series_cv(
        self,
        states: np.ndarray,
        n_splits: int = 5,
        prices: Optional[np.ndarray] = None,
        state_to_price_fn: Optional[Callable] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run time-series cross-validation.

        Similar to walk-forward but with fixed number of splits.
        Each split uses all previous data for training.

        Args:
            states: State sequence
            n_splits: Number of CV folds
            prices: Price sequence (optional)
            state_to_price_fn: Function to convert states to prices
            verbose: Print progress

        Returns:
            BacktestResult with CV results
        """
        states = np.asarray(states)
        n = len(states)

        if prices is not None:
            prices = np.asarray(prices)

        # Calculate split points
        min_train = max(self.min_train_size, n // (n_splits + 2))
        test_size = (n - min_train) // n_splits

        windows = []
        total_train_time = 0.0
        total_predict_time = 0.0

        if verbose:
            logger.info(f"Starting {n_splits}-fold time-series CV")

        for fold in range(n_splits):
            train_end = min_train + fold * test_size
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_end - test_start < 10:  # Skip too small test sets
                continue

            # Extract data
            train_states = states[:train_end]
            test_states = states[test_start:test_end]

            if prices is not None:
                test_prices = prices[test_start:test_end]
            else:
                test_prices = None

            # Train
            start_time = datetime.now()
            self.model.fit(train_states)
            train_time = (datetime.now() - start_time).total_seconds()
            total_train_time += train_time

            # Predict
            start_time = datetime.now()
            predicted_states = []

            order = getattr(self.model, 'order', 1)

            if order == 1:
                history = int(train_states[-1])
            else:
                history = list(train_states[-order:])

            for i in range(len(test_states)):
                pred = self._get_prediction(history)
                predicted_states.append(pred)

                if order == 1:
                    history = int(test_states[i])
                else:
                    history = list(history[1:]) + [int(test_states[i])]

            predicted_states = np.array(predicted_states)
            predict_time = (datetime.now() - start_time).total_seconds()
            total_predict_time += predict_time

            # Convert prices
            predicted_prices = None
            if state_to_price_fn is not None:
                predicted_prices = np.array([state_to_price_fn(s) for s in predicted_states])

            # Create window
            window = BacktestWindow(
                window_id=fold,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                predicted_states=predicted_states,
                actual_states=test_states,
                predicted_prices=predicted_prices,
                actual_prices=test_prices,
                model_params={
                    'n_states': getattr(self.model, 'n_states', None),
                    'order': order,
                    'alpha': getattr(self.model, 'alpha', None)
                },
                train_time=train_time,
                predict_time=predict_time
            )

            self._calculate_window_metrics(
                window,
                predicted_states,
                test_states,
                predicted_prices,
                test_prices
            )

            windows.append(window)

        # Calculate overall metrics
        overall_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score',
                      'mae', 'rmse', 'mape', 'direction_accuracy']:
            values = []
            for w in windows:
                val = getattr(w, metric)
                if val is not None:
                    values.append(val)
            if values:
                overall_metrics[metric] = float(np.mean(values))
                overall_metrics[f'{metric}_std'] = float(np.std(values))

        result = BacktestResult(
            windows=windows,
            overall_metrics=overall_metrics,
            metadata={
                'method': 'time_series_cv',
                'n_splits': n_splits,
                'total_train_time': total_train_time,
                'total_predict_time': total_predict_time
            }
        )

        return result

    def statistical_significance_test(
        self,
        result1: BacktestResult,
        result2: BacktestResult,
        metric: str = 'accuracy',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test if difference between two backtest results is statistically significant.

        Uses paired t-test on window-level metrics.

        Args:
            result1: First backtest result
            result2: Second backtest result
            metric: Metric to compare
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        values1 = result1.get_metric_series(metric)
        values2 = result2.get_metric_series(metric)

        if len(values1) != len(values2):
            raise ValueError("Results must have same number of windows for paired test")

        if len(values1) < 2:
            return {
                'significant': False,
                'reason': 'Insufficient samples for statistical test'
            }

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(values1, values2)

        # Effect size (Cohen's d)
        diff = values1 - values2
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

        return {
            'metric': metric,
            'mean_1': float(np.mean(values1)),
            'mean_2': float(np.mean(values2)),
            'mean_diff': float(np.mean(diff)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_cohens_d(cohens_d),
            'winner': 1 if np.mean(values1) > np.mean(values2) else 2
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return 'negligible'
        elif d_abs < 0.5:
            return 'small'
        elif d_abs < 0.8:
            return 'medium'
        else:
            return 'large'

    def compare_models(
        self,
        models: List[Any],
        model_names: List[str],
        states: np.ndarray,
        prices: Optional[np.ndarray] = None,
        method: str = 'walk_forward',
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple models using backtesting.

        Args:
            models: List of model instances
            model_names: Names for each model
            states: State sequence
            prices: Price sequence (optional)
            method: Backtesting method ('walk_forward', 'cv', 'single')
            **kwargs: Additional arguments for backtesting method

        Returns:
            DataFrame comparing model performance
        """
        if len(models) != len(model_names):
            raise ValueError("Number of models and names must match")

        results = []

        for model, name in zip(models, model_names):
            logger.info(f"Backtesting model: {name}")

            # Temporarily set model
            original_model = self.model
            self.model = model

            try:
                # Filter kwargs for each method
                if method == 'walk_forward':
                    valid_kwargs = {k: v for k, v in kwargs.items()
                                  if k in ['state_to_price_fn', 'verbose']}
                    result = self.run_walk_forward(states, prices, **valid_kwargs)
                elif method == 'cv':
                    valid_kwargs = {k: v for k, v in kwargs.items()
                                  if k in ['n_splits', 'state_to_price_fn', 'verbose']}
                    result = self.run_time_series_cv(states, prices=prices, **valid_kwargs)
                elif method == 'single':
                    valid_kwargs = {k: v for k, v in kwargs.items()
                                  if k in ['state_to_price_fn']}
                    result = self.run_single_backtest(states, prices, **valid_kwargs)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Extract metrics
                metrics = {'model': name}
                metrics.update(result.overall_metrics)
                results.append(metrics)

            finally:
                # Restore original model
                self.model = original_model

        return pd.DataFrame(results).set_index('model')


def plot_backtest_results(
    result: BacktestResult,
    save_path: Optional[str] = None,
    show_confidence: bool = True
) -> None:
    """
    Visualize backtesting results.

    Args:
        result: BacktestResult to visualize
        save_path: Path to save figure (optional)
        show_confidence: Show confidence intervals
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("Matplotlib and seaborn required for plotting")
        return

    # Set style
    sns.set_style("whitegrid")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Backtesting Results', fontsize=16, fontweight='bold')

    # 1. Metrics over time
    ax = axes[0, 0]
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']

    for metric in metrics_to_plot:
        values = result.get_metric_series(metric)
        if len(values) > 0:
            ax.plot(range(len(values)), values, marker='o', label=metric, linewidth=2)

    ax.set_xlabel('Window Index')
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Regression metrics (if available)
    ax = axes[0, 1]
    mae_values = result.get_metric_series('mae')
    rmse_values = result.get_metric_series('rmse')

    if len(mae_values) > 0:
        ax2 = ax.twinx()
        ax.plot(range(len(mae_values)), mae_values, marker='s', label='MAE', color='blue', linewidth=2)
        ax2.plot(range(len(rmse_values)), rmse_values, marker='^', label='RMSE', color='red', linewidth=2)
        ax.set_xlabel('Window Index')
        ax.set_ylabel('MAE', color='blue')
        ax2.set_ylabel('RMSE', color='red')
        ax.set_title('Regression Metrics Over Time')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
    else:
        ax.text(0.5, 0.5, 'No regression metrics available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Regression Metrics Over Time')

    # 3. Metric distribution
    ax = axes[1, 0]
    accuracy_values = result.get_metric_series('accuracy')

    if len(accuracy_values) > 1:
        ax.hist(accuracy_values, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(accuracy_values), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(accuracy_values):.3f}')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')
        ax.set_title('Accuracy Distribution')
        ax.legend()

    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary text
    summary_text = "Overall Performance Summary\n" + "="*40 + "\n\n"

    for metric, value in result.overall_metrics.items():
        if not metric.endswith('_std'):
            summary_text += f"{metric:20s}: {value:8.4f}\n"

    summary_text += "\n" + "="*40 + "\n"
    summary_text += f"Method: {result.metadata.get('method', 'N/A')}\n"
    summary_text += f"Windows: {len(result.windows)}\n"
    summary_text += f"Train time: {result.metadata.get('total_train_time', 0):.2f}s\n"
    summary_text += f"Predict time: {result.metadata.get('total_predict_time', 0):.2f}s\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # This section demonstrates usage - can be removed in production

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_states = 5

    # Random walk with some structure
    states = np.zeros(n_samples, dtype=int)
    states[0] = n_states // 2

    for i in range(1, n_samples):
        # Tendency to stay in same state or move to adjacent states
        if np.random.rand() < 0.5:
            states[i] = states[i-1]
        else:
            change = np.random.choice([-1, 0, 1])
            states[i] = np.clip(states[i-1] + change, 0, n_states - 1)

    # Generate synthetic prices
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 2)

    print("Backtesting Framework Example")
    print("="*50)

    # Import MarkovChain (would normally import from module)
    # from src.core.models.markov import MarkovChain

    # For demonstration, create a simple mock
    class SimpleMarkov:
        def __init__(self, n_states=5, alpha=0.001):
            self.n_states = n_states
            self.alpha = alpha
            self.order = 1
            self.transition_matrix = None

        def fit(self, states):
            transitions = np.zeros((self.n_states, self.n_states))
            for i in range(len(states) - 1):
                transitions[states[i]][states[i+1]] += 1
            transitions += self.alpha
            self.transition_matrix = transitions / transitions.sum(axis=1, keepdims=True)
            return self

        def predict(self, state, n_steps=1):
            dist = np.zeros(self.n_states)
            dist[state] = 1.0
            for _ in range(n_steps):
                dist = dist @ self.transition_matrix
            return dist

    # Create model
    model = SimpleMarkov(n_states=n_states)

    # Create backtester
    backtester = Backtester(
        model=model,
        train_size=0.7,
        window_size=50,
        step_size=25,
        retrain_interval=2
    )

    # Run single backtest
    print("\n1. Running single train/test split...")
    result_single = backtester.run_single_backtest(states, prices)
    print(f"   Accuracy: {result_single.overall_metrics['accuracy']:.3f}")
    print(f"   RMSE: {result_single.overall_metrics.get('rmse', 'N/A')}")

    # Run walk-forward
    print("\n2. Running walk-forward analysis...")
    result_wf = backtester.run_walk_forward(states, prices, verbose=False)
    print(f"   Windows: {len(result_wf.windows)}")
    print(f"   Mean Accuracy: {result_wf.overall_metrics['accuracy']:.3f} ± {result_wf.overall_metrics['accuracy_std']:.3f}")

    # Run CV
    print("\n3. Running time-series cross-validation...")
    result_cv = backtester.run_time_series_cv(states, n_splits=5, prices=prices, verbose=False)
    print(f"   Folds: {len(result_cv.windows)}")
    print(f"   Mean Accuracy: {result_cv.overall_metrics['accuracy']:.3f}")

    # Print summary
    print("\n4. Summary statistics:")
    print(result_wf.summary())

    print("\n" + "="*50)
    print("Backtesting framework ready for production use!")

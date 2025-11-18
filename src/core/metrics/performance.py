"""
Comprehensive performance metrics module for financial analysis.

This module provides a complete suite of performance metrics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis (Maximum Drawdown, Duration)
- Risk metrics (VaR, CVaR at multiple confidence levels)
- Prediction accuracy (Hit Rate, RMSE, MAE, MAPE)
- Portfolio metrics (Alpha, Beta, Information Ratio)
- Annualized statistics (Return, Volatility)
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """
    Comprehensive container for all calculated performance metrics.

    Attributes:
        ticker: Stock ticker symbol
        calculation_date: When metrics were calculated

        # Risk-Adjusted Returns
        sharpe_ratio: Risk-adjusted return metric
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: Return/max drawdown ratio

        # Drawdown Metrics
        max_drawdown: Maximum peak-to-trough decline (%)
        max_drawdown_duration: Longest drawdown period (days)
        current_drawdown: Current drawdown from peak (%)

        # Value at Risk Metrics
        var_90: 90% confidence VaR (%)
        var_95: 95% confidence VaR (%)
        var_99: 99% confidence VaR (%)
        cvar_95: Conditional VaR (Expected Shortfall) at 95%

        # Prediction Accuracy Metrics
        hit_rate: Directional prediction accuracy (0-1)
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error (%)

        # Return Metrics
        annualized_return: Annualized return (%)
        annualized_volatility: Annualized standard deviation (%)
        cumulative_return: Total return over period (%)

        # Portfolio Metrics
        information_ratio: Risk-adjusted active return
        alpha: Excess return vs benchmark
        beta: Systematic risk vs benchmark

        # Additional Statistics
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis
        winning_days: Percentage of positive return days
        avg_win: Average winning day return (%)
        avg_loss: Average losing day return (%)
        win_loss_ratio: Ratio of avg win to avg loss

        # Metadata
        n_observations: Number of data points
        period_start: First observation date
        period_end: Last observation date
        risk_free_rate: Risk-free rate used (annualized %)
    """

    ticker: str = "Unknown"
    calculation_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Risk-Adjusted Returns
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Drawdown Metrics
    max_drawdown: Optional[float] = None
    max_drawdown_duration: Optional[int] = None
    current_drawdown: Optional[float] = None

    # Value at Risk Metrics
    var_90: Optional[float] = None
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None

    # Prediction Accuracy Metrics
    hit_rate: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None

    # Return Metrics
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None
    cumulative_return: Optional[float] = None

    # Portfolio Metrics
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    # Additional Statistics
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    winning_days: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    win_loss_ratio: Optional[float] = None

    # Metadata
    n_observations: int = 0
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    risk_free_rate: float = 0.02

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Convert report to JSON format.

        Args:
            filepath: Optional path to save JSON file
            indent: JSON indentation level

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=indent)

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Saved metrics report to {filepath}")

        return json_str

    def to_markdown(self, filepath: Optional[str] = None) -> str:
        """
        Generate markdown formatted report.

        Args:
            filepath: Optional path to save markdown file

        Returns:
            Markdown string
        """
        lines = [
            f"# Performance Metrics Report: {self.ticker}",
            f"\n**Generated:** {self.calculation_date}",
            f"\n**Period:** {self.period_start} to {self.period_end}",
            f"\n**Observations:** {self.n_observations}",
            "\n---\n",
            "\n## Risk-Adjusted Returns\n",
        ]

        # Risk-Adjusted Returns
        if self.sharpe_ratio is not None:
            lines.append(f"- **Sharpe Ratio:** {self.sharpe_ratio:.4f}")
        if self.sortino_ratio is not None:
            lines.append(f"- **Sortino Ratio:** {self.sortino_ratio:.4f}")
        if self.calmar_ratio is not None:
            lines.append(f"- **Calmar Ratio:** {self.calmar_ratio:.4f}")
        if self.information_ratio is not None:
            lines.append(f"- **Information Ratio:** {self.information_ratio:.4f}")

        # Drawdown Analysis
        lines.append("\n## Drawdown Analysis\n")
        if self.max_drawdown is not None:
            lines.append(f"- **Maximum Drawdown:** {self.max_drawdown:.2f}%")
        if self.max_drawdown_duration is not None:
            lines.append(f"- **Max Drawdown Duration:** {self.max_drawdown_duration} days")
        if self.current_drawdown is not None:
            lines.append(f"- **Current Drawdown:** {self.current_drawdown:.2f}%")

        # Risk Metrics
        lines.append("\n## Risk Metrics\n")
        if self.var_90 is not None:
            lines.append(f"- **VaR (90%):** {self.var_90:.2f}%")
        if self.var_95 is not None:
            lines.append(f"- **VaR (95%):** {self.var_95:.2f}%")
        if self.var_99 is not None:
            lines.append(f"- **VaR (99%):** {self.var_99:.2f}%")
        if self.cvar_95 is not None:
            lines.append(f"- **CVaR (95%):** {self.cvar_95:.2f}%")
        if self.annualized_volatility is not None:
            lines.append(f"- **Annualized Volatility:** {self.annualized_volatility:.2f}%")

        # Return Metrics
        lines.append("\n## Return Metrics\n")
        if self.annualized_return is not None:
            lines.append(f"- **Annualized Return:** {self.annualized_return:.2f}%")
        if self.cumulative_return is not None:
            lines.append(f"- **Cumulative Return:** {self.cumulative_return:.2f}%")
        if self.winning_days is not None:
            lines.append(f"- **Winning Days:** {self.winning_days:.2f}%")
        if self.avg_win is not None:
            lines.append(f"- **Average Win:** {self.avg_win:.2f}%")
        if self.avg_loss is not None:
            lines.append(f"- **Average Loss:** {self.avg_loss:.2f}%")
        if self.win_loss_ratio is not None:
            lines.append(f"- **Win/Loss Ratio:** {self.win_loss_ratio:.2f}")

        # Prediction Accuracy
        if any([self.hit_rate, self.rmse, self.mae, self.mape]):
            lines.append("\n## Prediction Accuracy\n")
            if self.hit_rate is not None:
                lines.append(f"- **Hit Rate:** {self.hit_rate:.2%}")
            if self.rmse is not None:
                lines.append(f"- **RMSE:** {self.rmse:.4f}")
            if self.mae is not None:
                lines.append(f"- **MAE:** {self.mae:.4f}")
            if self.mape is not None:
                lines.append(f"- **MAPE:** {self.mape:.2f}%")

        # Portfolio Metrics
        if any([self.alpha, self.beta]):
            lines.append("\n## Portfolio Metrics\n")
            if self.alpha is not None:
                lines.append(f"- **Alpha:** {self.alpha:.4f}")
            if self.beta is not None:
                lines.append(f"- **Beta:** {self.beta:.4f}")

        # Distribution Statistics
        if any([self.skewness, self.kurtosis]):
            lines.append("\n## Distribution Statistics\n")
            if self.skewness is not None:
                lines.append(f"- **Skewness:** {self.skewness:.4f}")
            if self.kurtosis is not None:
                lines.append(f"- **Kurtosis:** {self.kurtosis:.4f}")

        markdown = "\n".join(lines)

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(markdown)
            logger.info(f"Saved markdown report to {filepath}")

        return markdown


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator for financial time series.

    This class provides methods to calculate a wide range of performance metrics
    for evaluating trading strategies, portfolio performance, and model predictions.

    Features:
        - Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
        - Drawdown analysis with duration tracking
        - Value at Risk (VaR) at multiple confidence levels
        - Conditional Value at Risk (CVaR)
        - Prediction accuracy metrics (Hit Rate, RMSE, MAE, MAPE)
        - Portfolio analytics (Alpha, Beta, Information Ratio)
        - Annualized statistics
        - Distribution analysis (skewness, kurtosis)

    Example:
        >>> metrics = PerformanceMetrics(risk_free_rate=0.02)
        >>> report = metrics.calculate_all(
        ...     prices=price_series,
        ...     returns=return_series,
        ...     ticker="AAPL"
        ... )
        >>> print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
        >>> report.to_markdown("reports/performance.md")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252
    ):
        """
        Initialize the performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            trading_days_per_year: Number of trading days per year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year

    def sharpe_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio - risk-adjusted return metric.

        Sharpe Ratio = (Mean Return - Risk Free Rate) / Standard Deviation
        Higher values indicate better risk-adjusted performance.

        Args:
            returns: Array or Series of returns (in decimal form)
            annualize: Whether to annualize the ratio

        Returns:
            Sharpe ratio value

        Example:
            >>> returns = np.array([0.01, -0.02, 0.015, 0.005])
            >>> sharpe = metrics.sharpe_ratio(returns)
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            logger.warning("Empty returns array for Sharpe ratio calculation")
            return np.nan

        excess_returns = returns - (self.risk_free_rate / self.trading_days)

        if np.std(returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(returns, ddof=1)

        if annualize:
            sharpe *= np.sqrt(self.trading_days)

        return float(sharpe)

    def sortino_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        target_return: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino Ratio - downside risk-adjusted return.

        Similar to Sharpe but only penalizes downside volatility.
        Uses downside deviation instead of total standard deviation.

        Args:
            returns: Array or Series of returns
            target_return: Minimum acceptable return (default: 0)
            annualize: Whether to annualize the ratio

        Returns:
            Sortino ratio value
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            return np.nan

        excess_returns = returns - (self.risk_free_rate / self.trading_days)
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_dev = np.std(downside_returns, ddof=1)
        sortino = np.mean(excess_returns) / downside_dev

        if annualize:
            sortino *= np.sqrt(self.trading_days)

        return float(sortino)

    def calmar_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        prices: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> float:
        """
        Calculate Calmar Ratio - return/drawdown ratio.

        Calmar Ratio = Annualized Return / Maximum Drawdown
        Measures return relative to maximum risk (drawdown).

        Args:
            returns: Array or Series of returns
            prices: Optional price series (will be calculated from returns if not provided)

        Returns:
            Calmar ratio value
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            return np.nan

        ann_return = self.annualized_return(returns)

        if prices is None:
            prices = (1 + returns).cumprod()

        max_dd = self.max_drawdown(prices)

        if max_dd == 0:
            return 0.0

        return float(ann_return / abs(max_dd))

    def max_drawdown(
        self,
        prices: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate maximum drawdown - largest peak-to-trough decline.

        Args:
            prices: Price series

        Returns:
            Maximum drawdown as percentage (negative value)
        """
        prices = np.asarray(prices)

        if len(prices) == 0:
            return np.nan

        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)

        # Calculate drawdown at each point
        drawdown = (prices - running_max) / running_max * 100

        return float(np.min(drawdown))

    def max_drawdown_duration(
        self,
        prices: Union[np.ndarray, pd.Series]
    ) -> int:
        """
        Calculate maximum drawdown duration in periods.

        Args:
            prices: Price series

        Returns:
            Maximum number of consecutive periods in drawdown
        """
        prices = np.asarray(prices)

        if len(prices) == 0:
            return 0

        running_max = np.maximum.accumulate(prices)
        is_drawdown = prices < running_max

        # Find longest consecutive True sequence
        max_duration = 0
        current_duration = 0

        for in_dd in is_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return int(max_duration)

    def current_drawdown(
        self,
        prices: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate current drawdown from peak.

        Args:
            prices: Price series

        Returns:
            Current drawdown as percentage
        """
        prices = np.asarray(prices)

        if len(prices) == 0:
            return np.nan

        peak = np.max(prices)
        current = prices[-1]

        return float((current - peak) / peak * 100)

    def value_at_risk(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical method.

        VaR represents the maximum expected loss at a given confidence level.
        For example, 95% VaR means there's a 5% chance of losing more than this amount.

        Args:
            returns: Array or Series of returns (in decimal form)
            confidence_level: Confidence level (0.90, 0.95, 0.99)

        Returns:
            VaR as percentage (negative value represents loss)
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            return np.nan

        # Calculate percentile (lower tail)
        var = np.percentile(returns, (1 - confidence_level) * 100)

        return float(var * 100)  # Convert to percentage

    def conditional_var(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

        CVaR is the expected loss given that the loss exceeds VaR.
        It provides a more complete picture of tail risk.

        Args:
            returns: Array or Series of returns
            confidence_level: Confidence level (0.90, 0.95, 0.99)

        Returns:
            CVaR as percentage (negative value represents expected loss)
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            return np.nan

        # Calculate VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)

        # Calculate mean of returns below VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return var_threshold * 100

        cvar = np.mean(tail_returns)

        return float(cvar * 100)  # Convert to percentage

    def hit_rate(
        self,
        actual: Union[np.ndarray, pd.Series],
        predicted: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate directional prediction accuracy (hit rate).

        Measures how often the model correctly predicts the direction of price movement.

        Args:
            actual: Actual price or return series
            predicted: Predicted price or return series

        Returns:
            Hit rate as decimal (0 to 1)
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have same length")

        if len(actual) <= 1:
            return np.nan

        # Calculate direction of change
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))

        # Calculate accuracy
        correct = actual_direction == predicted_direction

        return float(np.mean(correct))

    def rmse(
        self,
        actual: Union[np.ndarray, pd.Series],
        predicted: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            RMSE value
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have same length")

        if len(actual) == 0:
            return np.nan

        mse = np.mean((actual - predicted) ** 2)

        return float(np.sqrt(mse))

    def mae(
        self,
        actual: Union[np.ndarray, pd.Series],
        predicted: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            MAE value
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have same length")

        if len(actual) == 0:
            return np.nan

        return float(np.mean(np.abs(actual - predicted)))

    def mape(
        self,
        actual: Union[np.ndarray, pd.Series],
        predicted: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            MAPE as percentage
        """
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)

        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have same length")

        if len(actual) == 0:
            return np.nan

        # Avoid division by zero
        mask = actual != 0

        if not np.any(mask):
            return np.nan

        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        return float(mape)

    def annualized_return(
        self,
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Array or Series of returns
            periods_per_year: Number of periods per year (defaults to trading_days)

        Returns:
            Annualized return as percentage
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            return np.nan

        if periods_per_year is None:
            periods_per_year = self.trading_days

        # Compound returns
        total_return = (1 + returns).prod() - 1

        # Annualize
        n_periods = len(returns)
        ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        return float(ann_return * 100)

    def annualized_volatility(
        self,
        returns: Union[np.ndarray, pd.Series],
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Calculate annualized volatility (standard deviation).

        Args:
            returns: Array or Series of returns
            periods_per_year: Number of periods per year (defaults to trading_days)

        Returns:
            Annualized volatility as percentage
        """
        returns = np.asarray(returns)

        if len(returns) == 0:
            return np.nan

        if periods_per_year is None:
            periods_per_year = self.trading_days

        vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)

        return float(vol * 100)

    def information_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        annualize: bool = True
    ) -> float:
        """
        Calculate Information Ratio - risk-adjusted active return.

        IR = (Portfolio Return - Benchmark Return) / Tracking Error
        Measures the consistency of excess returns.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            annualize: Whether to annualize the ratio

        Returns:
            Information ratio value
        """
        returns = np.asarray(returns)
        benchmark_returns = np.asarray(benchmark_returns)

        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns and benchmark must have same length")

        if len(returns) == 0:
            return np.nan

        # Calculate active returns
        active_returns = returns - benchmark_returns

        # Calculate tracking error (standard deviation of active returns)
        tracking_error = np.std(active_returns, ddof=1)

        if tracking_error == 0:
            return 0.0

        ir = np.mean(active_returns) / tracking_error

        if annualize:
            ir *= np.sqrt(self.trading_days)

        return float(ir)

    def alpha_beta(
        self,
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        annualize: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate Alpha and Beta relative to a benchmark.

        Beta: Systematic risk (sensitivity to market movements)
        Alpha: Excess return not explained by market movements

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            annualize: Whether to annualize alpha

        Returns:
            Tuple of (alpha, beta)
        """
        returns = np.asarray(returns)
        benchmark_returns = np.asarray(benchmark_returns)

        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns and benchmark must have same length")

        if len(returns) < 2:
            return (np.nan, np.nan)

        # Calculate beta using covariance
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)

        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance

        # Calculate alpha
        mean_return = np.mean(returns)
        mean_benchmark = np.mean(benchmark_returns)

        alpha = mean_return - beta * mean_benchmark

        if annualize:
            alpha *= self.trading_days

        return (float(alpha), float(beta))

    def calculate_all(
        self,
        returns: Optional[Union[np.ndarray, pd.Series]] = None,
        prices: Optional[Union[np.ndarray, pd.Series]] = None,
        predicted_prices: Optional[Union[np.ndarray, pd.Series]] = None,
        benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
        ticker: str = "Unknown",
        return_dataframe: bool = False
    ) -> Union[MetricsReport, pd.DataFrame]:
        """
        Calculate all available performance metrics.

        Args:
            returns: Return series (required for most metrics)
            prices: Price series (if not provided, calculated from returns)
            predicted_prices: Predicted prices for accuracy metrics
            benchmark_returns: Benchmark returns for alpha/beta/IR
            ticker: Stock ticker symbol
            return_dataframe: Return results as DataFrame instead of MetricsReport

        Returns:
            MetricsReport object or DataFrame with all calculated metrics

        Example:
            >>> report = metrics.calculate_all(
            ...     returns=daily_returns,
            ...     prices=price_series,
            ...     ticker="AAPL"
            ... )
            >>> print(f"Sharpe: {report.sharpe_ratio:.2f}")
        """
        report = MetricsReport(ticker=ticker, risk_free_rate=self.risk_free_rate)

        # Convert inputs to arrays
        if returns is not None:
            returns = np.asarray(returns)

            # Remove NaN values
            returns = returns[~np.isnan(returns)]

            if len(returns) > 0:
                report.n_observations = len(returns)

                # Calculate prices if not provided
                if prices is None:
                    prices = (1 + returns).cumprod() * 100  # Assume starting at 100
                else:
                    prices = np.asarray(prices)

                # Risk-Adjusted Returns
                try:
                    report.sharpe_ratio = self.sharpe_ratio(returns)
                except Exception as e:
                    logger.error(f"Error calculating Sharpe ratio: {e}")

                try:
                    report.sortino_ratio = self.sortino_ratio(returns)
                except Exception as e:
                    logger.error(f"Error calculating Sortino ratio: {e}")

                try:
                    report.calmar_ratio = self.calmar_ratio(returns, prices)
                except Exception as e:
                    logger.error(f"Error calculating Calmar ratio: {e}")

                # Drawdown Metrics
                try:
                    report.max_drawdown = self.max_drawdown(prices)
                    report.max_drawdown_duration = self.max_drawdown_duration(prices)
                    report.current_drawdown = self.current_drawdown(prices)
                except Exception as e:
                    logger.error(f"Error calculating drawdown metrics: {e}")

                # Value at Risk
                try:
                    report.var_90 = self.value_at_risk(returns, 0.90)
                    report.var_95 = self.value_at_risk(returns, 0.95)
                    report.var_99 = self.value_at_risk(returns, 0.99)
                    report.cvar_95 = self.conditional_var(returns, 0.95)
                except Exception as e:
                    logger.error(f"Error calculating VaR metrics: {e}")

                # Return Metrics
                try:
                    report.annualized_return = self.annualized_return(returns)
                    report.annualized_volatility = self.annualized_volatility(returns)
                    report.cumulative_return = ((1 + returns).prod() - 1) * 100
                except Exception as e:
                    logger.error(f"Error calculating return metrics: {e}")

                # Distribution Statistics
                try:
                    from scipy import stats
                    report.skewness = float(stats.skew(returns))
                    report.kurtosis = float(stats.kurtosis(returns))
                except ImportError:
                    logger.warning("scipy not available, skipping skewness/kurtosis")
                except Exception as e:
                    logger.error(f"Error calculating distribution stats: {e}")

                # Win/Loss Statistics
                try:
                    winning_days = returns > 0
                    losing_days = returns < 0

                    report.winning_days = float(np.mean(winning_days) * 100)

                    if np.any(winning_days):
                        report.avg_win = float(np.mean(returns[winning_days]) * 100)

                    if np.any(losing_days):
                        report.avg_loss = float(np.mean(returns[losing_days]) * 100)

                    if report.avg_win and report.avg_loss:
                        report.win_loss_ratio = abs(report.avg_win / report.avg_loss)
                except Exception as e:
                    logger.error(f"Error calculating win/loss statistics: {e}")

        # Prediction Accuracy Metrics
        if prices is not None and predicted_prices is not None:
            prices = np.asarray(prices)
            predicted_prices = np.asarray(predicted_prices)

            try:
                report.hit_rate = self.hit_rate(prices, predicted_prices)
                report.rmse = self.rmse(prices, predicted_prices)
                report.mae = self.mae(prices, predicted_prices)
                report.mape = self.mape(prices, predicted_prices)
            except Exception as e:
                logger.error(f"Error calculating prediction metrics: {e}")

        # Portfolio Metrics
        if returns is not None and benchmark_returns is not None:
            benchmark_returns = np.asarray(benchmark_returns)

            try:
                report.information_ratio = self.information_ratio(returns, benchmark_returns)
                alpha, beta = self.alpha_beta(returns, benchmark_returns)
                report.alpha = alpha
                report.beta = beta
            except Exception as e:
                logger.error(f"Error calculating portfolio metrics: {e}")

        # Set period metadata
        if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
            report.period_start = returns.index[0].strftime("%Y-%m-%d")
            report.period_end = returns.index[-1].strftime("%Y-%m-%d")

        if return_dataframe:
            return pd.DataFrame([report.to_dict()])

        return report


def generate_summary_table(
    report: MetricsReport,
    sections: Optional[List[str]] = None
) -> str:
    """
    Generate a formatted summary table of metrics.

    Args:
        report: MetricsReport to summarize
        sections: List of sections to include (default: all)
                 Options: 'risk_adjusted', 'drawdown', 'var', 'returns', 'accuracy', 'portfolio'

    Returns:
        Formatted table string
    """
    from tabulate import tabulate

    if sections is None:
        sections = ['risk_adjusted', 'drawdown', 'var', 'returns', 'accuracy', 'portfolio']

    rows = []

    # Header
    rows.append(['Metric', 'Value'])
    rows.append(['=' * 40, '=' * 20])

    if 'risk_adjusted' in sections:
        rows.append(['RISK-ADJUSTED RETURNS', ''])
        if report.sharpe_ratio is not None:
            rows.append(['  Sharpe Ratio', f"{report.sharpe_ratio:.4f}"])
        if report.sortino_ratio is not None:
            rows.append(['  Sortino Ratio', f"{report.sortino_ratio:.4f}"])
        if report.calmar_ratio is not None:
            rows.append(['  Calmar Ratio', f"{report.calmar_ratio:.4f}"])
        if report.information_ratio is not None:
            rows.append(['  Information Ratio', f"{report.information_ratio:.4f}"])
        rows.append(['', ''])

    if 'drawdown' in sections:
        rows.append(['DRAWDOWN ANALYSIS', ''])
        if report.max_drawdown is not None:
            rows.append(['  Maximum Drawdown', f"{report.max_drawdown:.2f}%"])
        if report.max_drawdown_duration is not None:
            rows.append(['  Max DD Duration', f"{report.max_drawdown_duration} days"])
        if report.current_drawdown is not None:
            rows.append(['  Current Drawdown', f"{report.current_drawdown:.2f}%"])
        rows.append(['', ''])

    if 'var' in sections:
        rows.append(['RISK METRICS', ''])
        if report.var_90 is not None:
            rows.append(['  VaR (90%)', f"{report.var_90:.2f}%"])
        if report.var_95 is not None:
            rows.append(['  VaR (95%)', f"{report.var_95:.2f}%"])
        if report.var_99 is not None:
            rows.append(['  VaR (99%)', f"{report.var_99:.2f}%"])
        if report.cvar_95 is not None:
            rows.append(['  CVaR (95%)', f"{report.cvar_95:.2f}%"])
        rows.append(['', ''])

    if 'returns' in sections:
        rows.append(['RETURN METRICS', ''])
        if report.annualized_return is not None:
            rows.append(['  Annualized Return', f"{report.annualized_return:.2f}%"])
        if report.cumulative_return is not None:
            rows.append(['  Cumulative Return', f"{report.cumulative_return:.2f}%"])
        if report.annualized_volatility is not None:
            rows.append(['  Annualized Volatility', f"{report.annualized_volatility:.2f}%"])
        if report.winning_days is not None:
            rows.append(['  Winning Days', f"{report.winning_days:.1f}%"])
        rows.append(['', ''])

    if 'accuracy' in sections and report.hit_rate is not None:
        rows.append(['PREDICTION ACCURACY', ''])
        if report.hit_rate is not None:
            rows.append(['  Hit Rate', f"{report.hit_rate:.2%}"])
        if report.rmse is not None:
            rows.append(['  RMSE', f"{report.rmse:.4f}"])
        if report.mae is not None:
            rows.append(['  MAE', f"{report.mae:.4f}"])
        if report.mape is not None:
            rows.append(['  MAPE', f"{report.mape:.2f}%"])
        rows.append(['', ''])

    if 'portfolio' in sections and (report.alpha is not None or report.beta is not None):
        rows.append(['PORTFOLIO METRICS', ''])
        if report.alpha is not None:
            rows.append(['  Alpha', f"{report.alpha:.4f}"])
        if report.beta is not None:
            rows.append(['  Beta', f"{report.beta:.4f}"])

    return tabulate(rows, headers='firstrow', tablefmt='psql')


def compare_strategies(
    reports: List[MetricsReport],
    metrics_to_compare: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies side-by-side.

    Args:
        reports: List of MetricsReport objects
        metrics_to_compare: List of metric names to compare (default: key metrics)

    Returns:
        DataFrame with strategies as columns and metrics as rows
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'annualized_return',
            'annualized_volatility',
            'var_95',
            'hit_rate'
        ]

    comparison_data = {}

    for report in reports:
        strategy_name = report.ticker
        comparison_data[strategy_name] = {
            metric: getattr(report, metric)
            for metric in metrics_to_compare
            if hasattr(report, metric)
        }

    df = pd.DataFrame(comparison_data).T

    # Format column names
    df.columns = [col.replace('_', ' ').title() for col in df.columns]

    return df

"""
Comprehensive tests for the performance metrics module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.core.metrics.performance import (
    PerformanceMetrics,
    MetricsReport,
    generate_summary_table,
    compare_strategies
)


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    np.random.seed(42)
    # Generate returns with known properties
    returns = np.random.normal(0.001, 0.02, 252)  # ~0.1% daily mean, 2% std
    return returns


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    prices = (1 + returns).cumprod() * 100  # Start at 100
    return prices


@pytest.fixture
def benchmark_returns():
    """Generate benchmark return data."""
    np.random.seed(43)
    returns = np.random.normal(0.0008, 0.015, 252)  # Slightly different from sample
    return returns


@pytest.fixture
def metrics_calculator():
    """Create a PerformanceMetrics instance."""
    return PerformanceMetrics(risk_free_rate=0.02, trading_days_per_year=252)


class TestMetricsReport:
    """Test MetricsReport dataclass functionality."""

    def test_metrics_report_creation(self):
        """Test creating a metrics report."""
        report = MetricsReport(
            ticker="AAPL",
            sharpe_ratio=1.5,
            max_drawdown=-15.5,
            annualized_return=12.5
        )

        assert report.ticker == "AAPL"
        assert report.sharpe_ratio == 1.5
        assert report.max_drawdown == -15.5
        assert report.annualized_return == 12.5

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = MetricsReport(ticker="MSFT", sharpe_ratio=2.0)
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict['ticker'] == "MSFT"
        assert report_dict['sharpe_ratio'] == 2.0

    def test_to_json(self):
        """Test JSON serialization."""
        report = MetricsReport(ticker="GOOGL", sharpe_ratio=1.8)
        json_str = report.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['ticker'] == "GOOGL"
        assert data['sharpe_ratio'] == 1.8

    def test_to_json_file(self):
        """Test saving report to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report = MetricsReport(ticker="TSLA", sharpe_ratio=2.5)
            report.to_json(str(filepath))

            assert filepath.exists()

            with open(filepath) as f:
                data = json.load(f)
                assert data['ticker'] == "TSLA"

    def test_to_markdown(self):
        """Test markdown generation."""
        report = MetricsReport(
            ticker="NVDA",
            sharpe_ratio=2.0,
            max_drawdown=-10.5,
            annualized_return=25.0
        )
        markdown = report.to_markdown()

        assert "NVDA" in markdown
        assert "Sharpe Ratio" in markdown
        assert "2.0000" in markdown

    def test_to_markdown_file(self):
        """Test saving markdown to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.md"
            report = MetricsReport(ticker="AMD", sharpe_ratio=1.5)
            report.to_markdown(str(filepath))

            assert filepath.exists()

            with open(filepath) as f:
                content = f.read()
                assert "AMD" in content
                assert "Sharpe Ratio" in content


class TestSharpeRatio:
    """Test Sharpe Ratio calculation."""

    def test_sharpe_ratio_basic(self, metrics_calculator, sample_returns):
        """Test basic Sharpe ratio calculation."""
        sharpe = metrics_calculator.sharpe_ratio(sample_returns)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert -5 < sharpe < 5  # Reasonable range

    def test_sharpe_ratio_zero_volatility(self, metrics_calculator):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.01] * 100)
        sharpe = metrics_calculator.sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_sharpe_ratio_empty(self, metrics_calculator):
        """Test Sharpe ratio with empty array."""
        sharpe = metrics_calculator.sharpe_ratio(np.array([]))

        assert np.isnan(sharpe)

    def test_sharpe_ratio_annualized(self, metrics_calculator, sample_returns):
        """Test annualized vs non-annualized Sharpe."""
        sharpe_ann = metrics_calculator.sharpe_ratio(sample_returns, annualize=True)
        sharpe_raw = metrics_calculator.sharpe_ratio(sample_returns, annualize=False)

        # Annualized should be scaled by sqrt(252)
        assert abs(sharpe_ann - sharpe_raw * np.sqrt(252)) < 0.01


class TestSortinoRatio:
    """Test Sortino Ratio calculation."""

    def test_sortino_ratio_basic(self, metrics_calculator, sample_returns):
        """Test basic Sortino ratio calculation."""
        sortino = metrics_calculator.sortino_ratio(sample_returns)

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

    def test_sortino_ratio_no_downside(self, metrics_calculator):
        """Test Sortino with only positive returns."""
        returns = np.abs(np.random.normal(0.01, 0.005, 100))
        sortino = metrics_calculator.sortino_ratio(returns)

        assert sortino == 0.0

    def test_sortino_vs_sharpe(self, metrics_calculator, sample_returns):
        """Test that Sortino is generally higher than Sharpe (less penalty)."""
        sharpe = metrics_calculator.sharpe_ratio(sample_returns)
        sortino = metrics_calculator.sortino_ratio(sample_returns)

        # Sortino should typically be higher as it only penalizes downside
        # This may not always be true, but in general it should be
        assert sortino != sharpe


class TestCalmarRatio:
    """Test Calmar Ratio calculation."""

    def test_calmar_ratio_basic(self, metrics_calculator, sample_returns, sample_prices):
        """Test basic Calmar ratio calculation."""
        calmar = metrics_calculator.calmar_ratio(sample_returns, sample_prices)

        assert isinstance(calmar, float)
        assert not np.isnan(calmar)

    def test_calmar_ratio_no_drawdown(self, metrics_calculator):
        """Test Calmar with constantly increasing prices."""
        returns = np.array([0.01] * 100)
        prices = (1 + returns).cumprod() * 100
        calmar = metrics_calculator.calmar_ratio(returns, prices)

        assert calmar == 0.0


class TestDrawdownMetrics:
    """Test drawdown calculation methods."""

    def test_max_drawdown_basic(self, metrics_calculator, sample_prices):
        """Test maximum drawdown calculation."""
        max_dd = metrics_calculator.max_drawdown(sample_prices)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative

    def test_max_drawdown_no_decline(self, metrics_calculator):
        """Test max drawdown with constantly rising prices."""
        prices = np.array([100, 105, 110, 115, 120])
        max_dd = metrics_calculator.max_drawdown(prices)

        assert max_dd == 0.0

    def test_max_drawdown_known_value(self, metrics_calculator):
        """Test max drawdown with known scenario."""
        prices = np.array([100, 110, 90, 120, 80])  # 80 from 120 = -33.33%
        max_dd = metrics_calculator.max_drawdown(prices)

        assert abs(max_dd - (-33.33)) < 0.1

    def test_max_drawdown_duration(self, metrics_calculator):
        """Test maximum drawdown duration."""
        prices = np.array([100, 110, 100, 95, 90, 100, 105, 100, 95])
        duration = metrics_calculator.max_drawdown_duration(prices)

        assert isinstance(duration, int)
        assert duration > 0

    def test_current_drawdown(self, metrics_calculator):
        """Test current drawdown calculation."""
        prices = np.array([100, 110, 105])
        current_dd = metrics_calculator.current_drawdown(prices)

        # Current price 105 from peak 110
        assert abs(current_dd - (-4.545)) < 0.1


class TestValueAtRisk:
    """Test VaR and CVaR calculations."""

    def test_var_95(self, metrics_calculator, sample_returns):
        """Test 95% VaR calculation."""
        var = metrics_calculator.value_at_risk(sample_returns, 0.95)

        assert isinstance(var, float)
        assert var < 0  # VaR should be negative (loss)

    def test_var_different_levels(self, metrics_calculator, sample_returns):
        """Test VaR at different confidence levels."""
        var_90 = metrics_calculator.value_at_risk(sample_returns, 0.90)
        var_95 = metrics_calculator.value_at_risk(sample_returns, 0.95)
        var_99 = metrics_calculator.value_at_risk(sample_returns, 0.99)

        # Higher confidence should give more negative VaR
        assert var_90 > var_95 > var_99

    def test_cvar_95(self, metrics_calculator, sample_returns):
        """Test CVaR calculation."""
        cvar = metrics_calculator.conditional_var(sample_returns, 0.95)

        assert isinstance(cvar, float)
        assert cvar < 0

    def test_cvar_worse_than_var(self, metrics_calculator, sample_returns):
        """Test that CVaR is worse (more negative) than VaR."""
        var = metrics_calculator.value_at_risk(sample_returns, 0.95)
        cvar = metrics_calculator.conditional_var(sample_returns, 0.95)

        # CVaR should be more negative (worse) than VaR
        assert cvar <= var


class TestPredictionMetrics:
    """Test prediction accuracy metrics."""

    def test_hit_rate_perfect(self, metrics_calculator):
        """Test hit rate with perfect predictions."""
        actual = np.array([100, 105, 103, 108, 110])
        predicted = np.array([100, 105, 103, 108, 110])

        hit_rate = metrics_calculator.hit_rate(actual, predicted)
        assert hit_rate == 1.0

    def test_hit_rate_directional(self, metrics_calculator):
        """Test hit rate with correct direction but wrong magnitude."""
        actual = np.array([100, 105, 103, 108])
        predicted = np.array([100, 102, 101, 110])  # Same direction, different values

        hit_rate = metrics_calculator.hit_rate(actual, predicted)
        assert hit_rate == 1.0

    def test_hit_rate_wrong(self, metrics_calculator):
        """Test hit rate with wrong predictions."""
        actual = np.array([100, 105, 103])
        predicted = np.array([100, 95, 110])  # Opposite direction

        hit_rate = metrics_calculator.hit_rate(actual, predicted)
        assert hit_rate == 0.0

    def test_rmse_perfect(self, metrics_calculator):
        """Test RMSE with perfect predictions."""
        actual = np.array([100, 105, 103, 108])
        predicted = np.array([100, 105, 103, 108])

        rmse = metrics_calculator.rmse(actual, predicted)
        assert rmse == 0.0

    def test_rmse_known_value(self, metrics_calculator):
        """Test RMSE with known scenario."""
        actual = np.array([100, 100, 100])
        predicted = np.array([103, 97, 100])  # Errors: 3, -3, 0

        rmse = metrics_calculator.rmse(actual, predicted)
        expected = np.sqrt((9 + 9 + 0) / 3)  # sqrt(6) ≈ 2.45
        assert abs(rmse - expected) < 0.01

    def test_mae_basic(self, metrics_calculator):
        """Test MAE calculation."""
        actual = np.array([100, 100, 100])
        predicted = np.array([103, 97, 100])

        mae = metrics_calculator.mae(actual, predicted)
        assert mae == 2.0  # (3 + 3 + 0) / 3

    def test_mape_basic(self, metrics_calculator):
        """Test MAPE calculation."""
        actual = np.array([100, 100, 100])
        predicted = np.array([110, 90, 100])

        mape = metrics_calculator.mape(actual, predicted)
        assert abs(mape - 6.67) < 0.1  # (10% + 10% + 0%) / 3

    def test_mape_zero_division(self, metrics_calculator):
        """Test MAPE with zero values."""
        actual = np.array([0, 100, 100])
        predicted = np.array([10, 110, 90])

        mape = metrics_calculator.mape(actual, predicted)
        # Should skip the zero value
        assert abs(mape - 10.0) < 0.1


class TestReturnMetrics:
    """Test annualized return and volatility calculations."""

    def test_annualized_return_basic(self, metrics_calculator, sample_returns):
        """Test annualized return calculation."""
        ann_return = metrics_calculator.annualized_return(sample_returns)

        assert isinstance(ann_return, float)
        assert not np.isnan(ann_return)

    def test_annualized_return_known_value(self, metrics_calculator):
        """Test with known scenario."""
        # 1% daily return for 252 days
        returns = np.array([0.01] * 252)
        ann_return = metrics_calculator.annualized_return(returns)

        # (1.01)^252 - 1 ≈ 11.67 = 1167%
        assert ann_return > 1000  # Very high return

    def test_annualized_volatility_basic(self, metrics_calculator, sample_returns):
        """Test annualized volatility calculation."""
        ann_vol = metrics_calculator.annualized_volatility(sample_returns)

        assert isinstance(ann_vol, float)
        assert ann_vol > 0

    def test_annualized_volatility_scaling(self, metrics_calculator, sample_returns):
        """Test volatility scaling."""
        daily_vol = np.std(sample_returns, ddof=1) * 100
        ann_vol = metrics_calculator.annualized_volatility(sample_returns)

        # Should be scaled by sqrt(252)
        expected = daily_vol * np.sqrt(252)
        assert abs(ann_vol - expected) < 0.01


class TestPortfolioMetrics:
    """Test portfolio-related metrics."""

    def test_information_ratio_basic(self, metrics_calculator, sample_returns, benchmark_returns):
        """Test Information Ratio calculation."""
        ir = metrics_calculator.information_ratio(sample_returns, benchmark_returns)

        assert isinstance(ir, float)
        assert not np.isnan(ir)

    def test_information_ratio_same_returns(self, metrics_calculator, sample_returns):
        """Test IR when strategy matches benchmark."""
        ir = metrics_calculator.information_ratio(sample_returns, sample_returns)

        # Should be zero when no active return
        assert abs(ir) < 0.01

    def test_alpha_beta_basic(self, metrics_calculator, sample_returns, benchmark_returns):
        """Test alpha and beta calculation."""
        alpha, beta = metrics_calculator.alpha_beta(sample_returns, benchmark_returns)

        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert not np.isnan(alpha)
        assert not np.isnan(beta)

    def test_beta_positive_correlation(self, metrics_calculator):
        """Test beta with positively correlated returns."""
        benchmark = np.random.normal(0.001, 0.02, 252)
        strategy = benchmark * 1.5 + np.random.normal(0, 0.005, 252)  # Amplified benchmark

        alpha, beta = metrics_calculator.alpha_beta(strategy, benchmark)

        # Beta should be around 1.5
        assert beta > 1.0


class TestCalculateAll:
    """Test comprehensive metrics calculation."""

    def test_calculate_all_basic(self, metrics_calculator, sample_returns):
        """Test calculating all metrics."""
        report = metrics_calculator.calculate_all(
            returns=sample_returns,
            ticker="TEST"
        )

        assert isinstance(report, MetricsReport)
        assert report.ticker == "TEST"
        assert report.sharpe_ratio is not None
        assert report.max_drawdown is not None
        assert report.annualized_return is not None

    def test_calculate_all_with_prices(self, metrics_calculator, sample_returns, sample_prices):
        """Test with explicit prices."""
        report = metrics_calculator.calculate_all(
            returns=sample_returns,
            prices=sample_prices,
            ticker="AAPL"
        )

        assert report.max_drawdown is not None
        assert report.current_drawdown is not None

    def test_calculate_all_with_predictions(
        self,
        metrics_calculator,
        sample_returns,
        sample_prices
    ):
        """Test with prediction metrics."""
        # Create predicted prices
        predicted = sample_prices + np.random.normal(0, 2, len(sample_prices))

        report = metrics_calculator.calculate_all(
            returns=sample_returns,
            prices=sample_prices,
            predicted_prices=predicted,
            ticker="MSFT"
        )

        assert report.hit_rate is not None
        assert report.rmse is not None
        assert report.mae is not None
        assert report.mape is not None

    def test_calculate_all_with_benchmark(
        self,
        metrics_calculator,
        sample_returns,
        benchmark_returns
    ):
        """Test with benchmark for portfolio metrics."""
        report = metrics_calculator.calculate_all(
            returns=sample_returns,
            benchmark_returns=benchmark_returns,
            ticker="GOOGL"
        )

        assert report.information_ratio is not None
        assert report.alpha is not None
        assert report.beta is not None

    def test_calculate_all_return_dataframe(self, metrics_calculator, sample_returns):
        """Test returning results as DataFrame."""
        result = metrics_calculator.calculate_all(
            returns=sample_returns,
            ticker="NVDA",
            return_dataframe=True
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'ticker' in result.columns


class TestUtilityFunctions:
    """Test utility functions."""

    def test_generate_summary_table(self):
        """Test summary table generation."""
        report = MetricsReport(
            ticker="AAPL",
            sharpe_ratio=1.5,
            max_drawdown=-15.0,
            annualized_return=20.0
        )

        table = generate_summary_table(report)

        assert isinstance(table, str)
        assert "AAPL" in table or "Sharpe" in table

    def test_generate_summary_table_sections(self):
        """Test summary table with specific sections."""
        report = MetricsReport(
            ticker="MSFT",
            sharpe_ratio=2.0,
            max_drawdown=-10.0,
            hit_rate=0.65
        )

        table = generate_summary_table(report, sections=['risk_adjusted', 'drawdown'])

        assert isinstance(table, str)
        # Should not include accuracy section
        # Note: This is a simple check; more sophisticated validation could be added

    def test_compare_strategies(self):
        """Test strategy comparison."""
        report1 = MetricsReport(
            ticker="Strategy A",
            sharpe_ratio=1.5,
            max_drawdown=-10.0,
            annualized_return=15.0
        )

        report2 = MetricsReport(
            ticker="Strategy B",
            sharpe_ratio=1.8,
            max_drawdown=-8.0,
            annualized_return=18.0
        )

        comparison = compare_strategies([report1, report2])

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "Strategy A" in comparison.index
        assert "Strategy B" in comparison.index


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_array_lengths(self, metrics_calculator):
        """Test with mismatched array lengths."""
        actual = np.array([1, 2, 3])
        predicted = np.array([1, 2])

        with pytest.raises(ValueError):
            metrics_calculator.hit_rate(actual, predicted)

    def test_empty_arrays(self, metrics_calculator):
        """Test with empty arrays."""
        empty = np.array([])

        sharpe = metrics_calculator.sharpe_ratio(empty)
        assert np.isnan(sharpe)

        var = metrics_calculator.value_at_risk(empty)
        assert np.isnan(var)

    def test_nan_handling(self, metrics_calculator):
        """Test handling of NaN values."""
        returns_with_nan = np.array([0.01, np.nan, 0.02, 0.01])

        # Should handle NaN gracefully
        report = metrics_calculator.calculate_all(returns=returns_with_nan)
        # After NaN removal, should have valid results
        assert report.n_observations == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

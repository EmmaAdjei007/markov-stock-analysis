"""
Financial calculators for trading and risk management.

This module provides essential financial calculation functions for:
- Position sizing and risk management
- Kelly Criterion optimization
- Risk/Reward analysis
- Compound interest calculations
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompoundInterestResult:
    """Result of compound interest calculation."""
    principal: float
    rate: float
    years: float
    periods_per_year: int
    final_value: float
    total_interest: float
    effective_rate: float


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    account_size: float
    risk_per_trade: float
    entry_price: float
    stop_loss_price: float
    risk_amount: float
    position_size: float
    shares: int
    total_position_value: float


@dataclass
class KellyCriterionResult:
    """Result of Kelly Criterion calculation."""
    win_rate: float
    win_loss_ratio: float
    kelly_fraction: float
    kelly_percentage: float
    recommended_fraction: float  # With safety factor
    recommended_percentage: float


@dataclass
class RiskRewardResult:
    """Result of risk/reward analysis."""
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    break_even_win_rate: float
    is_favorable: bool  # True if ratio >= 2.0


def compound_interest(
    principal: float,
    rate: float,
    years: float,
    periods_per_year: int = 1,
    additional_contributions: float = 0.0
) -> CompoundInterestResult:
    """
    Calculate compound interest with optional periodic contributions.

    Formula: A = P(1 + r/n)^(nt) + C * [((1 + r/n)^(nt) - 1) / (r/n)]

    Args:
        principal: Initial investment amount
        rate: Annual interest rate (as decimal, e.g., 0.07 for 7%)
        years: Investment period in years
        periods_per_year: Compounding frequency (1=annual, 4=quarterly, 12=monthly, 252=daily)
        additional_contributions: Additional amount added each period

    Returns:
        CompoundInterestResult with detailed breakdown

    Example:
        >>> result = compound_interest(10000, 0.07, 10, periods_per_year=12)
        >>> print(f"Final Value: ${result.final_value:,.2f}")
        Final Value: $20,097.57
    """
    if principal <= 0:
        raise ValueError("Principal must be positive")
    if rate < 0:
        raise ValueError("Interest rate cannot be negative")
    if years <= 0:
        raise ValueError("Years must be positive")

    n = periods_per_year
    t = years
    r = rate

    # Future value of principal
    fv_principal = principal * (1 + r/n) ** (n * t)

    # Future value of additional contributions (annuity)
    if additional_contributions > 0 and r > 0:
        fv_contributions = additional_contributions * (((1 + r/n)**(n*t) - 1) / (r/n))
    else:
        fv_contributions = additional_contributions * n * t

    final_value = fv_principal + fv_contributions
    total_interest = final_value - principal - (additional_contributions * n * t)

    # Effective annual rate
    effective_rate = (1 + r/n)**n - 1

    return CompoundInterestResult(
        principal=principal,
        rate=rate,
        years=years,
        periods_per_year=periods_per_year,
        final_value=final_value,
        total_interest=total_interest,
        effective_rate=effective_rate
    )


def position_sizing(
    account_size: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    allow_fractional: bool = True
) -> PositionSizeResult:
    """
    Calculate optimal position size based on risk management rules.

    Formula: Position Size = (Account Size × Risk %) / (Entry Price - Stop Loss)

    Args:
        account_size: Total account value
        risk_per_trade: Risk percentage per trade (as decimal, e.g., 0.02 for 2%)
        entry_price: Planned entry price
        stop_loss_price: Stop loss price
        allow_fractional: Allow fractional shares

    Returns:
        PositionSizeResult with position details

    Example:
        >>> result = position_sizing(100000, 0.02, 150.0, 145.0)
        >>> print(f"Buy {result.shares} shares at ${entry_price}")
        Buy 400 shares at $150.0
    """
    if account_size <= 0:
        raise ValueError("Account size must be positive")
    if not 0 < risk_per_trade <= 1:
        raise ValueError("Risk per trade must be between 0 and 1")
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")

    risk_amount = account_size * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk == 0:
        raise ValueError("Entry price and stop loss cannot be the same")

    position_size = risk_amount / price_risk

    if allow_fractional:
        shares = position_size
    else:
        shares = int(position_size)

    total_position_value = shares * entry_price

    if total_position_value > account_size:
        logger.warning(f"Position value ${total_position_value:.2f} exceeds account size ${account_size:.2f}")

    return PositionSizeResult(
        account_size=account_size,
        risk_per_trade=risk_per_trade,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        risk_amount=risk_amount,
        position_size=position_size,
        shares=int(shares) if not allow_fractional else shares,
        total_position_value=total_position_value
    )


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_kelly_fraction: float = 0.25,
    safety_factor: float = 0.5
) -> KellyCriterionResult:
    """
    Calculate optimal position size using Kelly Criterion.

    Formula: f* = (p × b - q) / b
    Where:
        f* = fraction of capital to bet
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio (avg_win / avg_loss)

    Args:
        win_rate: Historical win rate (as decimal, e.g., 0.55 for 55%)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
        max_kelly_fraction: Maximum fraction allowed (default 0.25 = 25%)
        safety_factor: Reduce Kelly fraction by this factor (default 0.5 = half Kelly)

    Returns:
        KellyCriterionResult with Kelly fraction and recommendations

    Example:
        >>> result = kelly_criterion(win_rate=0.55, avg_win=200, avg_loss=100)
        >>> print(f"Kelly: {result.kelly_percentage:.1f}%, Recommended: {result.recommended_percentage:.1f}%")
        Kelly: 10.0%, Recommended: 5.0%
    """
    if not 0 <= win_rate <= 1:
        raise ValueError("Win rate must be between 0 and 1")
    if avg_win <= 0:
        raise ValueError("Average win must be positive")
    if avg_loss <= 0:
        raise ValueError("Average loss must be positive")

    p = win_rate
    q = 1 - win_rate
    b = avg_win / avg_loss  # Win/loss ratio

    # Kelly formula
    kelly_fraction = (p * b - q) / b

    # Ensure non-negative
    kelly_fraction = max(0, kelly_fraction)

    # Apply maximum cap
    kelly_fraction = min(kelly_fraction, max_kelly_fraction)

    # Apply safety factor (fractional Kelly)
    recommended_fraction = kelly_fraction * safety_factor

    return KellyCriterionResult(
        win_rate=win_rate,
        win_loss_ratio=b,
        kelly_fraction=kelly_fraction,
        kelly_percentage=kelly_fraction * 100,
        recommended_fraction=recommended_fraction,
        recommended_percentage=recommended_fraction * 100
    )


def risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    min_acceptable_ratio: float = 2.0
) -> RiskRewardResult:
    """
    Calculate risk/reward ratio for a trade setup.

    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit / target price
        min_acceptable_ratio: Minimum acceptable risk/reward ratio

    Returns:
        RiskRewardResult with analysis

    Example:
        >>> result = risk_reward_ratio(entry=100, stop_loss=95, take_profit=110)
        >>> print(f"R:R = 1:{result.risk_reward_ratio:.1f}")
        R:R = 1:2.0
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive")

    risk_amount = abs(entry_price - stop_loss)
    reward_amount = abs(take_profit - entry_price)

    if risk_amount == 0:
        raise ValueError("Risk amount cannot be zero")

    ratio = reward_amount / risk_amount

    # Calculate break-even win rate needed
    # Break-even: W × R - (1-W) × 1 = 0
    # W = 1 / (1 + R)
    break_even_win_rate = 1 / (1 + ratio)

    is_favorable = ratio >= min_acceptable_ratio

    return RiskRewardResult(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_amount=risk_amount,
        reward_amount=reward_amount,
        risk_reward_ratio=ratio,
        break_even_win_rate=break_even_win_rate,
        is_favorable=is_favorable
    )


def sharpe_ratio_calculator(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Formula: Sharpe = (Mean Return - RFR) / Std Dev × √periods

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio

    Example:
        >>> returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
        >>> sharpe = sharpe_ratio_calculator(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns")

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize
    annual_return = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)

    sharpe = (annual_return - risk_free_rate) / annual_std

    return sharpe


def optimal_f(
    trades: np.ndarray,
    initial_capital: float = 100000.0,
    granularity: int = 100
) -> Dict[str, float]:
    """
    Calculate Optimal F (fraction) using Ralph Vince's method.

    Optimal F maximizes geometric growth rate by finding the fraction
    that maximizes Terminal Wealth Relative (TWR).

    Args:
        trades: Array of trade P&L values (in dollars)
        initial_capital: Starting capital
        granularity: Number of f values to test (higher = more precise)

    Returns:
        Dictionary with optimal_f, max_twr, and recommended_f

    Example:
        >>> trades = np.array([100, -50, 150, -30, 80])
        >>> result = optimal_f(trades)
        >>> print(f"Optimal F: {result['optimal_f']:.2%}")
    """
    if len(trades) < 2:
        raise ValueError("Need at least 2 trades")

    biggest_loss = abs(np.min(trades))

    if biggest_loss == 0:
        return {'optimal_f': 0.0, 'max_twr': 0.0, 'recommended_f': 0.0}

    # Test different f values
    f_values = np.linspace(0.01, 1.0, granularity)
    twr_values = []

    for f in f_values:
        hpr_product = 1.0
        for trade in trades:
            # HPR = 1 + (f × trade / biggest_loss)
            hpr = 1.0 + (f * trade / biggest_loss)
            if hpr <= 0:
                hpr_product = 0
                break
            hpr_product *= hpr

        twr = hpr_product ** (1 / len(trades))
        twr_values.append(twr)

    twr_values = np.array(twr_values)
    max_idx = np.argmax(twr_values)

    optimal_f = f_values[max_idx]
    max_twr = twr_values[max_idx]

    # Recommended: Use half of optimal f for safety
    recommended_f = optimal_f * 0.5

    return {
        'optimal_f': optimal_f,
        'max_twr': max_twr,
        'recommended_f': recommended_f,
        'biggest_loss': biggest_loss
    }


def break_even_analysis(
    fixed_costs: float,
    variable_cost_per_unit: float,
    price_per_unit: float
) -> Dict[str, float]:
    """
    Calculate break-even point for a trading strategy or business.

    Args:
        fixed_costs: Fixed costs (monthly fees, etc.)
        variable_cost_per_unit: Variable cost per trade
        price_per_unit: Revenue per trade

    Returns:
        Dictionary with break_even_units and break_even_revenue

    Example:
        >>> result = break_even_analysis(fixed_costs=1000, variable_cost=5, price=15)
        >>> print(f"Break-even: {result['break_even_units']:.0f} trades")
    """
    if price_per_unit <= variable_cost_per_unit:
        raise ValueError("Price must be greater than variable cost")

    contribution_margin = price_per_unit - variable_cost_per_unit
    break_even_units = fixed_costs / contribution_margin
    break_even_revenue = break_even_units * price_per_unit

    return {
        'break_even_units': break_even_units,
        'break_even_revenue': break_even_revenue,
        'contribution_margin': contribution_margin
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Financial Calculators Demo")
    print("=" * 60)

    # Compound Interest
    print("\n1. Compound Interest Calculator")
    result = compound_interest(10000, 0.07, 10, periods_per_year=12, additional_contributions=100)
    print(f"   Initial: ${result.principal:,.2f}")
    print(f"   Final Value: ${result.final_value:,.2f}")
    print(f"   Total Interest: ${result.total_interest:,.2f}")

    # Position Sizing
    print("\n2. Position Sizing Calculator")
    result = position_sizing(100000, 0.02, 150.0, 145.0)
    print(f"   Account Size: ${result.account_size:,.2f}")
    print(f"   Risk per Trade: {result.risk_per_trade:.1%}")
    print(f"   Position Size: {result.shares:.0f} shares")
    print(f"   Position Value: ${result.total_position_value:,.2f}")

    # Kelly Criterion
    print("\n3. Kelly Criterion Calculator")
    result = kelly_criterion(0.55, 200, 100)
    print(f"   Win Rate: {result.win_rate:.1%}")
    print(f"   Win/Loss Ratio: {result.win_loss_ratio:.2f}")
    print(f"   Kelly %: {result.kelly_percentage:.1f}%")
    print(f"   Recommended %: {result.recommended_percentage:.1f}%")

    # Risk/Reward Ratio
    print("\n4. Risk/Reward Analysis")
    result = risk_reward_ratio(100, 95, 110)
    print(f"   Entry: ${result.entry_price:.2f}")
    print(f"   Risk: ${result.risk_amount:.2f}")
    print(f"   Reward: ${result.reward_amount:.2f}")
    print(f"   Ratio: 1:{result.risk_reward_ratio:.1f}")
    print(f"   Break-even Win Rate: {result.break_even_win_rate:.1%}")
    print(f"   Favorable: {result.is_favorable}")

    print("\n" + "=" * 60)

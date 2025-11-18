"""
Enhanced data preprocessor with multiple state assignment strategies.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StateAssignmentMethod(Enum):
    """Available methods for state assignment."""
    QUANTILE = "quantile"  # Equal-sized bins based on distribution
    VOLATILITY_ADJUSTED = "volatility"  # Consider volatility regimes
    KMEANS = "kmeans"  # K-means clustering
    FIXED_THRESHOLD = "fixed"  # Fixed percentage thresholds


@dataclass
class StateInfo:
    """Information about state assignments."""
    n_states: int
    method: str
    labels: List[str]
    boundaries: Optional[np.ndarray] = None
    statistics: Optional[Dict] = None


class DataPreprocessor:
    """
    Enhanced data preprocessor with multiple state assignment strategies.

    Features:
        - Multiple state assignment methods
        - Volatility regime detection
        - Technical indicator integration
        - Robust data validation
        - Automatic state labeling

    Example:
        >>> preprocessor = DataPreprocessor(n_states=5)
        >>> processed = preprocessor.process(data, method="quantile")
        >>> state_info = preprocessor.get_state_info()
    """

    def __init__(
        self,
        n_states: int = 5,
        use_log_returns: bool = False,
        output_dir: str = "data/processed"
    ):
        """
        Initialize the data preprocessor.

        Args:
            n_states: Number of discrete states
            use_log_returns: Use log returns instead of percentage returns
            output_dir: Directory for saving processed data
        """
        self.n_states = n_states
        self.use_log_returns = use_log_returns
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State labels based on number of states
        self.state_labels = self._get_state_labels(n_states)
        self.state_info: Optional[StateInfo] = None

    def _get_state_labels(self, n_states: int) -> List[str]:
        """
        Generate appropriate state labels based on number of states.

        Args:
            n_states: Number of states

        Returns:
            List of state labels
        """
        labels_map = {
            3: ["Bearish", "Neutral", "Bullish"],
            5: ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"],
            7: ["Extreme Bear", "Strong Bear", "Bear", "Neutral", "Bull", "Strong Bull", "Extreme Bull"],
        }

        if n_states in labels_map:
            return labels_map[n_states]
        else:
            # Generate generic labels
            return [f"State_{i}" for i in range(n_states)]

    def calculate_returns(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close'
    ) -> pd.DataFrame:
        """
        Calculate daily returns with proper handling.

        Args:
            df: Input DataFrame with price data
            price_column: Column name for prices

        Returns:
            DataFrame with 'Daily_Return' column added
        """
        df = df.copy()

        if self.use_log_returns:
            df['Daily_Return'] = np.log(df[price_column] / df[price_column].shift(1)) * 100
        else:
            df['Daily_Return'] = df[price_column].pct_change() * 100

        return df

    def assign_states_quantile(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, StateInfo]:
        """
        Assign states using quantile-based discretization.

        Args:
            df: DataFrame with 'Daily_Return' column

        Returns:
            Tuple of (processed DataFrame, state info)
        """
        df = df.copy()

        # Calculate quantiles
        try:
            df['State'] = pd.qcut(
                df['Daily_Return'],
                q=self.n_states,
                labels=False,
                duplicates='drop'
            )
        except ValueError as e:
            logger.warning(f"Quantile assignment failed, using fallback: {e}")
            # Fallback to equal-width binning
            df['State'] = pd.cut(
                df['Daily_Return'],
                bins=self.n_states,
                labels=False
            )

        # Add state descriptions
        df['State_Description'] = df['State'].apply(
            lambda x: self.state_labels[int(x)] if pd.notna(x) else "Unknown"
        )

        # Calculate state boundaries
        boundaries = []
        for i in range(self.n_states):
            state_returns = df[df['State'] == i]['Daily_Return']
            if len(state_returns) > 0:
                boundaries.append((state_returns.min(), state_returns.max()))
            else:
                boundaries.append((np.nan, np.nan))

        # Create state info
        state_info = StateInfo(
            n_states=self.n_states,
            method="quantile",
            labels=self.state_labels,
            boundaries=np.array(boundaries),
            statistics={
                "state_counts": df['State'].value_counts().to_dict(),
                "state_mean_returns": df.groupby('State')['Daily_Return'].mean().to_dict()
            }
        )

        return df, state_info

    def assign_states_volatility(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[pd.DataFrame, StateInfo]:
        """
        Assign states considering volatility regimes.

        Args:
            df: DataFrame with 'Daily_Return' column
            window: Rolling window for volatility calculation

        Returns:
            Tuple of (processed DataFrame, state info)
        """
        df = df.copy()

        # Calculate rolling volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=window).std()

        # Normalize returns by volatility
        df['Normalized_Return'] = df['Daily_Return'] / (df['Volatility'] + 1e-6)

        # Assign states based on normalized returns
        try:
            df['State'] = pd.qcut(
                df['Normalized_Return'],
                q=self.n_states,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            df['State'] = pd.cut(
                df['Normalized_Return'],
                bins=self.n_states,
                labels=False
            )

        df['State_Description'] = df['State'].apply(
            lambda x: self.state_labels[int(x)] if pd.notna(x) else "Unknown"
        )

        state_info = StateInfo(
            n_states=self.n_states,
            method="volatility_adjusted",
            labels=self.state_labels,
            statistics={
                "state_counts": df['State'].value_counts().to_dict(),
                "mean_volatility_by_state": df.groupby('State')['Volatility'].mean().to_dict()
            }
        )

        return df, state_info

    def assign_states_fixed(
        self,
        df: pd.DataFrame,
        thresholds: Optional[List[float]] = None
    ) -> Tuple[pd.DataFrame, StateInfo]:
        """
        Assign states using fixed percentage thresholds.

        Args:
            df: DataFrame with 'Daily_Return' column
            thresholds: Custom thresholds (default: [-2, -0.5, 0.5, 2] for 5 states)

        Returns:
            Tuple of (processed DataFrame, state info)
        """
        df = df.copy()

        # Default thresholds for 5 states
        if thresholds is None:
            if self.n_states == 3:
                thresholds = [-0.5, 0.5]
            elif self.n_states == 5:
                thresholds = [-2.0, -0.5, 0.5, 2.0]
            elif self.n_states == 7:
                thresholds = [-3.0, -1.5, -0.5, 0.5, 1.5, 3.0]
            else:
                # Generate evenly spaced thresholds
                step = 4.0 / self.n_states
                thresholds = [i * step - 2.0 for i in range(1, self.n_states)]

        # Assign states based on thresholds
        df['State'] = pd.cut(
            df['Daily_Return'],
            bins=[-np.inf] + thresholds + [np.inf],
            labels=False
        )

        df['State_Description'] = df['State'].apply(
            lambda x: self.state_labels[int(x)] if pd.notna(x) else "Unknown"
        )

        state_info = StateInfo(
            n_states=self.n_states,
            method="fixed_threshold",
            labels=self.state_labels,
            boundaries=np.array(thresholds),
            statistics={
                "thresholds": thresholds,
                "state_counts": df['State'].value_counts().to_dict()
            }
        )

        return df, state_info

    def add_technical_indicators(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add common technical indicators to the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()

        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential moving average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        df['RSI'] = 100 - (100 / (1 + rs))

        return df

    def process(
        self,
        data: Dict[str, pd.DataFrame],
        method: Literal["quantile", "volatility", "fixed"] = "quantile",
        add_indicators: bool = False,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process multiple tickers with state assignment.

        Args:
            data: Dictionary of ticker -> DataFrame
            method: State assignment method
            add_indicators: Add technical indicators
            save: Save processed data to disk

        Returns:
            Dictionary of processed DataFrames
        """
        processed_data = {}

        for ticker, df in data.items():
            logger.info(f"Processing {ticker} with method '{method}'")

            # Calculate returns
            df = self.calculate_returns(df)

            # Remove NA values
            df = df.dropna()

            # Assign states based on method
            if method == "quantile":
                df, state_info = self.assign_states_quantile(df)
            elif method == "volatility":
                df, state_info = self.assign_states_volatility(df)
            elif method == "fixed":
                df, state_info = self.assign_states_fixed(df)
            else:
                raise ValueError(f"Unknown method: {method}")

            self.state_info = state_info

            # Add technical indicators if requested
            if add_indicators:
                df = self.add_technical_indicators(df)

            # Save processed data
            if save:
                output_path = self.output_dir / f"{ticker}_processed.csv"
                df.to_csv(output_path)
                logger.info(f"Saved processed data to {output_path}")

            processed_data[ticker] = df

        return processed_data

    def load_processed(self, ticker: str) -> pd.DataFrame:
        """
        Load previously processed data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Processed DataFrame

        Raises:
            FileNotFoundError: If processed data doesn't exist
        """
        file_path = self.output_dir / f"{ticker}_processed.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Processed data not found: {file_path}")

        return pd.read_csv(file_path, index_col='Date', parse_dates=True)

    def get_state_info(self) -> Optional[StateInfo]:
        """Get information about the latest state assignment."""
        return self.state_info

    def get_returns_by_state(self, df: pd.DataFrame) -> List[np.ndarray]:
        """
        Get historical returns grouped by state.

        Args:
            df: Processed DataFrame with State column

        Returns:
            List of arrays, each containing returns for that state
        """
        returns_by_state = []

        for state in range(self.n_states):
            state_returns = df[df['State'] == state]['Daily_Return'].values
            returns_by_state.append(state_returns)

        return returns_by_state


# Backward compatibility function
def assign_states(
    data: Dict[str, pd.DataFrame],
    n_states: int = 5,
    use_log_returns: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Legacy function for backward compatibility.

    Use DataPreprocessor class for new code.
    """
    preprocessor = DataPreprocessor(n_states=n_states, use_log_returns=use_log_returns)
    return preprocessor.process(data, method="quantile")

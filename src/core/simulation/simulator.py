"""
Enhanced Monte Carlo simulator with parallel processing and optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not available, simulations will be slower")
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@dataclass
class SimulationResult:
    """Container for simulation results."""
    paths: np.ndarray  # Shape: (n_simulations, n_days + 1)
    initial_price: float
    n_simulations: int
    n_days: int
    median_path: np.ndarray
    mean_path: np.ndarray
    percentiles: Dict[int, np.ndarray]  # e.g., {5: path, 25: path, ...}
    final_prices: np.ndarray
    metadata: Dict


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _simulate_single_path_numba(
    initial_price: float,
    transition_matrix: np.ndarray,
    returns_arrays: List[np.ndarray],
    state_sizes: np.ndarray,
    days: int,
    volatility: float,
    init_state: int,
    random_states: np.ndarray,
    random_returns: np.ndarray,
    random_noise: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized single path simulation.

    Args:
        initial_price: Starting price
        transition_matrix: State transition probabilities
        returns_arrays: Flattened returns for each state
        state_sizes: Number of returns per state
        days: Simulation days
        volatility: Gaussian noise std dev
        init_state: Initial state
        random_states: Pre-generated random numbers for state transitions
        random_returns: Pre-generated random numbers for return selection
        random_noise: Pre-generated random noise

    Returns:
        Price path array
    """
    n_states = transition_matrix.shape[0]
    prices = np.empty(days + 1)
    prices[0] = initial_price

    current_state = init_state

    # Calculate cumulative sizes for indexing into flattened returns
    cumsum_sizes = np.cumsum(state_sizes)

    for day in range(days):
        # Determine next state using cumulative probabilities
        probs = transition_matrix[current_state]
        cumsum_probs = np.cumsum(probs)
        next_state = np.searchsorted(cumsum_probs, random_states[day])

        # Get return for this state
        if next_state == 0:
            start_idx = 0
        else:
            start_idx = cumsum_sizes[next_state - 1]
        end_idx = cumsum_sizes[next_state]

        state_returns = returns_arrays[start_idx:end_idx]

        if len(state_returns) > 0:
            ret_idx = int(random_returns[day] * len(state_returns))
            if ret_idx >= len(state_returns):
                ret_idx = len(state_returns) - 1
            return_pct = state_returns[ret_idx]
        else:
            return_pct = 0.0

        # Add noise and calculate next price
        noise = random_noise[day] * volatility
        next_price = prices[day] * (1.0 + (return_pct + noise) / 100.0)
        prices[day + 1] = next_price

        current_state = next_state

    return prices


class MonteCarloSimulator:
    """
    Enhanced Monte Carlo simulator with parallel processing.

    Features:
        - Numba JIT compilation for speed (if available)
        - Parallel processing for multiple simulations
        - Configurable initial state and volatility
        - Automatic percentile calculation
        - Memory-efficient batch processing

    Example:
        >>> simulator = MonteCarloSimulator(n_simulations=1000, n_days=30)
        >>> result = simulator.run(
        ...     initial_price=150.0,
        ...     transition_matrix=transition_matrix,
        ...     returns_by_state=returns
        ... )
        >>> print(f"Median final price: ${result.median_path[-1]:.2f}")
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        n_days: int = 30,
        volatility: float = 0.01,
        init_state: Optional[int] = None,
        seed: Optional[int] = None,
        use_parallel: bool = True,
        n_jobs: int = -1
    ):
        """
        Initialize the simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            n_days: Number of days to simulate
            volatility: Gaussian noise standard deviation (in percentage points)
            init_state: Initial state (None for random)
            seed: Random seed for reproducibility
            use_parallel: Use parallel processing
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.n_simulations = n_simulations
        self.n_days = n_days
        self.volatility = volatility
        self.init_state = init_state
        self.seed = seed
        self.use_parallel = use_parallel and n_simulations >= 100
        self.n_jobs = n_jobs

        self.rng = np.random.default_rng(seed)

    def _prepare_returns_for_numba(
        self,
        returns_by_state: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten returns list for Numba compatibility.

        Returns:
            (flattened_returns, state_sizes)
        """
        state_sizes = np.array([len(ret) for ret in returns_by_state])
        flattened = np.concatenate([np.asarray(ret) for ret in returns_by_state])
        return flattened, state_sizes

    def _simulate_batch(
        self,
        batch_size: int,
        initial_price: float,
        transition_matrix: np.ndarray,
        returns_by_state: List[np.ndarray],
        n_states: int
    ) -> np.ndarray:
        """Simulate a batch of paths."""
        paths = np.empty((batch_size, self.n_days + 1))

        if HAS_NUMBA:
            # Use optimized Numba version
            returns_flat, state_sizes = self._prepare_returns_for_numba(returns_by_state)

            for i in range(batch_size):
                # Pre-generate all random numbers
                random_states = self.rng.random(self.n_days)
                random_returns = self.rng.random(self.n_days)
                random_noise = self.rng.standard_normal(self.n_days)

                init_state = self.init_state if self.init_state is not None else self.rng.integers(0, n_states)

                paths[i] = _simulate_single_path_numba(
                    initial_price,
                    transition_matrix,
                    returns_flat,
                    state_sizes,
                    self.n_days,
                    self.volatility,
                    init_state,
                    random_states,
                    random_returns,
                    random_noise
                )
        else:
            # Use standard numpy version
            for i in range(batch_size):
                paths[i] = self._simulate_single_path_numpy(
                    initial_price,
                    transition_matrix,
                    returns_by_state,
                    n_states
                )

        return paths

    def _simulate_single_path_numpy(
        self,
        initial_price: float,
        transition_matrix: np.ndarray,
        returns_by_state: List[np.ndarray],
        n_states: int
    ) -> np.ndarray:
        """Standard numpy simulation (fallback when Numba unavailable)."""
        prices = np.empty(self.n_days + 1)
        prices[0] = initial_price

        if self.init_state is not None:
            current_state = self.init_state
        else:
            current_state = self.rng.integers(0, n_states)

        for day in range(self.n_days):
            # Sample next state
            probs = transition_matrix[current_state]
            probs = probs / probs.sum()  # Normalize
            next_state = self.rng.choice(n_states, p=probs)

            # Sample return
            state_returns = returns_by_state[next_state]
            if len(state_returns) > 0:
                return_pct = self.rng.choice(state_returns)
            else:
                return_pct = 0.0

            # Add noise
            noise = self.rng.normal(0, self.volatility)
            next_price = prices[day] * (1.0 + (return_pct + noise) / 100.0)
            prices[day + 1] = max(next_price, 0.0)  # Prevent negative prices

            current_state = next_state

        return prices

    def run(
        self,
        initial_price: float,
        transition_matrix: np.ndarray,
        returns_by_state: List[np.ndarray],
        percentiles: List[int] = [5, 25, 50, 75, 95]
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            initial_price: Starting price
            transition_matrix: State transition probability matrix
            returns_by_state: List of return arrays for each state
            percentiles: Percentiles to calculate

        Returns:
            SimulationResult containing all simulation data

        Example:
            >>> result = simulator.run(150.0, trans_matrix, returns)
            >>> print(f"95th percentile: ${result.percentiles[95][-1]:.2f}")
        """
        n_states = transition_matrix.shape[0]

        logger.info(f"Starting {self.n_simulations} simulations over {self.n_days} days")

        if self.use_parallel:
            # Parallel execution
            n_workers = self.n_jobs if self.n_jobs > 0 else None
            batch_size = max(1, self.n_simulations // (n_workers or 4))

            paths_list = []

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                remaining = self.n_simulations

                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    future = executor.submit(
                        self._simulate_batch,
                        current_batch,
                        initial_price,
                        transition_matrix,
                        returns_by_state,
                        n_states
                    )
                    futures.append(future)
                    remaining -= current_batch

                for future in as_completed(futures):
                    paths_list.append(future.result())

            all_paths = np.vstack(paths_list)
        else:
            # Sequential execution
            all_paths = self._simulate_batch(
                self.n_simulations,
                initial_price,
                transition_matrix,
                returns_by_state,
                n_states
            )

        # Calculate statistics
        median_path = np.median(all_paths, axis=0)
        mean_path = np.mean(all_paths, axis=0)

        percentile_paths = {}
        for p in percentiles:
            percentile_paths[p] = np.percentile(all_paths, p, axis=0)

        final_prices = all_paths[:, -1]

        metadata = {
            'volatility': self.volatility,
            'init_state': self.init_state,
            'seed': self.seed,
            'used_numba': HAS_NUMBA,
            'parallel': self.use_parallel
        }

        result = SimulationResult(
            paths=all_paths,
            initial_price=initial_price,
            n_simulations=self.n_simulations,
            n_days=self.n_days,
            median_path=median_path,
            mean_path=mean_path,
            percentiles=percentile_paths,
            final_prices=final_prices,
            metadata=metadata
        )

        logger.info(f"Simulation complete. Final price range: ${final_prices.min():.2f} - ${final_prices.max():.2f}")

        return result


def simulate_price_path(
    initial_price: float,
    transition_matrix: np.ndarray,
    returns_by_state: List[np.ndarray],
    days: int,
    volatility: float = 0.01,
    init_state: int = None,
    rng: object = None
) -> np.ndarray:
    """
    Legacy function for backward compatibility.

    Use MonteCarloSimulator class for new code.
    """
    if rng is None:
        rng = np.random.default_rng()

    simulator = MonteCarloSimulator(
        n_simulations=1,
        n_days=days,
        volatility=volatility,
        init_state=init_state
    )

    # Use the numpy version directly for single path
    n_states = transition_matrix.shape[0]
    return simulator._simulate_single_path_numpy(
        initial_price,
        transition_matrix,
        returns_by_state,
        n_states
    )

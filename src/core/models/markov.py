"""
Enhanced Markov chain models with higher-order and ensemble support.
"""

import numpy as np
from scipy.stats import entropy
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MarkovModelInfo:
    """Information about a trained Markov model."""
    order: int
    n_states: int
    smoothing_alpha: float
    entropy_mean: float
    entropy_std: float
    most_stable_state: int
    most_volatile_state: int


class MarkovChain:
    """
    First-order Markov chain with enhanced features.

    Features:
        - Laplace smoothing for sparse transitions
        - Entropy calculation for uncertainty quantification
        - State stability analysis
        - Stationary distribution calculation
        - Model persistence

    Example:
        >>> model = MarkovChain(n_states=5, alpha=0.001)
        >>> model.fit(states)
        >>> next_state_probs = model.predict(current_state=2)
        >>> model_info = model.get_model_info()
    """

    def __init__(
        self,
        n_states: int = 5,
        alpha: float = 1e-3
    ):
        """
        Initialize Markov chain model.

        Args:
            n_states: Number of discrete states
            alpha: Smoothing parameter (Laplace smoothing)
        """
        self.n_states = n_states
        self.alpha = alpha
        self.transition_matrix: Optional[np.ndarray] = None
        self.row_entropy: Optional[np.ndarray] = None
        self.model_info: Optional[MarkovModelInfo] = None

    def fit(self, states: np.ndarray) -> 'MarkovChain':
        """
        Train the Markov model on observed state sequence.

        Args:
            states: 1D array of integer state labels (0 to n_states-1)

        Returns:
            self for method chaining

        Raises:
            ValueError: If states contain invalid values
        """
        states = np.asarray(states, dtype=int)

        # Validate states
        if states.min() < 0 or states.max() >= self.n_states:
            raise ValueError(
                f"States must be in range [0, {self.n_states-1}], "
                f"got range [{states.min()}, {states.max()}]"
            )

        # Build transition count matrix
        transitions = np.zeros((self.n_states, self.n_states), dtype=float)

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transitions[current_state][next_state] += 1

        # Apply Laplace smoothing
        if self.alpha > 0:
            transitions += self.alpha

        # Normalize to probabilities
        row_sums = transitions.sum(axis=1)

        # Handle zero-sum rows (shouldn't happen with smoothing, but be safe)
        zero_rows = row_sums == 0
        if zero_rows.any():
            logger.warning(f"Found {zero_rows.sum()} empty states, using uniform distribution")
            transitions[zero_rows] = 1.0
            row_sums[zero_rows] = self.n_states

        self.transition_matrix = transitions / row_sums[:, np.newaxis]

        # Calculate entropy for each state
        self.row_entropy = np.array([entropy(row) for row in self.transition_matrix])

        # Generate model info
        self._generate_model_info()

        return self

    def predict(
        self,
        current_state: int,
        n_steps: int = 1
    ) -> np.ndarray:
        """
        Predict state distribution after n steps.

        Args:
            current_state: Current state (0 to n_states-1)
            n_steps: Number of steps ahead to predict

        Returns:
            Probability distribution over states

        Raises:
            ValueError: If model not fitted or invalid state
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if not 0 <= current_state < self.n_states:
            raise ValueError(f"Invalid state: {current_state}")

        # Start with one-hot encoded current state
        state_dist = np.zeros(self.n_states)
        state_dist[current_state] = 1.0

        # Apply transition matrix n times
        for _ in range(n_steps):
            state_dist = state_dist @ self.transition_matrix

        return state_dist

    def sample_next_state(
        self,
        current_state: int,
        rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Sample next state given current state.

        Args:
            current_state: Current state
            rng: Random number generator

        Returns:
            Sampled next state
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.transition_matrix[current_state]
        return int(rng.choice(self.n_states, p=probs))

    def get_stationary_distribution(
        self,
        method: str = 'eigenvalue',
        max_iter: int = 1000
    ) -> np.ndarray:
        """
        Calculate stationary distribution of the Markov chain.

        Args:
            method: 'eigenvalue' or 'power_iteration'
            max_iter: Maximum iterations for power iteration

        Returns:
            Stationary distribution
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted")

        if method == 'eigenvalue':
            # Find eigenvector with eigenvalue 1
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / stationary.sum()
        else:
            # Power iteration
            state_dist = np.ones(self.n_states) / self.n_states
            for _ in range(max_iter):
                new_dist = state_dist @ self.transition_matrix
                if np.allclose(new_dist, state_dist):
                    break
                state_dist = new_dist
            stationary = state_dist

        return np.abs(stationary)  # Ensure positive

    def _generate_model_info(self) -> None:
        """Generate model information for analysis."""
        most_stable = int(np.argmin(self.row_entropy))
        most_volatile = int(np.argmax(self.row_entropy))

        self.model_info = MarkovModelInfo(
            order=1,
            n_states=self.n_states,
            smoothing_alpha=self.alpha,
            entropy_mean=float(np.mean(self.row_entropy)),
            entropy_std=float(np.std(self.row_entropy)),
            most_stable_state=most_stable,
            most_volatile_state=most_volatile
        )

    def get_model_info(self) -> Optional[MarkovModelInfo]:
        """Get model information."""
        return self.model_info

    def get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix."""
        if self.transition_matrix is None:
            raise ValueError("Model not fitted")
        return self.transition_matrix.copy()

    def get_entropy(self) -> np.ndarray:
        """Get entropy for each state."""
        if self.row_entropy is None:
            raise ValueError("Model not fitted")
        return self.row_entropy.copy()


class HigherOrderMarkov:
    """
    Higher-order Markov chain (2nd or 3rd order).

    In a k-th order Markov chain, the next state depends on the previous k states.
    This can capture longer-term dependencies in the data.

    Example:
        >>> model = HigherOrderMarkov(n_states=5, order=2)
        >>> model.fit(states)
        >>> next_state = model.sample_next_state(history=[2, 3])
    """

    def __init__(
        self,
        n_states: int = 5,
        order: int = 2,
        alpha: float = 1e-3
    ):
        """
        Initialize higher-order Markov model.

        Args:
            n_states: Number of discrete states
            order: Order of the Markov chain (2 or 3)
            alpha: Smoothing parameter
        """
        if order < 2:
            raise ValueError("Use MarkovChain class for first-order models")
        if order > 3:
            logger.warning("Orders > 3 may require excessive memory")

        self.n_states = n_states
        self.order = order
        self.alpha = alpha

        # Transition dictionary: tuple of states -> probability distribution
        self.transitions: Dict[tuple, np.ndarray] = {}

    def _get_state_sequences(self, states: np.ndarray) -> List[Tuple]:
        """Extract all k-length state sequences from data."""
        sequences = []
        for i in range(len(states) - self.order):
            seq = tuple(states[i:i + self.order])
            next_state = states[i + self.order]
            sequences.append((seq, next_state))
        return sequences

    def fit(self, states: np.ndarray) -> 'HigherOrderMarkov':
        """
        Train the higher-order Markov model.

        Args:
            states: 1D array of integer state labels

        Returns:
            self for method chaining
        """
        states = np.asarray(states, dtype=int)

        # Build transition counts
        counts = defaultdict(lambda: np.zeros(self.n_states))

        sequences = self._get_state_sequences(states)

        for history, next_state in sequences:
            counts[history][next_state] += 1

        # Apply smoothing and normalize
        for history in counts:
            count_array = counts[history] + self.alpha
            self.transitions[history] = count_array / count_array.sum()

        logger.info(f"Fitted {self.order}-order Markov model with {len(self.transitions)} unique histories")

        return self

    def predict(
        self,
        history: List[int],
        n_steps: int = 1
    ) -> np.ndarray:
        """
        Predict state distribution given history.

        Args:
            history: List of previous states (length = order)
            n_steps: Steps ahead (only 1 supported for now)

        Returns:
            Probability distribution over next states
        """
        if len(history) != self.order:
            raise ValueError(f"History must have length {self.order}")

        history_tuple = tuple(history)

        if history_tuple in self.transitions:
            return self.transitions[history_tuple].copy()
        else:
            # Unseen history: use uniform distribution
            logger.warning(f"Unseen history {history_tuple}, using uniform distribution")
            return np.ones(self.n_states) / self.n_states

    def sample_next_state(
        self,
        history: List[int],
        rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Sample next state given history.

        Args:
            history: List of previous states
            rng: Random number generator

        Returns:
            Sampled next state
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.predict(history)
        return int(rng.choice(self.n_states, p=probs))


class EnsembleMarkov:
    """
    Ensemble of multiple Markov models with different orders.

    Combines predictions from first, second, and third-order models
    using weighted averaging.

    Example:
        >>> ensemble = EnsembleMarkov(n_states=5, orders=[1, 2])
        >>> ensemble.fit(states)
        >>> next_state = ensemble.sample_next_state([2, 3])
    """

    def __init__(
        self,
        n_states: int = 5,
        orders: List[int] = [1, 2],
        weights: Optional[List[float]] = None,
        alpha: float = 1e-3
    ):
        """
        Initialize ensemble model.

        Args:
            n_states: Number of states
            orders: List of model orders to include
            weights: Weights for each model (default: equal weights)
            alpha: Smoothing parameter
        """
        self.n_states = n_states
        self.orders = sorted(orders)
        self.alpha = alpha

        if weights is None:
            self.weights = np.ones(len(orders)) / len(orders)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()

        # Create models
        self.models = {}
        for order in orders:
            if order == 1:
                self.models[order] = MarkovChain(n_states=n_states, alpha=alpha)
            else:
                self.models[order] = HigherOrderMarkov(n_states=n_states, order=order, alpha=alpha)

    def fit(self, states: np.ndarray) -> 'EnsembleMarkov':
        """Train all models in the ensemble."""
        for order, model in self.models.items():
            logger.info(f"Training order-{order} model")
            model.fit(states)
        return self

    def predict(self, history: List[int]) -> np.ndarray:
        """
        Ensemble prediction given history.

        Args:
            history: List of previous states (uses last k states for each model)

        Returns:
            Weighted average probability distribution
        """
        predictions = []

        for order, weight in zip(self.orders, self.weights):
            model = self.models[order]

            if order == 1:
                # Use last state
                pred = model.predict(history[-1] if history else 0)
            else:
                # Use last 'order' states
                if len(history) >= order:
                    pred = model.predict(history[-order:])
                else:
                    # Not enough history, skip this model
                    continue

            predictions.append(weight * pred)

        if not predictions:
            # Fallback
            return np.ones(self.n_states) / self.n_states

        return np.sum(predictions, axis=0)

    def sample_next_state(
        self,
        history: List[int],
        rng: Optional[np.random.Generator] = None
    ) -> int:
        """Sample next state using ensemble prediction."""
        if rng is None:
            rng = np.random.default_rng()

        probs = self.predict(history)
        return int(rng.choice(self.n_states, p=probs))


# Backward compatibility function
def get_transition_matrix(
    states: np.ndarray,
    n_states: int = 5,
    alpha: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy function for backward compatibility.

    Use MarkovChain class for new code.
    """
    model = MarkovChain(n_states=n_states, alpha=alpha)
    model.fit(states)
    return model.get_transition_matrix(), model.get_entropy()

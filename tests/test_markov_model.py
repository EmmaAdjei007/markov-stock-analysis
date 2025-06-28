import pytest
import numpy as np
from src.markov_model import get_transition_matrix

def test_get_transition_matrix():
    """
    Test the transition matrix calculation.
    """
    states = np.array([0, 1, 0, 2, 1, 0])  # Example state sequence
    transition_matrix, row_entropy = get_transition_matrix(states, n_states=3)
    
    # Check transition matrix shape
    assert transition_matrix.shape == (3, 3), "Transition matrix should be 3x3"
    
    # Check row sums (probabilities should sum to 1)
    assert np.allclose(transition_matrix.sum(axis=1), 1), "Rows should sum to 1"
    
    # Check row entropy
    assert len(row_entropy) == 3, "Row entropy should have 3 values"
    assert all(isinstance(e, float) for e in row_entropy), "Row entropy values should be floats"

def test_get_transition_matrix_single_state():
    """
    Test transition matrix with a single state.
    """
    states = np.array([0, 0, 0])  # All states are the same
    transition_matrix, row_entropy = get_transition_matrix(states, n_states=1)
    
    # Check transition matrix shape
    assert transition_matrix.shape == (1, 1), "Transition matrix should be 1x1"
    assert transition_matrix[0, 0] == 1, "Single state should transition to itself"
    
    # Check row entropy
    assert row_entropy[0] == 0, "Row entropy should be 0 for a single state"
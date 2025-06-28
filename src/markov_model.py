import numpy as np
from scipy.stats import entropy
from typing import Dict, Tuple

def get_transition_matrix(states: np.ndarray, n_states: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct transition matrix and calculate row entropy.
    """
    transitions = np.zeros((n_states, n_states))
    
    for i in range(len(states) - 1):
        current_state = int(states[i])
        next_state = int(states[i + 1])
        transitions[current_state][next_state] += 1

    epsilon = 1e-8
    row_sums = transitions.sum(axis=1) + epsilon
    transition_matrix = transitions / row_sums[:, np.newaxis]
    
    # Calculate row entropy
    row_entropy = [entropy(row) for row in transition_matrix]
    
    return transition_matrix, row_entropy
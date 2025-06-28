import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
import os
from typing import Dict

def save_plot(fig, filename: str):
    """
    Save a plot to the reports/figures folder.
    """
    os.makedirs("reports/figures", exist_ok=True)
    fig.savefig(f"reports/figures/{filename}.png", bbox_inches='tight')
    plt.close(fig)


def plot_transition_matrices(transition_matrices: Dict[str, np.ndarray], state_labels: List[str], save_path=None):
    """
    Plot transition matrices with proper figure handling
    """
    for ticker, matrix in transition_matrices.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='YlOrRd',
                   xticklabels=state_labels, 
                   yticklabels=state_labels,
                   ax=ax)
        ax.set_title(f'Transition Probability Matrix for {ticker}')
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')
        
        if save_path:
            fig.savefig(f"{save_path}/{ticker}_transition_matrix.png", 
                       bbox_inches='tight', 
                       dpi=300)
        return fig 
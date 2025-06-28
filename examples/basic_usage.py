"""
Basic usage example for Markov Chain Stock Analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from markov_model import MarkovStockModel
from data_preprocessor import DataPreprocessor
import pandas as pd

def main():
    # Load sample data
    data_path = os.path.join('data', 'processed', 'sample_AAPL_processed.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} data points for analysis")
        
        # Initialize and fit model
        model = MarkovStockModel(n_states=5)
        model.fit(df['returns'])
        
        # Generate predictions
        predictions = model.simulate(days=30, n_simulations=1000)
        print(f"Generated {len(predictions)} day prediction")
        
    else:
        print(f"Sample data not found at {data_path}")
        print("Please ensure you have processed data available")

if __name__ == "__main__":
    main()

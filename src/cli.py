import argparse
from src.data_fetcher import get_stock_data
from src.data_preprocessor import assign_states
from src.markov_model import get_transition_matrix
from src.visualization import plot_transition_matrices

def main():
    parser = argparse.ArgumentParser(description="Stock Price Analysis using Markov Chains")
    parser.add_argument("--tickers", nargs='+', required=True, help="Stock ticker symbols (e.g., AAPL MSFT)")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Start date for analysis")
    parser.add_argument("--end_date", type=str, default="2023-01-01", help="End date for analysis")
    parser.add_argument("--n_states", type=int, default=5, help="Number of states for Markov model")
    
    args = parser.parse_args()
    
    # Fetch data
    data = get_stock_data(args.tickers, args.start_date, args.end_date)
    
    # Assign states
    processed_data = assign_states(data, n_states=args.n_states)
    
    # Analyze Markov chain
    transition_matrices = {}
    for ticker, df in processed_data.items():
        transition_matrix, _ = get_transition_matrix(df['State'].values, n_states=args.n_states)
        transition_matrices[ticker] = transition_matrix
    
    # Plot transition matrices
    state_labels = ['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']
    plot_transition_matrices(transition_matrices, state_labels)

if __name__ == "__main__":
    main()
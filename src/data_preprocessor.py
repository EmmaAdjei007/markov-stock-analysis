import numpy as np
import pandas as pd
import os
from typing import Dict

import numpy as np
import pandas as pd
import os
from typing import Dict

def assign_states(data: Dict[str, pd.DataFrame], n_states: int = 5, use_log_returns: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Assign states based on daily returns or log returns and save processed data.
    """
    processed_data = {}
    os.makedirs("data/processed", exist_ok=True)
    
    for ticker, df in data.items():
        if use_log_returns:
            df['Daily_Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
        else:
            df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Drop NA values and make an explicit copy to avoid SettingWithCopyWarning
        df = df.dropna().copy()
        
        # Use simple assignment (which is equivalent to df.loc[:, 'State'] = ...)
        df['State'] = pd.qcut(df['Daily_Return'], q=n_states, labels=False)
        
        state_labels = ['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']
        df['State_Description'] = pd.Categorical([state_labels[int(i)] for i in df['State']])
        
        # Save processed data
        processed_file = os.path.join("data/processed", f"{ticker}_processed.csv")
        df.to_csv(processed_file)
        
        processed_data[ticker] = df
    
    return processed_data

def load_processed_data(ticker: str) -> pd.DataFrame:
    """
    Load processed data from the processed data folder.
    """
    processed_file = os.path.join("data/processed", f"{ticker}_processed.csv")
    if os.path.exists(processed_file):
        return pd.read_csv(processed_file, index_col='Date', parse_dates=True)
    else:
        raise FileNotFoundError(f"Processed data for {ticker} not found.")
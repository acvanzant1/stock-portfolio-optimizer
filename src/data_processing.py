import pandas as pd
import numpy as np

def process_stock_data(stock_data):
    """Process raw stock data into a usable format for optimization."""
    try:
        # Extract adjusted close prices
        adj_close = stock_data['Adj Close']

        # Volatility of stocks
        volatility = adj_close.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

        # Percentage change 
        cov_matrix = adj_close.pct_change().apply(lambda x: np.log(1+x)).cov()

        # Yearly returns
        ind_er = adj_close.resample('Y').last().pct_change().mean()

        # Calculate daily returns
        returns = adj_close.pct_change().dropna()

        return adj_close, volatility, cov_matrix, ind_er, returns
    except Exception as e:
        print(f"Error processing stock data: {e}")
        return pd.DataFrame(), pd.DataFrame()


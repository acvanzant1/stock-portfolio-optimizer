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
        ind_er = adj_close.resample('YE').last().pct_change().mean()

        # Calculate daily returns
        returns = adj_close.pct_change().dropna()

        return adj_close, volatility, cov_matrix, ind_er, returns
    except Exception as e:
        print(f"Error processing stock data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    

def calculate_portfolio_metrics(weights, returns, cov_matrix, mu, risk_free_rate=0.02, initial_value=15000):
    # Volatility (annualized)
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(250)
    
    # Sharpe ratio
    portfolio_return = weights.T @ mu
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Cumulative returns
    portfolio_returns = returns @ weights
    cumulative_returns = (1 + portfolio_returns).cumprod()
    final_value = initial_value * cumulative_returns.iloc[-1]

    # Calculate annualized return
    n_years = len(returns) / 252  
    cumulative_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + cumulative_return) ** (1/n_years) - 1
    
    # Diversification ratio (stocks invested / total stocks)
    threshold = 0.01 
    stocks_invested = sum(weights > threshold)
    total_stocks = len(weights)
    diversification_ratio = stocks_invested / total_stocks
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'volatility': portfolio_volatility,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'final_value': final_value,
        'diversification_ratio': diversification_ratio,
        'stocks_invested': stocks_invested,
        'total_stocks': total_stocks
    }


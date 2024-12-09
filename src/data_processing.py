import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Processing the fetched data into usable information
def process_stock_data(stock_data):
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
    
# Code to plot a line graph of the prices of the stocks being analyzed over the time period 
def plot_prices(stock_data):
    fig, ax = plt.subplots(figsize=(15, 8))
    adj_close = stock_data['Adj Close']
    adj_close.plot(ax=ax)
    plt.title('Adjusted Close Prices Over Time', pad=20)  
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plotting a bar chart of the annual volatility of the stocks
def plot_volatility(stock_data):
    plt.figure(figsize=(12, 5))
    adj_close = stock_data['Adj Close']
    volatility = adj_close.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    volatility.plot(kind='bar')
    plt.title('Annualized Volatility by Stock')
    plt.xlabel('Stock')
    plt.ylabel('Volatility')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plotting the yearly returns of the stocks
def plot_yearly_returns(stock_data):
    plt.figure(figsize=(12, 5))
    adj_close = stock_data['Adj Close']
    yearly_returns = adj_close.resample('YE').last().pct_change().mean()
    yearly_returns.plot(kind='bar')
    plt.title('Average Yearly Returns by Stock')
    plt.xlabel('Stock')
    plt.ylabel('Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Calculating the metrics for the genetic algorithm that are returned at the end of each run
def calculate_portfolio_metrics_genetic(weights, returns, cov_matrix, mu, risk_free_rate=0.02, initial_value=15000):
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

# Calculates the metrics that are returned at the end of the run for the gradient descent 
def calculate_portfolio_metrics_gradient(weights, returns, cov_matrix, mu, risk_free_rate=0.02, initial_value=15000):
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
    threshold = 0.001
    stocks_invested = sum(weights > threshold)
    total_stocks = len(weights)
    diversification_ratio = stocks_invested / total_stocks
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'final_value': final_value,
        'diversification_ratio': diversification_ratio,
        'stocks_invested': stocks_invested,
        'total_stocks': total_stocks
    }

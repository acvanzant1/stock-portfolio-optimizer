import numpy as np
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
import matplotlib.pyplot as plt

class MultiObjectiveGradientDescent:
    def __init__(self, objectives, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.objectives = objectives
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    def compute_gradients(self, x):
        gradients = np.zeros_like(x)
        for obj in self.objectives:
            grad = self.numerical_gradient(obj, x)
            gradients += grad
        return gradients / len(self.objectives)

    def numerical_gradient(self, func, x, epsilon=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.array(x, dtype=float)
            x_minus = np.array(x, dtype=float)
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        return grad

    def project_weights(self, x):
        """
        Projects weights to satisfy constraints:
        1. Non-negative weights (x >= 0)
        2. Sum of weights equals 1 (∑x = 1)
        """
        x = np.maximum(x, 0)  # Ensure non-negative weights
        x /= np.sum(x)        # Normalize so the sum equals 1
        return x

    def optimize(self, x0):
        x = np.array(x0, dtype=float)
        x = self.project_weights(x)  # Ensure the initial point satisfies constraints

        for iteration in range(self.max_iter):
            gradients = self.compute_gradients(x)
            x_new = x - self.learning_rate * gradients

            # Project the new weights to satisfy constraints
            x_new = self.project_weights(x_new)

            # Check for convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"Converged in {iteration} iterations.")
                break

            x = x_new

        return x

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
    threshold = 0.001
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

def plot_gradient_descent_pie(optimized_params, tickers):
   # Create allocation dictionary and filter above 0.1%
   allocation = {name: weight for name, weight in zip(tickers, optimized_params) if weight > 0.001}
   
   # Prepare data for pie chart
   labels = list(allocation.keys())
   sizes = list(allocation.values())
   
   # Create pie chart
   fig1, ax1 = plt.subplots()
   ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
   ax1.axis('equal')
   
   plt.title('Portfolio Allocation - Gradient Descent')
   return plt.gcf()

def main():
    # Fetch data
    tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA", "GOOG", 
               "AVGO", "JPM", "LLY", "UNH", "V", "XOM", "MA", "COST", "HD", 
               "PG", "WMT", "NFLX", "JNJ", "CRM", "BAC"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    stock_data = fetch_stock_data(tickers, start_date, end_date)


    # Process stock data
    adj_close, volatility, cov_matrix, ind_er, returns_data = process_stock_data(stock_data)

    mu = np.array(ind_er)
    cov = np.array(cov_matrix)
    
    # Define example objective functions
    def objective1(x):
        # Expected risk 
        return np.sqrt(x.T @ cov_matrix @ x) * np.sqrt(250)

    def objective2(x):
        # Expected return 
        return -np.sum(x * mu)
    
    # Initialize optimizer
    optimizer = MultiObjectiveGradientDescent(objectives=[objective1, objective2])

    # Use uniform weights as the initial point for optimization
    initial_point = np.ones(len(tickers)) / len(tickers)  # Equal weights
    optimized_params = optimizer.optimize(initial_point)
    
    # Print the original results
    print("Optimized Parameters:", optimized_params)
    print("Sum of weights:", np.sum(optimized_params))
    print("Non-negative weights:", np.all(optimized_params >= 0))

    # Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(
        optimized_params, 
        returns_data,
        cov_matrix,
        mu    
        )

    print("\nPortfolio Metrics:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.4%}")
    print(f"Annualized Return: {metrics['annualized_return']:.4%}")
    print(f"Final Portfolio Value: ${metrics['final_value']:.2f}")
    print(f"Diversification: {metrics['stocks_invested']} out of {metrics['total_stocks']} stocks ({metrics['diversification_ratio']:.2%})")

    plot_gradient_descent_pie(optimized_params, tickers)
    plt.show()

if __name__ == "__main__":
    main() 
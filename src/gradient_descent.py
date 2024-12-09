import numpy as np
from data_fetching import fetch_stock_data
from data_processing import process_stock_data, calculate_portfolio_metrics_gradient
import matplotlib.pyplot as plt

# Class setting up the gradient descent problem
class MultiObjectiveGradientDescent:
    def __init__(self, objectives, learning_rate=0.01, max_iter=1000, tolerance=1e-6):
        self.objectives = objectives
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance

    # Computing the gradients using the two objective functions
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
    
    # Ensuring no negative weights and that all weights sum to 1
    def project_weights(self, x):
        x = np.maximum(x, 0) 
        x /= np.sum(x)        
        return x
    
    # Function whichc projects the weights onto the new iteration in line with the object function calculations
    def optimize(self, x0):
        x = np.array(x0, dtype=float)
        x = self.project_weights(x)

        for iteration in range(self.max_iter):
            gradients = self.compute_gradients(x)
            x_new = x - self.learning_rate * gradients
            x_new = self.project_weights(x_new)

            # Check for convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                print(f"Converged in {iteration} iterations.")
                break

            x = x_new

        return x
    
# Plot the pie chart showing the stock ratios that should be invested to maximize returns and minimize risk 
def plot_gradient_descent_pie(optimized_params, tickers):
   allocation = {name: weight for name, weight in zip(tickers, optimized_params) if weight > 0.001}

   labels = list(allocation.keys())
   sizes = list(allocation.values())

   fig1, ax1 = plt.subplots()
   ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
   ax1.axis('equal')
   
   plt.title('Portfolio Allocation - Gradient Descent')

   return plt.gcf()

def main():
    # Tickers of the stocks being analyzed
    tickers = [
   "LLY", "UNH", "NVO", "JNJ", "ABBV", "MRK", "AZN", "NVS", "ABT", "TMO",
   "ISRG", "DHR", "AMGN", "SYK", "PFE", "BSX", "SNY", "BMY", "VRTX", "GILD",
   "MDT", "ELV", "CI", "REGN", "HCA"
]
    # Time period being analyzed 
    start_date = "2020-01-01"
    end_date = "2021-12-31"
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Process stock data
    adj_close, volatility, cov_matrix, ind_er, returns_data = process_stock_data(stock_data)

    mu = np.array(ind_er)
    cov = np.array(cov_matrix)
    
    def objective1(x):
        # Expected risk 
        return np.sqrt(x.T @ cov_matrix @ x) * np.sqrt(250)

    def objective2(x):
        # Expected return 
        return -np.sum(x * mu)
    
    # Loading the objective functions into the gradient descent class
    optimizer = MultiObjectiveGradientDescent(objectives=[objective1, objective2])

    # Setting up the initial point to start from and feeding that into the optimizer variable to run the gradient descent
    initial_point = np.ones(len(tickers)) / len(tickers)
    optimized_params = optimizer.optimize(initial_point)
    
    print("Optimized Parameters:", optimized_params)
    print("Sum of weights:", np.sum(optimized_params))
    print("Non-negative weights:", np.all(optimized_params >= 0))

    # Calculate portfolio metrics that are returned at the end of the run
    metrics = calculate_portfolio_metrics_gradient(
        optimized_params, 
        returns_data,
        cov_matrix,
        mu    
        )

    print("\nPortfolio Metrics:")
    print(f"Return: {metrics['return']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Final Portfolio Value: ${metrics['final_value']:.2f}")
    print(f"Diversification: {metrics['stocks_invested']} out of {metrics['total_stocks']} stocks ({metrics['diversification_ratio']:.2%})")

    plot_gradient_descent_pie(optimized_params, tickers)
    plt.show()

if __name__ == "__main__":
    main() 
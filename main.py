import numpy as np
import matplotlib.pyplot as plt
from src.data_fetching import fetch_stock_data
from src.data_processing import process_stock_data
from src.nsga2 import get_efficient_frontier
from src.gradient_descent import MultiObjectiveGradientDescent

def run_genetic_algorithm(returns_data):
    # Perform optimization using NSGA2
    risks, returns, weights = get_efficient_frontier(returns_data)
    return risks, returns, weights

def run_gradient_descent(returns_data):
    # Define example objective functions
    def objective1(x):
        return np.sum(x**2)

    def objective2(x):
        return np.sum((x - 2)**2)

    # Initialize optimizer
    optimizer = MultiObjectiveGradientDescent(objectives=[objective1, objective2])

    # Use the mean of returns as the initial point for optimization
    initial_point = returns_data.mean().values
    optimized_params = optimizer.optimize(initial_point)
    return optimized_params

def compare_results(genetic_results, gradient_results):
    risks, returns, weights = genetic_results
    optimized_params = gradient_results

    print("\nComparison of Optimization Results:")
    print("Genetic Algorithm:")
    print(f"Number of portfolios: {len(risks)}")
    print(f"Risk range: {min(risks):.4f} to {max(risks):.4f}")
    print(f"Return range: {min(returns):.4f} to {max(returns):.4f}")

    print("\nGradient Descent Optimized Parameters:")
    print(optimized_params)

    # Plot the efficient frontier
    plt.figure(figsize=(12, 8))
    plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')
    plt.xlabel('Expected Risk (Volatility)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.show()

def main():
    # Fetch and process stock data
    tickers = ["AAPL", "GOOGL", "AMZN", "MSFT"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    _, returns_data = process_stock_data(stock_data)

    # Run both optimization methods
    genetic_results = run_genetic_algorithm(returns_data)
    gradient_results = run_gradient_descent(returns_data)

    # Compare results
    compare_results(genetic_results, gradient_results)

if __name__ == "__main__":
    main()
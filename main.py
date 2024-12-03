from src.data_fetching import fetch_stock_data
import numpy as np 
from src.data_fetching import fetch_stock_data
from src.data_processing import process_stock_data
from src.nsga2 import get_efficient_frontier
from src.gradient_descent import MultiObjectiveGradientDescent
import matplotlib.pyplot as plt


def main():
    tickers = ["AAPL", "GOOGL", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    adj_close_prices = fetch_stock_data(tickers, start_date, end_date)
    
    risk, returns = process_stock_data(adj_close_prices)
    
    def objective1(x):
        return np.sum(x**2)
    
    def objective2(x):
        return np.sum((x - 2)**2)
    
    optimizer = MultiObjectiveGradientDescent(objectives=[objective1, objective2])
    initial_point = np.array(fetch_stock_data(tickers, start_date, end_date))
    optimized_params = optimizer.optimize(initial_point)
    print("Optimized Parameters:", optimized_params)
    
    print("\nEfficient Frontier Results:")
    print(f"Number of portfolios: {len(risk)}")
    print(f"Risk range: {min(risk):.4f} to {max(risk):.4f}")
    print(f"Return range: {min(returns):.4f} to {max(returns):.4f}")

    plt.figure(figsize=(12, 8))
    plt.plot(risk, returns, 'b-', linewidth=2, label='Efficient Frontier')
    plt.xlabel('Expected Risk (Volatility)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()

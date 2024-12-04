import numpy as np
import matplotlib.pyplot as plt
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from nsga2 import get_efficient_frontier

def plot_efficient_frontier(risks, returns, show_points=True, show_labels=True):
    """Plot the efficient frontier with annotations"""
    plt.figure(figsize=(12, 8))  # Increased figure size for better readability
    
    # Plot efficient frontier line
    plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    if show_points:
        # Plot individual portfolios
        plt.scatter(risks, returns, c='blue', alpha=0.5, s=30)
    
    if show_labels:
        # Annotate minimum risk and maximum return portfolios
        min_risk_idx = np.argmin(risks)
        max_return_idx = np.argmax(returns)
        
        plt.scatter(risks[min_risk_idx], returns[min_risk_idx], 
                   color='red', marker='*', s=150, label='Minimum Risk')
        plt.scatter(risks[max_return_idx], returns[max_return_idx], 
                   color='green', marker='*', s=150, label='Maximum Return')
        
        # Add annotations with more descriptive text
        plt.annotate(f'Minimum Risk: {returns[min_risk_idx]:.2%} Return',
                    (risks[min_risk_idx], returns[min_risk_idx]),
                    xytext=(10, 10), textcoords='offset points')
        plt.annotate(f'Maximum Return: {returns[max_return_idx]:.2%} Return',
                    (risks[max_return_idx], returns[max_return_idx]),
                    xytext=(10, -10), textcoords='offset points')
    
    # Formatting
    plt.xlabel('Expected Risk (Volatility)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=16)  # Increased title font size
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)  # Adjusted legend font size
    
    # Adjust margins and layout
    plt.tight_layout()
    
    return plt.gcf()

def plot_portfolio_composition(weights, tickers, title="Portfolio Composition"):
    """Plot portfolio composition as a pie chart"""
    # Filter out tiny allocations for cleaner visualization
    significant_allocations = weights > 0.01
    significant_weights = weights[significant_allocations]
    significant_tickers = [tickers[i] for i, flag in enumerate(significant_allocations) if flag]
    
    plt.figure(figsize=(8, 8))
    plt.pie(significant_weights, labels=significant_tickers, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(title, fontsize=14)
    
    return plt.gcf()

def main():
    # Fetch data
    tickers = ["AAPL", "AMZN", "MSFT", "GOOG"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Process data
    adj_close, returns_data = process_stock_data(stock_data)

    # Get efficient frontier
    risks, returns, weights = get_efficient_frontier(returns_data)

     # Print results with improved formatting
    print("\nEfficient Frontier Results:")
    print(f"Number of portfolios: {len(risks)}")
    print(f"Risk range: {min(risks):.4f} to {max(risks):.4f}")
    print(f"Return range: {min(returns):.4f} to {max(returns):.4f}")

    # Print sample portfolio allocations with headers
    print("\nSample Portfolio Allocations:")
    min_risk_idx = np.argmin(risks)
    max_return_idx = np.argmax(returns)

    print("\nMinimum Risk Portfolio:")
    print(f"{'Ticker':<5} {'Weight':<6}")
    for ticker, weight in zip(tickers, weights[min_risk_idx]):
        if weight > 0.01:  # Only show significant allocations
            print(f"{ticker:<5} {weight:.4f}")

    print("\nMaximum Return Portfolio:")
    print(f"{'Ticker':<5} {'Weight':<6}")
    for ticker, weight in zip(tickers, weights[max_return_idx]):
        if weight > 0.01:
            print(f"{ticker:<5} {weight:.4f}")


    plot_efficient_frontier(risks, returns)
    plt.show()

    min_risk_idx = np.argmin(risks)
    plot_portfolio_composition(weights[min_risk_idx], tickers, 
                             "Minimum Risk Portfolio Composition")
    plt.show()

    max_return_idx = np.argmax(returns)
    plot_portfolio_composition(weights[max_return_idx], tickers,
                             "Maximum Return Portfolio Composition")
    plt.show()

if __name__ == "__main__":
    main()
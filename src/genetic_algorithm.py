import numpy as np
import matplotlib.pyplot as plt
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from nsga2 import get_efficient_frontier

def plot_efficient_frontier(risks, returns, weights, tickers, show_points=True, show_labels=True):
    """Plot the efficient frontier with annotations and additional information"""
    plt.figure(figsize=(12, 8))
    
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
        
        # Add annotations
        plt.annotate('Minimum Risk',
                    (risks[min_risk_idx], returns[min_risk_idx]),
                    xytext=(10, 10), textcoords='offset points')
        plt.annotate('Maximum Return',
                    (risks[max_return_idx], returns[max_return_idx]),
                    xytext=(10, -10), textcoords='offset points')
        
        # Annotate highest Sharpe ratio portfolio
        sharpe_ratios = returns / risks
        max_sharpe_idx = np.argmax(sharpe_ratios)
        plt.scatter(risks[max_sharpe_idx], returns[max_sharpe_idx], 
                   color='purple', marker='*', s=150, label='Max Sharpe Ratio')
        plt.annotate('Max Sharpe Ratio',
                    (risks[max_sharpe_idx], returns[max_sharpe_idx]),
                    xytext=(-50, 10), textcoords='offset points')
        
        # Add legend with additional information
        sharpe_ratio_value = sharpe_ratios[max_sharpe_idx]
        min_risk_value = risks[min_risk_idx]
        max_return_value = returns[max_return_idx]
        
        # Determine investment suggestion based on Sharpe ratio
        investment_suggestion = "Consider Investing" if sharpe_ratio_value > 1 else "Re-evaluate Investment"
        
        legend_text = (f"Sharpe Ratio: {sharpe_ratio_value:.2f}\n"
                       f"Min Risk: {min_risk_value:.2%}\n"
                       f"Max Return: {max_return_value:.2%}\n"
                       f"Suggestion: {investment_suggestion}")
        
        plt.gcf().text(0.75, 0.5, legend_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Formatting
    plt.xlabel('Expected Risk (Volatility)', fontsize=12)
    plt.ylabel('Expected Return', fontsize=12)
    plt.title('Efficient Frontier', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    
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
    
    # Add legend and total number of assets
    plt.legend(title=f"Total Assets: {len(significant_tickers)}", loc="upper right")
    plt.text(-1.5, -1.5, f"Cumulative Weight: {sum(significant_weights):.2f}", fontsize=10)
    
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

    # Print results
    print("\nEfficient Frontier Results:")
    print(f"Number of portfolios: {len(risks)}")
    print(f"Risk range: {min(risks):.4f} to {max(risks):.4f}")
    print(f"Return range: {min(returns):.4f} to {max(returns):.4f}")

    # Print sample portfolio allocations
    print("\nSample Portfolio Allocations:")
    min_risk_idx = np.argmin(risks)
    max_return_idx = np.argmax(returns)

    print("\nMinimum Risk Portfolio:")
    for ticker, weight in zip(tickers, weights[min_risk_idx]):
        if weight > 0.01:  # Only show significant allocations
            print(f"{ticker:<5} {weight:.4f}")

    print("\nMaximum Return Portfolio:")
    for ticker, weight in zip(tickers, weights[max_return_idx]):
        if weight > 0.01:
            print(f"{ticker:<5} {weight:.4f}")

    plot_efficient_frontier(risks, returns, weights, tickers)
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
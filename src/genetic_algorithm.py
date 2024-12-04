import numpy as np
import matplotlib.pyplot as plt
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import operator 

<<<<<<< HEAD
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
    
=======
class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, cov, risk_free_rate=0.02, **kwargs):
        """Initialize portfolio optimization problem"""
        super().__init__(n_var=len(mu), 
                        n_obj=2,
                        xl=0.0,
                        xu=1.0,
                        **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x) * np.sqrt(250.0)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = [exp_risk, -exp_return]
        out["sharpe"] = sharpe

class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # Clean small weights and normalize
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)

def get_efficient_frontier(ind_er, cov_matrix):
    """Calculate efficient frontier using pymoo NSGA2"""
    # Calculate inputs for optimization
    mu = np.array(ind_er)
    cov = np.array(cov_matrix)

    # Setup optimization
    problem = PortfolioProblem(mu, cov)
    algorithm = NSGA2(repair=PortfolioRepair())

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=1,
        verbose=True
    )

    X, F, sharpe = res.opt.get("X", "F", "sharpe")
    F = F * [1, -1]
    max_sharpe = sharpe.argmax()

    return X, F, sharpe, max_sharpe, mu, cov

def plot_efficient_frontier(F, max_sharpe, show_points=True, show_labels=True):

    plt.scatter(F[:, 0], F[:, 1], facecolor="none", edgecolors="blue", alpha=0.5, label="Pareto-Optimal Portfolio")
    plt.scatter(F[max_sharpe, 0], F[max_sharpe, 1], marker="x", s=100, color="red", label="Max Sharpe Portfolio")
    plt.legend()
    plt.xlabel("expected volatility")
    plt.ylabel("expected return")

>>>>>>> 67ce87a8d9b87e29833c5ccd0fc96ba361723254
    return plt.gcf()

def plot_pie_chart(allocation):
    for al in allocation:
        if al[1] <= 1e-6:
            allocation.remove(al)

    col_name = []
    w1 = []
    for name, w in allocation:
        col_name.append(name)
        w1.append(w)
        
    fig1, ax1 = plt.subplots()
    ax1.pie(w1, labels=col_name, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return plt.gcf()

def main():
    # Fetch data
    tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA", "GOOG", 
    "AVGO", "JPM", "LLY", "UNH", "V", "XOM", "MA", "COST", "HD", 
    "PG", "WMT", "NFLX", "JNJ", "CRM", "BAC"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    stock_data = fetch_stock_data(tickers, start_date, end_date)

    # Process data
    adj_close, volatility, cov_matrix, ind_er, returns = process_stock_data(stock_data)

    # Get efficient frontier
    X, F, sharpe, max_sharpe, mu, cov = get_efficient_frontier(ind_er, cov_matrix)

<<<<<<< HEAD
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
=======
    allocation = {name: w for name, w in zip(adj_close.columns, X[max_sharpe])}
    allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)

    print("Allocation With Best Sharpe")
    for name, w in allocation:
        print(f"{name:<5} {w}")

    x = X[max_sharpe].T
    print("Best Sharpe: \nReturn     = ", x.T @ mu)
    print("Volatility = ", np.sqrt(x.T @ cov @ x) * np.sqrt(250.0))

    plot_efficient_frontier(F, max_sharpe)
>>>>>>> 67ce87a8d9b87e29833c5ccd0fc96ba361723254
    plt.show()

    plot_pie_chart(allocation)
    plt.show()

if __name__ == "__main__":
    main()
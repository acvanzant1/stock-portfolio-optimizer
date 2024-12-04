import numpy as np
import matplotlib.pyplot as plt
from data_fetching import fetch_stock_data
from data_processing import process_stock_data, calculate_portfolio_metrics
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import operator 

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
        ('n_gen', 1000),
        seed=1,
        verbose=True
    )

    X, F, sharpe = res.opt.get("X", "F", "sharpe")
    F = F * [1, -1]
    max_sharpe = sharpe.argmax()

    return X, F, sharpe, max_sharpe, mu, cov

def plot_efficient_frontier(F, max_sharpe, show_points=True, show_labels=True):
    plt.scatter(F[:, 0], F[:, 1], facecolor="blue", edgecolors="blue", alpha=0.5, label="Pareto-Optimal Portfolio")
    plt.scatter(F[max_sharpe, 0], F[max_sharpe, 1], marker="x", s=100, color="red", label="Max Sharpe Portfolio")
    plt.legend()
    plt.xlabel("expected volatility")
    plt.ylabel("expected return")

    return plt.gcf()

def plot_pie_chart(allocation):
    filtered_allocation = [(name, w) for name, w in allocation if w > 0.0001]
    
    col_name = [name for name, w in filtered_allocation]
    w1 = [w for name, w in filtered_allocation]
    
    fig1, ax1 = plt.subplots()
    ax1.pie(w1, labels=col_name, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    
    return plt.gcf()

def main():
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

    allocation = {name: w for name, w in zip(adj_close.columns, X[max_sharpe])}
    allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)

    print("Allocation With Best Sharpe")
    for name, w in allocation:
        print(f"{name:<5} {w}")

    x = X[max_sharpe].T
    print("Best Sharpe: \nReturn     = ", x.T @ mu)
    print("Volatility = ", np.sqrt(x.T @ cov @ x) * np.sqrt(250.0))

    metrics = calculate_portfolio_metrics(
        x, 
        returns,
        cov_matrix,
        mu
    )
    
    print("\nPortfolio Metrics:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Cumulative Return: {metrics['cumulative_return']:.4%}")
    print(f"Annualized Returns: {metrics['annualized_return']:.4%}")
    print(f"Final Portfolio Value: ${metrics['final_value']:.2f}")
    print(f"Diversification: {metrics['stocks_invested']} out of {metrics['total_stocks']} stocks ({metrics['diversification_ratio']:.2%})")

    plot_efficient_frontier(F, max_sharpe)
    plt.show()

    plot_pie_chart(allocation)
    plt.show()

if __name__ == "__main__":
    main()
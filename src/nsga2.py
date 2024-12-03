import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

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

def get_efficient_frontier(returns_data):
    """Calculate efficient frontier using pymoo NSGA2"""
    # Calculate inputs for optimization
    mu = returns_data.mean().values
    cov = returns_data.cov().values

    # Setup optimization
    problem = PortfolioProblem(mu, cov)
    algorithm = NSGA2(
        pop_size=100,
        repair=PortfolioRepair(),
        seed=1
    )

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 100),
        seed=1,
        verbose=True
    )

    # Extract and sort results
    risks = res.F[:, 0]
    returns = -res.F[:, 1]  # Negative since we minimized
    weights = res.X

    sort_idx = np.argsort(risks)
    risks = risks[sort_idx]
    returns = returns[sort_idx]
    weights = weights[sort_idx]

    return risks, returns, weights
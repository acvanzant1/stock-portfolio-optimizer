import numpy as np

def calculate_portfolio_metrics(weights, returns, risk_free_rate=0.01):
    """
    Calculate portfolio metrics: return, volatility, and Sharpe ratio.

    Args:
        weights (np.ndarray): Array of portfolio weights.
        returns (pd.DataFrame): DataFrame of stock returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

    Returns:
        dict: Dictionary containing portfolio return, volatility, and Sharpe ratio.
    """
    try:
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights /= weights.sum()

        # Calculate portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return

        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        )  # Annualized volatility

        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        return {
            "return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio
        }
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        return {}

if __name__ == "__main__":
    import pandas as pd
    # Example usage
    weights = [0.4, 0.4, 0.2]  # Example weights
    example_returns = pd.DataFrame({
        "AAPL": [0.01, -0.005, 0.02],
        "GOOGL": [0.015, -0.01, 0.025],
        "AMZN": [0.02, -0.005, 0.03]
    })
    metrics = calculate_portfolio_metrics(weights, example_returns)
    print("Portfolio Metrics:", metrics)

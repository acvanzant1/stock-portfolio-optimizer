import pandas as pd

def process_stock_data(raw_data):
    """
    Process raw stock data into a usable format for optimization.

    Args:
        raw_data (pd.DataFrame): Raw stock data with multi-level columns.

    Returns:
        pd.DataFrame: Processed data containing adjusted close prices and returns.
    """
    try:
        # Extract adjusted close prices
        adj_close = raw_data.xs('Adj Close', axis=1, level=1)

        # Calculate daily returns
        returns = adj_close.pct_change().dropna()

        return adj_close, returns
    except Exception as e:
        print(f"Error processing stock data: {e}")
        return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    from data_fetching import fetch_stock_data

    # Example usage
    tickers = ["AAPL", "GOOGL", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    raw_data = fetch_stock_data(tickers, start_date, end_date)
    adj_close, returns = process_stock_data(raw_data)
    print("Adjusted Close Prices:")
    print(adj_close.head())
    print("\nDaily Returns:")
    print(returns.head())

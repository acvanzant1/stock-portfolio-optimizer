import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for the given tickers.

    Args:
        tickers (list): List of stock ticker symbols (e.g., ["AAPL", "GOOGL"]).
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: DataFrame containing the stock data with multi-level columns.
    """
    try:
        # Fetch data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    print(stock_data.head())

import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data for the given tickers."""
    try:
        # Fetch data for all tickers
        data = yf.download(tickers, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()


import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def fetch_stock_data(ticker, date_range, interval):
    """
    Fetch historical stock data for a given ticker and period
    """

    if interval == "1m":
        max_days = 8
        start_date = pd.to_datetime("now") - pd.Timedelta(days=max_days)
        end_date = pd.to_datetime("now")
        date_range = [start_date, end_date]

    if interval == "1h":
        max_days = 60
        start_date = pd.to_datetime("now") - pd.Timedelta(days=max_days)
        end_date = pd.to_datetime("now")
        date_range = [start_date, end_date]

    try:
        data = yf.download(ticker, start=date_range[0], end=date_range[1], interval=interval, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {ticker} at {interval} interval.")
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty dataframe on failure

    # data = yf.download(ticker, start=date_range[0], end=date_range[1], interval=interval, progress=False)

    # end_date = datetime.now()
    # if period == '1wk':
    #     start_date = end_date - timedelta(days=7)
    #     data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    # else:
    #     data = yf.download(ticker, period=period, interval=interval, progress=False)

    # data = yf.download(ticker, period=period, interval=interval, progress=False)
    return data

def process_data(data):
    """Ensure data is timezone-aware, formatted correctly, and indexed by Datetime."""
    data.index = data.index.tz_localize('UTC') if data.index.tzinfo is None else data.index
    data = data.tz_convert('US/Eastern').reset_index().rename(columns={'Date': 'Datetime'})

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data['Datetime'] = pd.to_datetime(data['Datetime'])
    return data.set_index('Datetime').reset_index()


def calculate_metrics(data):
    """
    Calculate key metrics for the stock data
    """
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data['High'].max()
    low = data['Low'].min()
    volume = data['Volume'].sum()
    return last_close, change, pct_change, high, low, volume

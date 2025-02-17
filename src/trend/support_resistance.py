import pandas as pd
import plotly.graph_objects as go

def compute_pivot_points(data, interval):
    """Compute hourly (daily) pivot points."""
    # if interval == '1d':
    #     data = data.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    # elif interval == '1wk':
    #     data = data.resample('W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    # elif interval == '2wk':
    #     data = data.resample('2W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    # elif interval == '1mo':
    #     data = data.resample('M').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})

    data['Prev_High'] = data['High'].shift(1)
    data['Prev_Low'] = data['Low'].shift(1)
    data['Prev_Close'] = data['Close'].shift(1)
    data['Pivot'] = (data['Prev_High'] + data['Prev_Low'] + data['Prev_Close']) / 3
    data['Support1'] = (2 * data['Pivot']) - data['Prev_High']
    data['Resistance1'] = (2 * data['Pivot']) - data['Prev_Low']
    data['Support2'] = data['Pivot'] - (data['Prev_High'] - data['Prev_Low'])
    data['Resistance2'] = data['Pivot'] + (data['Prev_High'] - data['Prev_Low'])
    return data.dropna()


def compute_weekly_pivot_points(data, interval='1wk'):
    """Compute weekly pivot points by resampling to weekly OHLC (week ending on Friday)."""

    if interval == '1d':
        data = data.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    elif interval == '1wk':
        data = data.resample('W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    elif interval == '2wk':
        data = data.resample('2W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    elif interval == '1mo':
        data = data.resample('M').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})

    # data = data.resample('W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    data['Prev_High'] = data['High'].shift(1)
    data['Prev_Low'] = data['Low'].shift(1)
    data['Prev_Close'] = data['Close'].shift(1)
    data['Pivot'] = (data['Prev_High'] + data['Prev_Low'] + data['Prev_Close']) / 3
    data['Support1'] = (2 * data['Pivot']) - data['Prev_High']
    data['Resistance1'] = (2 * data['Pivot']) - data['Prev_Low']
    data['Support2'] = data['Pivot'] - (data['Prev_High'] - data['Prev_Low'])
    data['Resistance2'] = data['Pivot'] + (data['Prev_High'] - data['Prev_Low'])
    return data.dropna()


def merge_weekly_pivot_points(hourly_data, weekly_data):
    """Merge the most recent weekly pivot values into hourly data."""
    hourly_data = hourly_data.reset_index()
    weekly_data = weekly_data.reset_index()
    weekly_data = weekly_data[['Datetime', 'Pivot', 'Support1', 'Resistance1', 'Support2', 'Resistance2']]
    return pd.merge_asof(hourly_data, weekly_data, on='Datetime', direction='backward' , suffixes=('_', ''))


def add_support_resistance_data(data, sup_res_range):
    """Compute and merge pivot points into the data."""
    data = compute_pivot_points(data, sup_res_range)
    data.set_index('Datetime', inplace=True)
    weekly_data = compute_weekly_pivot_points(data, sup_res_range)
    return merge_weekly_pivot_points(data, weekly_data)
import pandas as pd
import plotly.graph_objects as go

def compute_pivot_points(data):
    """Compute hourly (daily) pivot points."""
    data['Prev_High'] = data['High'].shift(1)
    data['Prev_Low'] = data['Low'].shift(1)
    data['Prev_Close'] = data['Close'].shift(1)
    data['Pivot'] = (data['Prev_High'] + data['Prev_Low'] + data['Prev_Close']) / 3
    data['Support1'] = (2 * data['Pivot']) - data['Prev_High']
    data['Resistance1'] = (2 * data['Pivot']) - data['Prev_Low']
    data['Support2'] = data['Pivot'] - (data['Prev_High'] - data['Prev_Low'])
    data['Resistance2'] = data['Pivot'] + (data['Prev_High'] - data['Prev_Low'])
    return data.dropna()


def compute_weekly_pivot_points(data):
    """Compute weekly pivot points by resampling to weekly OHLC (week ending on Friday)."""
    pivot_points = data.resample('W-FRI').agg({'High': 'max', 'Low': 'min', 'Close': 'last'})
    pivot_points['Prev_High'] = pivot_points['High'].shift(1)
    pivot_points['Prev_Low'] = pivot_points['Low'].shift(1)
    pivot_points['Prev_Close'] = pivot_points['Close'].shift(1)
    pivot_points['Pivot'] = (pivot_points['Prev_High'] + pivot_points['Prev_Low'] + pivot_points['Prev_Close']) / 3
    pivot_points['Support1'] = (2 * pivot_points['Pivot']) - pivot_points['Prev_High']
    pivot_points['Resistance1'] = (2 * pivot_points['Pivot']) - pivot_points['Prev_Low']
    pivot_points['Support2'] = pivot_points['Pivot'] - (pivot_points['Prev_High'] - pivot_points['Prev_Low'])
    pivot_points['Resistance2'] = pivot_points['Pivot'] + (pivot_points['Prev_High'] - pivot_points['Prev_Low'])
    return pivot_points.dropna()


def merge_weekly_pivot_points(hourly_data, weekly_data):
    """Merge the most recent weekly pivot values into hourly data."""
    hourly_data = hourly_data.reset_index()
    weekly_data = weekly_data.reset_index()
    weekly_data = weekly_data[['Datetime', 'Pivot', 'Support1', 'Resistance1', 'Support2', 'Resistance2']]
    return pd.merge_asof(hourly_data, weekly_data, on='Datetime', direction='backward' , suffixes=('', '_Weekly'))


def add_support_resistance_data(data):
    """Compute and merge pivot points into the data."""
    data = compute_pivot_points(data)
    data.set_index('Datetime', inplace=True)
    weekly_data = compute_weekly_pivot_points(data)
    return merge_weekly_pivot_points(data, weekly_data)
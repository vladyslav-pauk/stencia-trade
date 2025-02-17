import pandas as pd
import plotly.graph_objects as go

def compute_pivot_points(data, interval, levels=2):
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


def compute_weekly_pivot_points(data, interval='1wk', levels=2):
    """Compute pivot points by resampling OHLC data for the given interval and computing multiple levels of support and resistance."""

    # Resampling according to the specified interval
    resample_mapping = {
        '1d': 'D',
        '1wk': 'W-FRI',
        '2wk': '2W-FRI',
        '1mo': 'M'
    }

    if interval not in resample_mapping:
        raise ValueError("Invalid interval. Choose from ['1d', '1wk', '2wk', '1mo'].")

    data = data.resample(resample_mapping[interval]).agg({'High': 'max', 'Low': 'min', 'Close': 'last'})

    # Compute previous high, low, and close
    data['Prev_High'] = data['High'].shift(1)
    data['Prev_Low'] = data['Low'].shift(1)
    data['Prev_Close'] = data['Close'].shift(1)

    # Compute central pivot point
    data['Pivot'] = (data['Prev_High'] + data['Prev_Low'] + data['Prev_Close']) / 3

    # Compute multiple support and resistance levels
    for i in range(1, levels + 1):
        data[f'Support{i}'] = data['Pivot'] - i * (data['Prev_High'] - data['Prev_Low']) / 2
        data[f'Resistance{i}'] = data['Pivot'] + i * (data['Prev_High'] - data['Prev_Low']) / 2

    return data.dropna()


def merge_weekly_pivot_points(hourly_data, weekly_data):
    """Merge the most recent weekly pivot values into hourly data."""
    levels = weekly_data.filter(like='Support').shape[1]
    hourly_data = hourly_data.reset_index()
    weekly_data = weekly_data.reset_index()
    weekly_data = weekly_data[['Datetime', 'Pivot'] + [f'Support{level}' for level in range(1, levels + 1)] + [f'Resistance{level}' for level in range(1, levels + 1)]]
    return pd.merge_asof(hourly_data, weekly_data, on='Datetime', direction='backward' , suffixes=('_', ''))


def add_support_resistance_data(data, settings):
    """Compute and merge pivot points into the data."""
    data = compute_pivot_points(data, settings.get('sup_res_range', '1wk'))
    data.set_index('Datetime', inplace=True)
    weekly_data = compute_weekly_pivot_points(data, settings.get('sup_res_range', '1wk'), levels=settings.get('num_levels', 2))
    return merge_weekly_pivot_points(data, weekly_data)
import numpy as np
from scipy.optimize import differential_evolution
from .chart_utils import add_segments_with_lppls_to_chart


# def analyze_stock_data(global_data, start_date, end_date, scaling_factor, min_length):
#     if global_data["data"] is None:
#         return None, "No data available for analysis. Please fetch data first."
#     if not start_date or not end_date:
#         return None, "Please select a valid date range for analysis."
#
#     data = global_data["data"]
#     windowed_data = data.loc[start_date:end_date]
#     if windowed_data.empty:
#         return None, "No data available in the selected date range."
#
#     segments = segment_time_series(windowed_data['Close'].values, scaling_factor, min_length)
#     if not segments:
#         return None, "No trends detected for analysis."
#
#     lppls_fits = [lppls_fit(segment, windowed_data['Close'].values) for segment in segments]
#     return add_segments_with_lppls_and_landscapes(windowed_data, global_data["symbol"], segments, lppls_fits), None


def analyze_stock_data(global_data, start_date, end_date, scaling_factor, min_length):
    """
    Analyze stock data within the specified date range and update global_data with segments.
    """
    if global_data["data"] is None:
        return None, "No data available for analysis. Please fetch data first."
    if not start_date or not end_date:
        return None, "Please select a valid date range for analysis."

    data = global_data["data"]
    windowed_data = data.loc[start_date:end_date]
    if windowed_data.empty:
        return None, "No data available in the selected date range."

    # Perform segmentation
    segments = segment_time_series(windowed_data['Close'].values, scaling_factor, min_length)
    if not segments:
        return None, "No trends detected for analysis."

    # Store segments in global_data for reuse in TDA
    global_data["segments"] = segments

    # Fit LPPLS to each segment and collect parameters
    lppls_fits = []
    for segment in segments:
        params = lppls_fit(segment, windowed_data['Close'].values)
        lppls_fits.append((segment, params))

    # Add all segments and LPPLS fits to the chart
    return add_segments_with_lppls_to_chart(windowed_data, global_data["symbol"], segments, lppls_fits), None


def segment_time_series(prices, scaling_factor=0.15, min_length=10):
    """
    Segment time series into upward/downward trends.
    """
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    segments = []

    i = 0
    while i < len(returns):
        start = i
        cumulative_return = 0

        # Detect trends based on cumulative returns
        while i < len(returns):
            cumulative_return += returns[i]
            if abs(cumulative_return) > scaling_factor:
                break
            i += 1

        # Ensure the segment meets the minimum length requirement
        if i - start >= min_length:
            segments.append((start, i))

        # Avoid infinite loops
        i += 1

    return segments


def lppls_fit(segment, prices):
    """
    Fit the LPPLS model to a segment of the data.
    """
    t = np.arange(segment[0], segment[1] + 1)
    log_prices = np.log(prices[segment[0]:segment[1] + 1])

    def lppls_function(params):
        tc, m, omega, A, B, C1, C2 = params
        trend = A + B * (tc - t) ** m
        oscillations = (
            C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) +
            C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
        )
        return np.sum((log_prices - (trend + oscillations)) ** 2)

    # Define parameter bounds for LPPLS fitting
    bounds = [
        (t[-1] + 1, t[-1] + 10),  # Critical time (tc)
        (0.1, 1),  # Power law exponent (m)
        (0.1, 10),  # Angular frequency (omega)
        (-10, 10),  # Linear trend (A)
        (-10, 10),  # Exponential growth/decay (B)
        (-10, 10),  # Cosine amplitude (C1)
        (-10, 10)   # Sine amplitude (C2)
    ]

    # Optimize LPPLS parameters
    result = differential_evolution(lppls_function, bounds, maxiter=50, popsize=10, tol=0.01)
    return result.x

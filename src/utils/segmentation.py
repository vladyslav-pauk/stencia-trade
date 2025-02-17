import numpy as np
import matplotlib.pyplot as plt


def segment_time_series(prices, w0=240, scaling_factor=15.0, min_length=10):
    """
    Segment the time series into upward and downward trends based on cumulative returns.
    """
    log_prices = np.log(prices)
    returns = np.diff(log_prices)  # Log-returns
    segments = []

    # Compute volatility for dynamic epsilon
    sigma = np.zeros_like(returns)
    for i in range(len(returns)):
        valid_slice = returns[max(0, i - w0):i]
        sigma[i] = np.std(valid_slice) if len(valid_slice) > 0 else 0.0
    sigma = np.where(sigma == 0, 1e-6, sigma)  # Avoid zero division
    epsilon = scaling_factor * sigma  # Dynamic tolerance

    i = 0
    while i < len(returns):
        start = i
        cumulative_return = 0

        # Detect trend (upward or downward)
        while i < len(returns):
            cumulative_return += returns[i]
            if cumulative_return > epsilon[i]:  # Upward trend
                trend_type = "upward"
                break
            elif cumulative_return < -epsilon[i]:  # Downward trend
                trend_type = "downward"
                break
            i += 1

        if i >= len(returns):
            break

        trend_end = i
        while i < len(returns):
            cumulative_return += returns[i]
            if trend_type == "upward" and cumulative_return < 0:
                break
            elif trend_type == "downward" and cumulative_return > 0:
                break
            i += 1

        if trend_end - start >= min_length:
            segments.append((start, trend_end))
            # print(f"Detected {trend_type} trend: start={start}, end={trend_end}")

        i += 1  # Ensure progress to avoid infinite loops

    # print(f"Completed segmentation with {len(segments)} segments")
    return segments


def filter_segments(segments, min_length=10):
    """
    Remove segments that are too short.
    """
    return [seg for seg in segments if seg[1] - seg[0] >= min_length]


def plot_segments(prices, segments):
    plt.figure(figsize=(15, 5))

    # Compute global minimum and maximum for the y-axis
    global_min = np.min(prices)
    global_max = np.max(prices)

    # Plot the price data
    plt.plot(prices, label="Price", color="blue")

    # Add vertical spans for the segments
    for start, end in segments:
        plt.axvspan(start, end, color="green" if prices[start] < prices[end] else "red", alpha=0.3)

    # Set y-axis limits to stretch the y-axis fully
    plt.ylim(global_min, global_max)

    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    cumulative_returns = np.cumsum(returns) / np.max(np.abs(np.cumsum(returns))) * global_max

    # Plot the cumulative returns
    # plt.plot(cumulative_returns, color="orange", label="Cumulative Returns")

    # Add labels, legend, and title
    plt.title("Segmented Time Series")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    return plt
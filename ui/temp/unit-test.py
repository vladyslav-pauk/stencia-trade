import yfinance as yf
import numpy as np
from utils.lppls import segment_time_series, lppls_fit
from utils.tda import perform_tda
from utils.chart_utils import add_lppls_to_chart, add_tda_to_chart

def fetch_real_data(symbol, period="max"):
    """
    Fetch real stock data for a given symbol.
    """
    stock_info = yf.Ticker(symbol)
    history = stock_info.history(period=period)
    if history.empty:
        raise ValueError(f"No historical data available for {symbol}.")
    return history


def analyze_data(data, start_date, end_date, segment_choice, tda_params):
    """
    Perform segmentation, LPPLS fitting, and TDA on the given data within the date range.
    """
    windowed_data = data.loc[start_date:end_date]
    if windowed_data.empty:
        print("No data available in the selected date range.")
        return

    # Segment the time series
    segments = segment_time_series(windowed_data['Close'].values, min_length=3)
    if not segments:
        print("No trends detected for analysis.")
        return

    print(f"Detected {len(segments)} segments.")
    if segment_choice >= len(segments):
        print(f"Invalid segment choice. Only {len(segments)} segments detected.")
        return

    # Analyze the selected segment
    selected_segment = segments[segment_choice]
    print(f"Analyzing segment {segment_choice}: Start={selected_segment[0]}, End={selected_segment[1]}")

    # LPPLS Fit
    params = lppls_fit(selected_segment, windowed_data['Close'].values)
    print("LPPLS Fit Parameters:")
    param_names = ["tc", "m", "omega", "A", "B", "C1", "C2"]
    for name, value in zip(param_names, params):
        print(f"{name}: {value:.4f}")

    # Add LPPLS to chart
    chart_html = add_lppls_to_chart(windowed_data, "AAPL", selected_segment, params)

    # with open("lppls_chart.html", "w") as f:
    #     f.write(chart_html)
    # print("LPPLS chart saved as 'lppls_chart.html'.")

    # Perform TDA on the selected segment
    w, d, N = tda_params["w"], tda_params["d"], tda_params["N"]
    prices = windowed_data['Close'].values[selected_segment[0]:selected_segment[1] + 1]

    try:
        norms = perform_tda(prices, w, d, N)
        print(f"Time Series of TDA Norms for segment {segment_choice}:")
        for idx, norm in enumerate(norms):
            print(f"  Window {idx}: Norm = {norm:.4f}")

        # Add TDA results to the chart
        tda_chart_html = add_tda_to_chart(windowed_data, "AAPL", selected_segment, norms)
        # with open("tda_chart.html", "w") as f:
        #     f.write(tda_chart_html)
        # print("TDA chart saved as 'tda_chart.html'.")

    except Exception as e:
        print(f"TDA Analysis Failed: {e}")


if __name__ == "__main__":
    try:
        # Fetch real data for AAPL
        print("Fetching AAPL data...")
        aapl_data = fetch_real_data("AAPL")

        # Specify date range for analysis
        start_date = "2015-01-01"
        end_date = "2023-01-01"

        # Segment choice and TDA parameters
        segment_choice = 2  # Choose a specific segment
        tda_params = {"w": 50, "d": 5, "N": 3}  # Example TDA parameters

        # Perform analysis
        analyze_data(aapl_data, start_date, end_date, segment_choice, tda_params)

    except Exception as e:
        print(f"Error: {e}")
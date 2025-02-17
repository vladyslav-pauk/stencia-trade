from flask import Flask, render_template, request
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd

app = Flask(__name__)

# Global variables to hold the fetched stock data
fetched_data = None
fetched_symbol = ""

@app.route("/", methods=["GET", "POST"])
def index():
    global fetched_data, fetched_symbol

    chart_data = None
    error = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "fetch":
            # Fetch stock data without date input
            symbol = request.form.get("symbol").upper()
            try:
                stock_info = yf.Ticker(symbol)
                history = stock_info.history(period="max")
                if not history.empty:
                    start_date = history.index.min().strftime('%Y-%m-%d')  # Earliest date
                    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Today's date

                    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if stock_data.empty:
                        error = f"No data found for symbol {symbol}."
                    else:
                        if isinstance(stock_data.columns, pd.MultiIndex):
                            stock_data.columns = stock_data.columns.get_level_values(0)

                        required_columns = ['Open', 'High', 'Low', 'Close']
                        if not all(col in stock_data.columns for col in required_columns):
                            error = "The required columns are missing in the fetched data."
                        else:
                            fetched_data = stock_data.dropna(subset=required_columns)
                            fetched_symbol = symbol
                            chart_data = create_chart(fetched_data, symbol)
                else:
                    error = f"No historical data available for {symbol}."
            except Exception as e:
                error = f"Error fetching data: {str(e)}"
                print(f"Exception: {e}")

        elif action == "analyze":
            # Perform analysis on the currently fetched data
            if fetched_data is None:
                error = "No data available for analysis. Please fetch data first."
            else:
                start_date = request.form.get("start_date")
                end_date = request.form.get("end_date")
                if not start_date or not end_date:
                    error = "Please select a date range for analysis."
                else:
                    windowed_data = fetched_data.loc[start_date:end_date]
                    if windowed_data.empty:
                        error = "No data available in the selected date range."
                    else:
                        segments = segment_time_series(windowed_data['Close'].values)
                        if segments:
                            params = lppls_fit(segments[0], windowed_data['Close'].values)
                            chart_data = add_lppls_to_chart(windowed_data, create_chart(windowed_data, fetched_symbol), segments[0], params, fetched_symbol)
                        else:
                            error = "No trends detected for analysis."

    return render_template("index.html", chart_data=chart_data, symbol=fetched_symbol, error=error)

def create_chart(data, symbol):
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    ))

    fig.update_layout(
        title=f"{symbol}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=800
    )
    return fig.to_html(full_html=False)

def add_lppls_to_chart(stock_data, chart_html, segment, params, symbol):
    t = np.arange(segment[0], segment[1] + 1)
    log_prices = np.log(stock_data['Close'].values[segment[0]:segment[1] + 1])
    tc, m, omega, A, B, C1, C2 = params

    trend = A + B * (tc - t) ** m
    oscillations = C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) + \
                   C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
    fitted = trend + oscillations

    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Candlestick"
    ))

    # Add LPPLS fit
    fig.add_trace(go.Scatter(
        x=stock_data.index[segment[0]:segment[1] + 1],
        y=np.exp(fitted),
        mode='lines',
        name="LPPLS Fit",
        line=dict(color='orange', dash='dot')
    ))

    fig.update_layout(
        title=f"{symbol}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=800
    )
    return fig.to_html(full_html=False)

def segment_time_series(prices, scaling_factor=15.0, min_length=10):
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    segments = []

    i = 0
    while i < len(returns):
        start = i
        cumulative_return = 0
        while i < len(returns):
            cumulative_return += returns[i]
            if cumulative_return > scaling_factor or cumulative_return < -scaling_factor:
                break
            i += 1

        if i - start >= min_length:
            segments.append((start, i))
        i += 1

    return segments

def lppls_fit(segment, prices):
    t = np.arange(segment[0], segment[1] + 1)
    log_prices = np.log(prices[segment[0]:segment[1] + 1])

    def lppls_function(params):
        tc, m, omega, A, B, C1, C2 = params
        trend = A + B * (tc - t) ** m
        oscillations = C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) + \
                       C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
        return np.sum((log_prices - (trend + oscillations)) ** 2)

    bounds = [
        (t[-1] + 1, t[-1] + 10),
        (0.1, 1),
        (0.1, 10),
        (-10, 10),
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]

    result = differential_evolution(lppls_function, bounds, maxiter=50, popsize=10, tol=0.01)
    return result.x

if __name__ == "__main__":
    app.run(debug=True)
import yfinance as yf
from utils.chart_utils import create_chart


def fetch_stock_data(symbol, global_data):
    """
    Fetch historical stock data for the given symbol.
    """
    try:
        stock_info = yf.Ticker(symbol)
        history = stock_info.history(period="max")
        if history.empty:
            return None, f"No historical data available for {symbol}."

        global_data["data"] = history
        global_data["symbol"] = symbol
        return create_chart(history, symbol), None
    except Exception as e:
        return None, f"Error fetching data: {e}"
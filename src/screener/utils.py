import numpy as np
import json
import yfinance as yf


def load_config(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

# Fetch stock data
def get_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# Fetch fundamental data (P/E Ratios)
def get_PE_ratios(ticker):
    stock = yf.Ticker(ticker)
    pe_trailing = stock.info.get("trailingPE", np.nan)
    pe_forward = stock.info.get("forwardPE", np.nan)
    return pe_trailing, pe_forward

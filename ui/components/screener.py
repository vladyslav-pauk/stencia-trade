# import pandas as pd
#
# def screener_tab(st):
#     st.session_state.selected_tab = "Screener"  # Store selected tab
#     with st.sidebar:
#         # st.header("Settings")
#         screener_criteria = st.selectbox("Select Screener Criteria",
#                                          ["Most Active", "Top Gainers", "Top Losers", "52 Week High", "52 Week Low"])
#         run_screener = st.button("Screen")
#
#     if run_screener:
#         st.session_state.screener_data = pd.DataFrame(
#             {"Ticker": ["AAPL", "TSLA", "NVDA"], "Price": [150.0, 700.0, 450.0], "Change %": [2.5, -1.2, 3.8]}
#         )
#
#     # st.subheader("Stock Screener Results")
#     if "screener_data" in st.session_state:
#         # st.info(f"Fetching stocks based on **{screener_criteria}** criteria...")
#         st.dataframe(st.session_state.screener_data)
#         # todo: add to watchlist buttons or checkmarks and button in sidebar
#     else:
#         st.info("Select screener criteria in the sidebar and click 'Run Screener'.")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tradingview_ta import TA_Handler, Interval, Exchange
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from src.utils.charts import create_chart, add_indicator_charts, plot_screener_results

# Define list of tickers
TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "HD", "MCD", "NFLX", "JNJ",
    # "PFE", "MRK", "ABT", "JPM", "BAC", "WFC", "GS", "T", "VZ", "PG", "KO", "PEP",
    # "WMT", "XOM", "CVX", "COP", "GE", "BA", "LMT", "DD", "SHW"
]


def fetch_tradingview_data(tickers):
    """Fetch trading recommendations from TradingView API"""
    tickers_data = []

    for ticker in tickers:
        try:
            # Try retrieving from NYSE first
            data = TA_Handler(
                symbol=ticker, screener="america", exchange="NYSE", interval="1d"
            ).get_analysis().summary
        except:
            # If not found in NYSE, try NASDAQ
            try:
                data = TA_Handler(
                    symbol=ticker, screener="america", exchange="NASDAQ", interval="1d"
                ).get_analysis().summary
            except:
                continue  # Skip ticker if not found

        tickers_data.append(data)

    return tickers_data


def analyze_tickers(tickers_data):
    """Process trading data into structured format"""
    recommendations, buys, sells, neutrals = [], [], [], []

    for data in tickers_data:
        recommendations.append(data.get("RECOMMENDATION"))
        buys.append(data.get("BUY"))
        sells.append(data.get("SELL"))
        neutrals.append(data.get("NEUTRAL"))

    df = pd.DataFrame({
        "Ticker": TICKERS,
        "Recommendations": recommendations,
        "Buys": buys,
        "Sells": sells,
        "Neutrals": neutrals
    })

    # Sort by recommendation strength
    order_categories = {"STRONG_BUY": 5, "BUY": 4, "NEUTRAL": 3, "SELL": 2, "STRONG_SELL": 1}
    df["Order"] = df["Recommendations"].map(order_categories)
    df = df.sort_values("Order", ascending=True).drop("Order", axis=1).reset_index(drop=True)

    return df


def filter_by_technical_indicators():
    """Perform additional filtering based on EMA, RSI, MACD, ADX"""
    df_indicators = pd.DataFrame()

    for ticker in TICKERS:
        try:
            data = TA_Handler(
                symbol=ticker, screener="america", exchange="NYSE", interval="1d"
            ).get_analysis().indicators
        except:
            try:
                data = TA_Handler(
                    symbol=ticker, screener="america", exchange="NASDAQ", interval="1d"
                ).get_analysis().indicators
            except:
                continue  # Skip if not found

        temp_df = pd.DataFrame(data, index=[ticker])
        df_indicators = pd.concat([df_indicators, temp_df])

    df_indicators.columns = df_indicators.columns.astype(str)

    # Apply filtering conditions
    df_filtered = df_indicators[
        (df_indicators['EMA10'] > df_indicators['EMA20']) &  # EMA Crossover
        (df_indicators['RSI'] > 75) &  # RSI Overbought
        (df_indicators['MACD.macd'] > df_indicators['MACD.signal']) &  # MACD Crossover
        (df_indicators['ADX'] > 30)  # Strong Trend
        ]

    return df_filtered.index.tolist()


def screener_tab(st):
    """Implements the Stock Screener Tab in Streamlit"""
    st.session_state.selected_tab = "Screener"

    with st.sidebar:
        st.header("Stock Screener")
        screener_criteria = st.selectbox("Select Screener Criteria", [
            "Most Active", "Top Gainers", "Top Losers", "52 Week High", "52 Week Low"
        ])
        run_screener = st.button("Screen")

        st.divider()
        st.subheader("Filters")
        st.multiselect("Select Filters", ["Volume", "Price", "Market Cap"])

        st.divider()
        st.subheader("Watchlist")
        selected_tickers = st.multiselect("Add to Watchlist", TICKERS)
        if st.button("Update Watchlist"):
            st.session_state.watchlist = selected_tickers
            st.success("Watchlist updated!")

    if run_screener:
        # Fetch TradingView recommendations
        tickers_data = fetch_tradingview_data(TICKERS)
        # screener_results = analyze_tickers(tickers_data)
        filtered_shares = filter_by_technical_indicators()

        # Store results

        st.session_state.screener_data = fetch_screener_data(TICKERS)
        # st.session_state.screener_data = screener_results
        st.session_state.filtered_shares = filtered_shares




    # Display screener results
    if "screener_data" in st.session_state:

        # st.subheader("TradingView Screener Results")

        st.plotly_chart(plot_screener_results(st.session_state.screener_data), use_container_width=True)

        # st.dataframe(st.session_state.screener_data)

        # Show watchlist
        if "watchlist" in st.session_state:
            st.subheader("Your Watchlist")
            st.write(", ".join(st.session_state.watchlist))

        # Show filtered technical stocks
        # st.subheader("Filtered stocks")
        st.write("Filtered stocks meeting technical criteria:", ", ".join(st.session_state.filtered_shares))

        # Generate charts
        # if st.button("Show Charts for Selected Stocks"):
        #     display_stock_charts(st.session_state.watchlist)

    else:
        st.info("Select criteria and click 'Screen Stocks'.")


def display_stock_charts(tickers):
    """Display stock charts using Yahoo Finance"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 30)).strftime("%Y-%m-%d")

    num_rows = len(tickers)

    if num_rows == 0:
        st.warning("No stocks selected for charts.")
        return

    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(12, 4 * num_rows))
    fig.suptitle("Daily Candlestick Charts", fontsize=14)

    if num_rows == 1:
        axes = [axes]  # Ensure iterable when there's only one subplot

    for i, symbol in enumerate(tickers):
        data = yf.download(symbol, start=start_date, end=end_date)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        mpf.plot(data, type='candle', ax=axes[i], volume=False, show_nontrading=False)
        axes[i].set_title(symbol)

    plt.tight_layout()
    st.pyplot(fig)


def fetch_screener_data(tickers):
    """Fetch trading recommendations for given tickers from TradingView."""
    tickers_data = []

    for ticker in tickers:
        try:
            data = TA_Handler(
                symbol=ticker,
                screener="america",
                exchange="NYSE",
                interval="1d"
            ).get_analysis().summary
        except:
            try:
                data = TA_Handler(
                    symbol=ticker,
                    screener="america",
                    exchange="NASDAQ",
                    interval="1d"
                ).get_analysis().summary
            except:
                continue  # Skip if data is not available

        tickers_data.append({"Ticker": ticker, **data})

    return pd.DataFrame(tickers_data)


# def screener_tab(st):
#     """Render the stock screener tab in Streamlit."""
#     st.session_state.selected_tab = "Screener"
#
#     with st.sidebar:
#         screener_criteria = st.selectbox("Select Screener Criteria",
#                                          ["Most Active", "Top Gainers", "Top Losers", "52 Week High", "52 Week Low"])
#         run_screener = st.button("Run Screener")
#
#     tickers = [
#         "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "HD", "MCD", "NFLX",
#         "JNJ", "PFE", "MRK", "ABT", "JPM", "BAC", "WFC", "GS", "T", "VZ", "PG",
#         "KO", "PEP", "WMT", "XOM", "CVX", "COP", "GE", "BA", "LMT", "DD", "SHW"
#     ]
#
#     if run_screener:
#         st.session_state.screener_data = fetch_screener_data(tickers)
#
#     if "screener_data" in st.session_state and not st.session_state.screener_data.empty:
#         st.plotly_chart(plot_screener_results(st.session_state.screener_data), use_container_width=True)
#
#         # Add watchlist selection
#         st.subheader("Watchlist")
#         selected_stocks = st.multiselect("Add to Watchlist", st.session_state.screener_data["Ticker"].tolist())
#
#         if selected_stocks:
#             st.write("Selected Stocks:", ", ".join(selected_stocks))
#     else:
#         st.info("Select screener criteria in the sidebar and click 'Run Screener'.")
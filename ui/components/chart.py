import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.chart import create_chart, add_indicator_charts
from src.utils.indicators import add_indicator_data
from src.utils.data import fetch_stock_data, process_data
from src.trend.support_resistance import add_support_resistance_data

def chart_tab(st):
    st.session_state.selected_tab = "Chart"  # Store selected tab
    with st.sidebar:
        # st.header("Settings")
        ticker = st.text_input("Ticker", "SPY")
        update_chart = st.button("Update")
        # todo: dropdown with watchlist

        st.divider()
        st.subheader("Time Settings")
        date_range = st.date_input("Date Range", [pd.to_datetime("2024-01-01"), pd.to_datetime("now")])
        interval = st.selectbox("Interval", ["1d", "1h", "1m"])

        st.divider()
        st.subheader("Indicators")
        indicators = st.multiselect("Technical Indicators", ["SMA", "EMA", "TDA", "S&R"])
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"])

        st.divider()
        st.subheader("Support & Resistance")
        sup_res_range = st.selectbox("Interval", ["1wk", "2wk", "1mo"])

    if update_chart or "chart_data" in st.session_state:
        if update_chart:
            # todo: date_range, and interval
            st.session_state.chart_data = fetch_stock_data(ticker, date_range, interval)
            st.session_state.chart_data = process_data(st.session_state.chart_data)
            st.session_state.chart_data = add_indicator_data(st.session_state.chart_data)
            st.session_state.chart_data = add_support_resistance_data(st.session_state.chart_data, sup_res_range)

            st.session_state.chart_fig = create_chart(st.session_state.chart_data, ticker, chart_type)
            st.session_state.chart_fig = add_indicator_charts(st.session_state.chart_fig, st.session_state.chart_data, indicators)

        # st.subheader(f"{ticker} Price Chart")
        st.plotly_chart(st.session_state.chart_fig, use_container_width=True)

        st.dataframe(
            st.session_state.chart_data.assign(
                Datetime=st.session_state.chart_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
            )
        )
    else:
        st.info("Click 'Update Chart' in the sidebar to generate a chart.")


## CHART ##
# if st.sidebar.button('Update'):
#     data = fetch_stock_data(ticker, time_period)
#     data = process_data(data)
#     data = add_indicator_data(data)
#     data = add_support_resistance_data(data)
#
#     # Display main metrics
#     last_close, change, pct_change, high, low, volume = calculate_metrics(data)
#     st.metric(label=f"{ticker} Last Price", value=f"{last_close.squeeze():.2f} USD", delta=f"{change.squeeze():.2f} ({pct_change.squeeze():.2f}%)")
#
#     col1, col2, col3 = st.columns(3)
#     col1.metric("High", f"{high.squeeze():.2f} USD")
#     col2.metric("Low", f"{low.squeeze():.2f} USD")
#     col3.metric("Volume", f"{volume.squeeze():,}")
#
#     # Plot the stock price chart
#
#     fig = create_chart(data, ticker, time_period, chart_type)
#     fig = add_indicator_charts(fig, data, indicators)
#     st.plotly_chart(fig, use_container_width=True)
#
#     # Display historical data and technical indicator data
#     st.subheader('Historical Data')
#     st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])
#
#     st.subheader('Technical Indicators')
#     st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']])

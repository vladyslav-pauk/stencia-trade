import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data import fetch_stock_data, process_data
from src.trend.support_resistance import add_support_resistance_data
from src.trade.strategy import support_resistance
from src.utils.charts import create_trader_chart
from .panels import indicator_settings_panel, trader_settings_loader

def trader_tab(st):
    st.session_state.selected_tab = "Trader"
    with st.sidebar:
        st.header("Trading Simulator")
        ticker = st.text_input("Ticker", "SPY")
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
        date_range = st.date_input("Date Range", [pd.to_datetime("2024-01-01"), pd.to_datetime("now")])
        interval = st.radio("Interval", ["1d", "1h", "1m"], horizontal=True)

        execute_trade = st.button("Backtest")

        # st.divider()
        # st.subheader("Indicator Settings")
        # range = st.selectbox("Pivot Range", ["1wk", "2wk", "1mo"])
        # Dynamic indicator settings
        # st.divider()
        # st.subheader("Indicator Settings")


        st.divider()
        st.subheader("Trading Strategy")
        # indicator = st.selectbox("Indicator", ["S&R"])

        strategy = st.selectbox("Strategy", ["Support-Resistance", "Moving Average Crossover"], label_visibility="collapsed")
        st = trader_settings_loader(st)

        indicator = {"Support-Resistance": "S&R", "Moving Average Crossover": "SMA"}[strategy]
        st.markdown("**Indicator Settings**")
        st = indicator_settings_panel(indicator, st)

        st.markdown("**Trader Settings**")
        entry_threshold = st.slider("Entry Threshold (%)", 0.0, 1.0, 0.01, step=0.01)
        stop_loss = st.slider("Stop Loss (%)", 0.01, 10.0, 3.0, step=0.1)
        take_profit = st.slider("Take Profit (%)", 0.01, 10.0, 5.0, step=0.1)
        best_parameters = st.button("Optimize Parameters")

        st.divider()
        st.subheader("Notifications")
        email = st.text_input("Email", "all-stenciatrade-aaaapehsn6kabhpikuy7ly7sty@stenciatrade.slack.com")
        set_notifications = st.button("Send Notifications")

    if execute_trade or "trade_summary" in st.session_state:
        if execute_trade:
            st.session_state.trader_data = fetch_stock_data(ticker, date_range, interval)
            st.session_state.trader_data = process_data(st.session_state.trader_data)
            st.session_state.trader_data = add_support_resistance_data(st.session_state.trader_data, st.session_state.indicator_settings['S&R'])
            st.session_state.trade_summary = support_resistance(st.session_state.trader_data, strategy, entry_threshold / 100, stop_loss / 100, take_profit / 100)

            st.session_state.trade_fig = create_trader_chart(st)
            # st.session_state.chart_fig = add_indicator_charts(st.session_state.chart_fig, st.session_state.chart_data, indicators)

        st.plotly_chart(st.session_state.trade_fig, use_container_width=True)
        st.dataframe(st.session_state.trade_summary)
    else:
        st.info("Click 'Backtest' to simulate the trading strategy.")

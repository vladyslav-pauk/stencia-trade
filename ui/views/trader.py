import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data import fetch_stock_data, process_data
from src.trade.strategy import support_resistance_strategy

from ui.components.notifications import notifications
from ui.components.dashboard import notifications_monitor
from ui.components.sidebar import indicator_settings_pane, trader_settings_loader_pane, trader_settings_pane
from ui.components.indicators import add_support_resistance_data
from ui.components.charts import create_trader_chart, add_indicator_charts


def trader_tab(st):
    """Trader Tab: Displays trading strategy backtesting and notifications."""
    st.session_state.selected_tab = "Trader"

    with st.sidebar:
        st.header("Trading Simulator")
        st.session_state.ticker = st.text_input("Ticker", "SPY")
        st.session_state.chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
        st.session_state.date_range = st.date_input("Date Range", [pd.to_datetime("2024-01-01"), pd.to_datetime("now")])
        st.session_state.interval = st.radio("Interval", ["1d", "1h", "1m"], horizontal=True)
        backtest_trade = st.button("Backtest")

        st.divider()
        st.subheader("Trading Strategy")
        st.session_state.strategy = st.selectbox("Strategy", ["Support-Resistance", "Moving Average Crossover"], label_visibility="collapsed")
        st = trader_settings_loader_pane(st)
        st.markdown("**Indicator Settings**")
        st = indicator_settings_pane({"Support-Resistance": "S&R", "Moving Average Crossover": "SMA"}[st.session_state.strategy], st)
        st.markdown("**Trader Settings**")
        st = trader_settings_pane(st)

        st.divider()
        st.subheader("Notifications")
        email = st.text_input("Email", "paukvp@gmail.com")# "all-stenciatrade-aaaapehsn6kabhpikuy7ly7sty@stenciatrade.slack.com")
        st.session_state.set_notifications = st.button("Send Notifications")
        st = notifications(st, email)

    if backtest_trade or "trade_summary" in st.session_state:
        if backtest_trade:
            st.session_state.trader_data = fetch_stock_data(st.session_state.ticker, st.session_state.date_range, st.session_state.interval)
            st.session_state.trader_data = process_data(st.session_state.trader_data)

            st.session_state.trader_data = add_support_resistance_data(
                st.session_state.trader_data,
                st.session_state.indicator_settings['S&R']
            )
            st.session_state.trade_summary = support_resistance_strategy(
                st.session_state.trader_data,
                st.session_state.strategy,
                st.session_state.trade_settings['entry_threshold'] / 100,
                st.session_state.trade_settings['stop_loss'] / 100,
                st.session_state.trade_settings['take_profit'] / 100
            )

            st.session_state.trade_fig = create_trader_chart(st)
            # st.session_state.trade_fig = add_indicator_charts(
            #     st.session_state.trade_fig,
            #     st.session_state.trader_data,
            #     [indicator]
            # )

        if "monitor_thread" in st.session_state and st.session_state.monitor_thread:
                notifications_monitor(st)

        st.plotly_chart(st.session_state.trade_fig, use_container_width=True)
        st.dataframe(st.session_state.trade_summary)
    else:
        st.info("Click 'Backtest' to simulate the trading strategy.")

    # fixme: use real dates in trader
    # todo: fix ranges and inside margins (0 should be at axis crossing).

    # fixme: unify chart and trader tabs
    # todo: separate trader and charter indicator settings, or unify and use the same
    # todo: chart type independent of chart tab (don't use st for this)

    # fixme: create notifications monitor tab

    # fixme: cumulative and relative profit calculation and extension
    # fixme: add show/hide table button

    # fixme: 1m interval error, add 1 wk interval (handle tda error when short range)

    # fixme: tune and check S&R indicators and strategy
    # fixme: add checkbox to add indicators to the strategy so we have combined strategy,
    #  save and load instead of indicator = {"Support-Resistance": "S&R", "Moving Average Crossover": "SMA"}[st.session_state.strategy]
    #  strategy is defined for each indicator;
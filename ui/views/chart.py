import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data import fetch_stock_data, process_data

from ui.components.sidebar import indicator_settings_pane, indicator_settings_loader_pane
from ui.components.indicators import add_indicator_data, add_support_resistance_data
from ui.components.charts import create_indicator_chart


def chart_tab(st):
    """Chart Tab: Displays stock price chart and technical indicators."""
    st.session_state.selected_tab = "Chart"

    with st.sidebar:
        st.header("Charting Tool")

        ticker = st.text_input("Ticker", "SPY")
        chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
        date_range = st.date_input("Date Range", [pd.to_datetime("2024-01-01"), pd.to_datetime("now")])
        interval = st.radio("Interval", ["1d", "1h", "1m"], horizontal=True)

        update_chart = st.button("Update")

        st.divider()
        st.subheader("Indicators")
        if "indicators" not in st.session_state:
            st.session_state.indicators = []
        st.session_state.indicators = st.multiselect(
            "Indicators",
            ["SMA", "EMA", "TDA", "S&R", "FIB", "TRE"],
            default=st.session_state.get("indicators", []),
            label_visibility="collapsed",
            placeholder="Select indicators..."
        )

        st = indicator_settings_loader_pane(st)

        st.markdown("**Settings**")
        selected_indicator = st.selectbox("Indicator", st.session_state.indicator_settings.keys(), label_visibility="collapsed")
        st = indicator_settings_pane(selected_indicator, st)

    if update_chart or "chart_data" in st.session_state:
        if update_chart:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

            st.session_state.chart_data = fetch_stock_data(ticker, (start_date, end_date), interval)
            st.session_state.chart_data = process_data(st.session_state.chart_data)

            st.session_state.chart_data = add_indicator_data(
                st.session_state.chart_data,
                st.session_state.indicators,
                st.session_state.indicator_settings
            )

            if "S&R" in st.session_state.indicators:
                st.session_state.chart_data = add_support_resistance_data(
                    st.session_state.chart_data,
                    st.session_state.indicator_settings.get("S&R", {})
                )

            st.session_state.chart_fig = create_indicator_chart(st.session_state.chart_data, ticker, chart_type, st.session_state.indicators)

        st.plotly_chart(st.session_state.chart_fig, use_container_width=True)
        st.dataframe(
            st.session_state.chart_data.assign(
                Datetime=st.session_state.chart_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
            )
        )
    else:
        st.info("Click 'Update' in the sidebar to generate a chart.")

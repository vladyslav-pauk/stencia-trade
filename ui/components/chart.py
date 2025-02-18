import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.charts import create_chart, add_indicator_charts
from src.utils.indicators import add_indicator_data
from src.utils.data import fetch_stock_data, process_data
from .panels import indicator_settings_panel, indicator_settings_loader
from src.trend.support_resistance import add_support_resistance_data


def chart_tab(st):
    """Chart Tab: Displays stock chart with dynamic indicators and settings."""

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

        # ✅ Preserve indicators selection correctly
        if "indicators" not in st.session_state:
            st.session_state.indicators = []

        selected_indicators = st.multiselect(
            "Indicators",
            ["SMA", "EMA", "TDA", "S&R", "FIB", "TRE"],
            default=st.session_state.get("indicators", []),
            label_visibility="collapsed",
            placeholder="Select indicators..."
        )

        # ✅ Update session state when the selection changes
        if selected_indicators != st.session_state.indicators:
            st.session_state.indicators = selected_indicators

        # ✅ Ensure indicator settings persist
        if "indicator_settings" not in st.session_state:
            st.session_state.indicator_settings = {ind: {} for ind in ["SMA", "EMA", "TDA", "S&R", "FIB", "TRE"]}

        # Load settings
        st = indicator_settings_loader(st)

        # **Indicator-Specific Settings**
        st.markdown("**Settings**")
        indicator = st.selectbox("Indicator", st.session_state.indicator_settings.keys(), label_visibility="collapsed")
        st = indicator_settings_panel(indicator, st)

    # ✅ Ensure `update_chart` only triggers when necessary
    if update_chart or "chart_data" in st.session_state:
        if update_chart:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

            st.session_state.chart_data = fetch_stock_data(ticker, (start_date, end_date), interval)
            st.session_state.chart_data = process_data(st.session_state.chart_data)

            # ✅ Preserve selected indicators and settings
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

            st.session_state.chart_fig = create_chart(st.session_state.chart_data, ticker, chart_type)
            st.session_state.chart_fig = add_indicator_charts(
                st.session_state.chart_fig,
                st.session_state.chart_data,
                st.session_state.indicators
            )

        # ✅ Ensure the updated chart is displayed
        st.plotly_chart(st.session_state.chart_fig, use_container_width=True)

        # ✅ Preserve data formatting
        st.dataframe(
            st.session_state.chart_data.assign(
                Datetime=st.session_state.chart_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
            )
        )
    else:
        st.info("Click 'Update' in the sidebar to generate a chart.")

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

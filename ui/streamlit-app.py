import streamlit as st

import utils
from src.utils.data import fetch_stock_data, process_data
from src.utils.chart import create_chart, add_indicator_charts, add_support_resistance
from src.utils.indicators import add_indicator_data
from src.trends.support_resistance import add_support_resistance_data

## LAYOUT ##
st.set_page_config(layout="wide")
st.title('Trading Dashboard')

## SIDEBAR ##
# Chart settings section
st.sidebar.header('Chart Settings')
ticker = st.sidebar.text_input('Ticker', 'SPY')
time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '3mo', '1y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA', 'EMA', 'TDA', 'S&R'])

# Sidebar information section
# st.sidebar.subheader('About')
# st.sidebar.info(
#     'This dashboard provides stock data and various indicators. Use the sidebar to customize your view.'
# )

# Sidebar section for real-time stock prices of selected symbols
# st.sidebar.header('Real-Time Stock Prices')
# stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
# for symbol in stock_symbols:
#     real_time_data = fetch_stock_data(symbol, '1d', '1m')
#     if not real_time_data.empty:
#         real_time_data = process_data(real_time_data)
#         last_price = real_time_data['Close'].iloc[-1].values[0]
#
#         change = last_price - real_time_data['Open'].iloc[0].values[0]
#         pct_change = (change / real_time_data['Open'].iloc[0].values[0] ) * 100
#         st.sidebar.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

## CHART ##
if st.sidebar.button('Update'):
    data = fetch_stock_data(ticker, time_period)
    data = process_data(data)
    data = add_indicator_data(data)
    data = add_support_resistance_data(data)

    # Display main metrics
    # last_close, change, pct_change, high, low, volume = calculate_metrics(data)
    # st.metric(label=f"{ticker} Last Price", value=f"{last_close.squeeze():.2f} USD", delta=f"{change.squeeze():.2f} ({pct_change.squeeze():.2f}%)")
    #
    # col1, col2, col3 = st.columns(3)
    # col1.metric("High", f"{high.squeeze():.2f} USD")
    # col2.metric("Low", f"{low.squeeze():.2f} USD")
    # col3.metric("Volume", f"{volume.squeeze():,}")

    # Plot the stock price chart

    fig = create_chart(data, ticker, time_period, chart_type)
    fig = add_indicator_charts(fig, data, indicators)
    st.plotly_chart(fig, use_container_width=True)

    # Display historical data and technical indicator data
    # st.subheader('Historical Data')
    # st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])
    #
    # st.subheader('Technical Indicators')
    # st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']])

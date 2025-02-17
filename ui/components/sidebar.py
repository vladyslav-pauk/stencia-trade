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

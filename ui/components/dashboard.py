from src.utils.data import fetch_stock_data, process_data

def notifications_monitor(st):
    if st.session_state.monitor_thread is not None and st.session_state.monitor_thread.is_alive():
        # st.subheader("Monitoring Status")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.text("Last Checked")
        col1.metric(label="Last Checked", value=st.session_state.last_checked, label_visibility = "collapsed")

        col2.text("Current Price")
        col2.metric(
            label="Current Price",
            value=f"${st.session_state.current_price:.2f}" if st.session_state.current_price else "N/A",
            label_visibility = "collapsed"
        )

        col3.text("Volume")
        col3.metric(
            label="Volume",
            value=f"${st.session_state.current_price:.2f}" if st.session_state.current_price else "N/A",
            label_visibility="collapsed"
        )

        with col4:
            st.text("Last Signal")
            st.metric(
                label="Signals Detected",
                value=st.session_state.signal_count,
                label_visibility = "collapsed"
            )
            last_signal = st.session_state.last_signal
            if last_signal:
                st.metric(
                    label="Last Signal",
                    value=f"{last_signal['Action']} @ ${last_signal['Price']:.2f}",
                    label_visibility="collapsed"
                )

        with col5:
            st.markdown("&nbsp;")
            if st.button("Stop Notifications", type="primary"):
                st.session_state.stop_event.set()
                st.success("Stopped monitoring trading signals.")

    return st


def stock_monitor(st):
    st.header('Real-Time Stock Prices')
    stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT']
    for symbol in stock_symbols:
        real_time_data = fetch_stock_data(symbol, '1d', '1m')
        if not real_time_data.empty:
            real_time_data = process_data(real_time_data)
            last_price = real_time_data['Close'].iloc[-1].values[0]

            change = last_price - real_time_data['Open'].iloc[0].values[0]
            pct_change = (change / real_time_data['Open'].iloc[0].values[0] ) * 100
            st.metric(f"{symbol}", f"{last_price:.2f} USD", f"{change:.2f} ({pct_change:.2f}%)")

    # todo: add small green change with arrow
    return st


# def stock_metrics(st):
#     if st.sidebar.button('Update'):
#         data = fetch_stock_data(ticker, time_period)
#         data = process_data(data)
#         data = add_indicator_data(data)
#         data = add_support_resistance_data(data)
#
#         # Display main metrics
#         last_close, change, pct_change, high, low, volume = calculate_metrics(data)
#         st.metric(label=f"{ticker} Last Price", value=f"{last_close.squeeze():.2f} USD", delta=f"{change.squeeze():.2f} ({pct_change.squeeze():.2f}%)")
#
#         col1, col2, col3 = st.columns(3)
#         col1.metric("High", f"{high.squeeze():.2f} USD")
#         col2.metric("Low", f"{low.squeeze():.2f} USD")
#         col3.metric("Volume", f"{volume.squeeze():,}")
#
#         # Plot the stock price chart
#
#         fig = create_chart(data, ticker, time_period, chart_type)
#         fig = add_indicator_charts(fig, data, indicators)
#         st.plotly_chart(fig, use_container_width=True)
#
#         # Display historical data and technical indicator data
#         st.subheader('Historical Data')
#         st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])
#
#         st.subheader('Technical Indicators')
#         st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']])
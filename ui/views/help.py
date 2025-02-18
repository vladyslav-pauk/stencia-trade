def help_tab(st):
    """Displays help text based on the selected tab in the sidebar."""

    text = {
        "About": "Stencia-Trade is a Streamlit application that allows users to analyze and visualize stock data, backtest trading strategies, and search for trading opportunities.",
        "Chart": "This is the chart tab.",
        "Trader": "This is the trader tab.",
        "Screener": "This is the screener tab.",
        "Portfolio": "This is the portfolio tab."
    }

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "About"

    with st.sidebar:
        st.header("Help")
        selected_tab = st.radio("Contents", list(text.keys()),
                                index=list(text.keys()).index(st.session_state.selected_page))


        if selected_tab != st.session_state.selected_page:
            st.session_state.selected_page = selected_tab
            st.rerun()

    st.header(st.session_state.selected_page)
    st.write(text.get(st.session_state.selected_page))
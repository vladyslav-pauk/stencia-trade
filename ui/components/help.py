def help_tab(st):
    """Displays help text based on the selected tab in the sidebar."""

    # Help text for each tab
    text = {
        "About": "This is the about tab.",
        "Chart": "This is the chart tab.",
        "Trader": "This is the trader tab.",
        "Screener": "This is the screener tab.",
        "Portfolio": "This is the portfolio tab."
    }

    # Ensure session state has a valid tab
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "About"  # Default to About

    with st.sidebar:
        st.header("Help")
        selected_tab = st.radio("Contents", list(text.keys()),
                                index=list(text.keys()).index(st.session_state.selected_page))

        # Only update the session state if the selection changes
        if selected_tab != st.session_state.selected_page:
            st.session_state.selected_page = selected_tab
            st.rerun()  # Force a refresh to apply changes instantly

    # Display the selected tab's content
    st.header(st.session_state.selected_page)
    st.write(text.get(st.session_state.selected_page))
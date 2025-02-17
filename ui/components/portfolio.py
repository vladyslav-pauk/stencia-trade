import pandas as pd
import plotly.graph_objects as go

def portfolio_tab(st):
    """Displays portfolio holdings, value, and performance metrics."""

    # Initialize portfolio session state if not set
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Shares", "Avg Price", "Current Price", "Value", "Change %"])

    if "portfolio_history" not in st.session_state:
        st.session_state.portfolio_history = pd.DataFrame(columns=["Date", "Total Value"])

    with st.sidebar:
        st.header("Portfolio Settings")
        cash_balance = st.number_input("Starting Cash ($)", min_value=0.0, value=10000.0, step=100.0)
        refresh_portfolio = st.button("Refresh")

    # Example: Fetch latest stock prices (dummy implementation)
    def fetch_latest_prices():
        return {"AAPL": 175.0, "TSLA": 680.0, "NVDA": 450.0}

    # Update portfolio with latest prices
    if refresh_portfolio:
        latest_prices = fetch_latest_prices()
        for ticker in st.session_state.portfolio["Ticker"]:
            if ticker in latest_prices:
                st.session_state.portfolio.loc[st.session_state.portfolio["Ticker"] == ticker, "Current Price"] = latest_prices[ticker]
                st.session_state.portfolio["Value"] = st.session_state.portfolio["Shares"] * st.session_state.portfolio["Current Price"]
                st.session_state.portfolio["Change %"] = ((st.session_state.portfolio["Current Price"] - st.session_state.portfolio["Avg Price"]) / st.session_state.portfolio["Avg Price"]) * 100

        # Update total portfolio value history
        total_value = st.session_state.portfolio["Value"].sum() + cash_balance
        new_entry = pd.DataFrame({"Date": [pd.Timestamp.now()], "Total Value": [total_value]})
        st.session_state.portfolio_history = pd.concat([st.session_state.portfolio_history, new_entry])

    # Display Portfolio Table
    st.subheader("Current Holdings")
    st.dataframe(st.session_state.portfolio)

    # Compute Portfolio Metrics
    total_portfolio_value = st.session_state.portfolio["Value"].sum() + cash_balance
    total_change_pct = ((total_portfolio_value - cash_balance) / cash_balance) * 100

    # Display Metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Portfolio Value ($)", f"{total_portfolio_value:,.2f}")
    col2.metric("Total Change (%)", f"{total_change_pct:.2f}%", delta=total_change_pct)

    # Plot Portfolio Value Over Time
    st.subheader("Portfolio Value Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.portfolio_history["Date"],
        y=st.session_state.portfolio_history["Total Value"],
        mode="lines+markers",
        name="Portfolio Value",
        line=dict(color="blue", width=2, dash="solid")
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Total Value ($)",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trading_results(trade_data, full_data):
    """Generate a combined plot of price movement, support-resistance bands, and trade actions."""

    fig = go.Figure()

    # Plot Price Line
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data["Close"],
        mode='lines',
        name='Price',
        line=dict(color='rgba(0, 100, 255, 0.5)', width=2)
    ))

    # Plot Cumulative Returns
    fig.add_trace(go.Scatter(
        x=trade_data["Date"],
        y=trade_data["Cumulative Returns"] * full_data["Close"].iloc[0],
        mode="lines",
        name="Returns",
        line=dict(color="yellow", width=2),
    ))

    # --- Weekly Support Band ---
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data['Support2'],
        mode='lines',
        line=dict(color='rgba(0, 0, 255, 0)'),  # Invisible lower bound
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data['Support1'],
        mode='lines',
        name='Support Zone',
        line=dict(color='rgba(0, 0, 255, 0)'),  # Invisible upper bound
        fill='tonexty',
        fillcolor='rgba(255, 100, 100, 0.5)'
    ))

    # --- Weekly Resistance Band ---
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data['Resistance1'],
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0)'),  # Invisible lower bound
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data['Resistance2'],
        mode='lines',
        name='Resistance Zone',
        line=dict(color='rgba(255, 0, 0, 0)'),  # Invisible upper bound
        fill='tonexty',
        fillcolor='rgba(0, 255, 150, 0.2)'
    ))

    # Buy & Sell signals
    buy_signals = trade_data[trade_data["Action"] == "Buy"]
    sell_signals = trade_data[trade_data["Action"] == "Sell"]

    fig.add_trace(go.Scatter(
        x=buy_signals["Date"],
        y=buy_signals["Price"],
        mode='markers',
        marker=dict(color="green", size=8, symbol="triangle-up"),
        name="Buy Actions",
        legendgroup="buy_sell",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=sell_signals["Date"],
        y=sell_signals["Price"],
        mode='markers',
        marker=dict(color="red", size=8, symbol="triangle-down"),
        name="Sell Actions",
        legendgroup="buy_sell",
        showlegend=True
    ))

    fig.update_layout(
        # title="Trading Strategy: Support & Resistance",
        # xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def plot_trading_actions(st):
    fig = go.Figure()
    # fig = make_subplots(rows=3, cols=1)

    if "Datetime" not in st.session_state.trade_summary.columns:
        st.session_state.trade_summary.rename(columns={"Date": "Datetime"}, inplace=True)

    x = st.session_state.trade_summary["Datetime"]

    fig.add_trace(go.Bar(
        x=x,
        y=st.session_state.trade_summary["Returns"],
        marker=dict(color=["green" if r > 0 else "red" for r in st.session_state.trade_summary["Returns"]]),
        showlegend=False
    ))
    # todo: separate legend for buy and sell

    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=st.session_state.trade_summary["Cumulative Returns"] - 1,
    #     mode="lines",
    #     name="Cumulative Returns",
    #     line=dict(color="blue", width=2)
    # ))
    #
    # fig.update_layout(
    #     # title="Trading Strategy: Support & Resistance",
    #     # xaxis_title="Date",
    #     yaxis_title="Price",
    #     height=300,
    #     margin=dict(l=0, r=0, t=0, b=0),
    # )
    fig.update_xaxes(showticklabels=False)
    # todo: relative to buy&hold

    return fig

# todo: make subplots so it's scaled togeher when zoomed, the action suplot has 3 times smaller hegith
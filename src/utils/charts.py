import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def colors(alpha=0.5):
    colors = {
        'support': f'rgba(255, 100, 100, {alpha})',
        'resistance': f'rgba(0, 255, 150, {alpha})',
        'pivot': f'rgba(155, 165, 190, {alpha})',
        'neutral': f'rgba(255, 165, 0, {alpha})',
        'EMA': 'orange',
        'SMA': 'yellow',
        'TDA': 'purple'
    }
    return colors


def create_chart(data, ticker, chart_type):
    """
    Create a Plotly chart based on the given data and parameters.
    """
    # range_breaks = []
    # if time_period == '3mo':
    #     range_breaks = [dict(bounds=["fri", "sun"])]
    # elif time_period == '1wk':
    #     range_breaks = [dict(bounds=[16, 9], pattern="hour")]

    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data['Datetime'],
            open=data['Open'].squeeze(),
            high=data['High'].squeeze(),
            low=data['Low'].squeeze(),
            close=data['Close'].squeeze(),
            name="Candlestick")
        )
    else:
        fig = px.line(
            data,
            x='Datetime',
            y=data['Close'].squeeze()
        )


    fig.update_xaxes(
        showgrid=True,
        # gridcolor="rgba(200, 200, 200, 0.5)",  # Light gray grid lines
        # griddash="dot",
    )

    fig.update_layout(
        title=f'Prices and Indicators',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        height=800,
        xaxis=dict(
            rangeslider=dict(visible=True),  # Adds zoom slider
            type="date",
            # rangebreaks=range_breaks
        ),
        yaxis=dict(
            automargin=True,  # Ensures margins auto-adjust
            rangemode="normal",  # Automatically adjusts Y-axis range
            fixedrange=False  # Allows zooming in
        ),
        margin=dict(l=10, r=10, t=30, b=70),
    )

    # fig.update_xaxes(rangebreaks=[
    #     dict(bounds=["sat", "mon"]),  # Remove weekends
    #     dict(bounds=[16, 9.5], pattern="hour")  # Remove non-trading hours (16:00 to 9:30)
    # ])
    fig.layout.template = 'plotly_dark'
    return fig


def add_support_resistance(fig, data):
    """
    Add multiple levels of support and resistance bands dynamically.

    Args:
        fig (go.Figure): The Plotly figure to modify.
        data (pd.DataFrame): Data containing support and resistance levels.
        levels (int): Number of support & resistance levels to plot.
        alpha (float): Transparency for the shaded regions (0 to 1).
    """
    levels = data.filter(like='Support').shape[1]

    # Dynamically add multiple support levels
    for i in range(1, levels + 1):
        support_lower = f'Support{i + 1}' if i + 1 <= levels + 1 else f'Support{i}'
        support_upper = f'Support{i}'

        if support_lower in data and support_upper in data:
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data[support_lower], mode='lines',
                line=dict(color='rgba(0, 0, 255, 0)'), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data[support_upper], mode='lines',
                name=f'Support {i}', line=dict(color='rgba(0, 0, 255, 0)'),
                fill='tonexty', fillcolor=colors(i / (levels + 1))['support']
            ))

    # Dynamically add multiple resistance levels
    for i in range(1, levels + 1):
        resistance_lower = f'Resistance{i}'
        resistance_upper = f'Resistance{i + 1}' if i + 1 <= levels + 1 else f'Resistance{i}'

        if resistance_lower in data and resistance_upper in data:
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data[resistance_lower], mode='lines',
                line=dict(color=f'rgba(255, 165, 0, 0)'), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=data['Datetime'], y=data[resistance_upper], mode='lines',
                name=f'Resistance {i}', line=dict(color=f'rgba(255, 165, 0, 0)'),
                fill='tonexty', fillcolor=colors(i / (levels + 1))['resistance']
            ))

    # Pivot Line
    if 'Pivot' in data:
        fig.add_trace(go.Scatter(
            x=data['Datetime'], y=data['Pivot'], mode='lines',
            name='Pivot', line=dict(color=colors()['pivot'], width=1, dash='dash')
        ))

    return fig


def add_indicator_charts(fig, data, indicators):
    for indicator in indicators:
        if indicator == 'S&R':
            fig = add_support_resistance(fig, data)
        else:
            fig.add_trace(go.Scatter(
                x=data['Datetime'],
                y=data[indicator],
                name=indicator,
                marker=dict(color=colors()[indicator]),
                line=dict(width=1)
            ))
    return fig


# def create_trader_chart(st):
#     """Creates a two-row Plotly subplot: Price + Trading Actions."""
#
#     # Create subplots with shared x-axis
#     fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
#                         row_heights=[0.7, 0.3],  # 70% for price, 30% for actions
#                         vertical_spacing=0.02)
#
#     # Ensure required data is available
#     if "trade_summary" not in st.session_state or "trader_data" not in st.session_state:
#         return go.Figure()  # Return empty figure if no data
#
#     # Get sub-figures
#     price_chart = plot_trading_results(st.session_state.trade_summary, st.session_state.trader_data)
#     actions_chart = plot_trading_actions(st)
#
#     # Add price chart traces (first row)
#     for trace in price_chart["data"]:
#         fig.add_trace(trace, row=1, col=1)
#
#     # Add action chart traces (second row)
#     for trace in actions_chart["data"]:
#         fig.add_trace(trace, row=2, col=1)
#
#     # Remove x-axis labels from the first plot (for alignment)
#     fig.update_xaxes(showticklabels=False, row=1, col=1)
#
#     # Set layout configurations
#     fig.update_layout(
#         height=700,
#         showlegend=True,
#         margin=dict(l=10, r=10, t=10, b=10),
#     )
#
#     return fig

def create_trader_chart(st):
    """Creates a two-row Plotly subplot: Price + Trading Actions with Buy & Hold line."""

    # ✅ Ensure required data is available
    if "trade_summary" not in st.session_state or "trader_data" not in st.session_state:
        return go.Figure()  # Return empty figure if no data

    trade_summary = st.session_state.trade_summary
    trader_data = st.session_state.trader_data

    # ✅ Extract required data
    x_dates = trade_summary["Date"]
    trade_returns = trade_summary["Returns"]
    cumulative_returns = trade_summary["Cumulative Returns"]

    # ✅ Extrapolate cumulative returns to start and end
    start_date, end_date = trader_data.index.min(), trader_data.index.max()
    cumulative_returns_extended = np.concatenate(
        ([cumulative_returns.iloc[0]], cumulative_returns, [cumulative_returns.iloc[-1]]))
    x_dates_extended = np.concatenate(([start_date], x_dates, [end_date]))

    # ✅ Compute Buy & Hold Returns (log-scale to avoid distortions)
    buy_hold_start = trader_data["Close"].iloc[0]
    buy_hold_returns = trader_data["Close"] / buy_hold_start  # Normalize to 1.0 at start
    buy_hold_extended = np.concatenate(([1.0], buy_hold_returns.loc[x_dates], [buy_hold_returns.iloc[-1]]))

    # ✅ Create Subplots (shared x-axis)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.8, 0.2],  # 70% for price, 30% for actions
                        vertical_spacing=0.02)

    # ✅ Price Chart (Row 1)
    price_chart = plot_trading_results(trade_summary, trader_data)
    for trace in price_chart["data"]:
        fig.add_trace(trace, row=1, col=1)

    # ✅ Action Chart (Row 2)
    actions_chart = plot_trading_actions(st)
    for trace in actions_chart["data"]:
        fig.add_trace(trace, row=2, col=1)

    fig.update_xaxes(
        showgrid=True,
        # gridcolor="rgba(200, 200, 200, 0.5)",  # Light gray grid lines
        # griddash="dot",
    )

    # Buy & Hold Line
    # fig.add_trace(go.Scatter(
    #     x=x_dates_extended,
    #     y=st.session_state.trade_summary["Price"] / st.session_state.trade_summary["Price"].iloc[0] - 1,
    #     mode="lines",
    #     name="Buy & Hold",
    #     line=dict(color="gray", width=2, dash="dash"),
    # ), row=2, col=1)

    # # ✅ Cumulative Returns (Extended)
    # fig.add_trace(go.Scatter(
    #     x=x_dates_extended,
    #     y=cumulative_returns_extended,
    #     mode="lines",
    #     name="Cumulative Returns",
    #     line=dict(color="blue", width=2),
    # ), row=2, col=1)

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title='Time', row=2, col=1)
    fig.update_yaxes(title='Return (%)', row=2, col=1)

    # ✅ Improve Layout
    fig.update_layout(
        title="Trader",
        # xaxis_title='Time',
        yaxis_title='Price (USD)',
        height=800,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=70),
    )

    return fig


def plot_trading_results(trade_data, full_data):
    """Generate a combined plot of price movement, support-resistance bands, and trade actions."""

    fig = go.Figure()

    # Plot Price Line
    fig.add_trace(go.Scatter(
        x=full_data.index,
        y=full_data["Close"],
        mode='lines',
        name='Price',
        line=dict(color='rgba(0, 100, 255, 0.5)', width=1)
    ))

    # Plot Cumulative Returns
    fig.add_trace(go.Scatter(
        x=trade_data["Date"],
        y=trade_data["Cumulative Returns"],
        mode="lines",
        name="Cumulative Returns",
        line=dict(color="orange", width=1),
    ))

    # --- Support Band ---
    # fig.add_trace(go.Scatter(
    #     x=full_data.index,
    #     y=full_data['Support2'],
    #     mode='lines',
    #     line=dict(color='rgba(0, 0, 255, 0)'),  # Invisible lower bound
    #     showlegend=False,
    #     hoverinfo='skip'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=full_data.index,
    #     y=full_data['Support1'],
    #     mode='lines',
    #     name='Support Zone',
    #     line=dict(color='rgba(0, 0, 255, 0)'),  # Invisible upper bound
    #     fill='tonexty',
    #     fillcolor=colors()['support']
    # ))

    # --- Resistance Band ---
    # fig.add_trace(go.Scatter(
    #     x=full_data.index,
    #     y=full_data['Resistance1'],
    #     mode='lines',
    #     line=dict(color='rgba(255, 0, 0, 0)'),  # Invisible lower bound
    #     showlegend=False,
    #     hoverinfo='skip'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=full_data.index,
    #     y=full_data['Resistance2'],
    #     mode='lines',
    #     name='Resistance Zone',
    #     line=dict(color='rgba(255, 0, 0, 0)'),  # Invisible upper bound
    #     fill='tonexty',
    #     fillcolor=colors()['resistance']
    # ))

    fig = add_support_resistance(fig, full_data)

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

    fig.add_trace(go.Scatter(
        x=x,
        y=st.session_state.trade_summary["Relative Returns"] - 1,
        mode="lines",
        name="Relative Returns",
        line=dict(color="yellow", width=1),
        # showlegend=False
    ))

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


def plot_screener_results(df):
    """Generate a Plotly horizontal bar chart for buy/sell/neutral recommendations."""
    # Define category order for sorting
    order_categories = {"STRONG_BUY": 5, "BUY": 4, "NEUTRAL": 3, "SELL": 2, "STRONG_SELL": 1}
    df["Order"] = df["RECOMMENDATION"].map(order_categories)
    df = df.sort_values("Order", ascending=True).reset_index(drop=True)

    # Create Plotly bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["Ticker"],
        x=df["BUY"],
        name="Buy Indicators",
        marker_color=colors()['resistance'],
        orientation="h"
    ))

    fig.add_trace(go.Bar(
        y=df["Ticker"],
        x=df["NEUTRAL"],
        name="Neutral Indicators",
        marker_color=colors()['neutral'],
        orientation="h"
    ))

    fig.add_trace(go.Bar(
        y=df["Ticker"],
        x=df["SELL"],
        name="Sell Indicators",
        marker_color=colors()['support'],
        orientation="h"
    ))

    # Add annotations for recommendations on the right-hand side of each bar
    for i, recommendation in enumerate(df["RECOMMENDATION"]):
        fig.add_annotation(
            x=df["BUY"].iloc[i] + df["NEUTRAL"].iloc[i] + df["SELL"].iloc[i] + 3,  # Position slightly outside the bars
            y=df["Ticker"].iloc[i],
            text=recommendation,
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="black",
            bordercolor="black"
        )

    # Update layout
    fig.update_layout(
        title="Stock Screener Recommendations",
        barmode="stack",
        xaxis_title="Number of Indicators",
        yaxis_title="Tickers",
        height=700,
        margin=dict(l=10, r=100, t=40, b=10)  # Extra right margin for labels
    )

    return fig
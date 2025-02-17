import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from src.trade.charts import plot_trading_results, plot_trading_actions
from src.utils.helpers import title


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
        title=f'{ticker.upper()}',
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
    """Add weekly support and resistance bands to the chart."""
    fig.add_trace(go.Scatter(
        x=data['Datetime'], y=data['Support2'], mode='lines',
        line=dict(color='rgba(0, 0, 255, 0)'), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'], y=data['Support1'], mode='lines',
        name='Support', line=dict(color='rgba(0, 0, 255, 0)'),
        fill='tonexty', fillcolor='rgba(0, 0, 255, 0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'], y=data['Resistance1'], mode='lines',
        line=dict(color='rgba(255, 165, 0, 0)'), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'], y=data['Resistance2'], mode='lines',
        name='Resistance', line=dict(color='rgba(255, 165, 0, 0)'),
        fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'], y=data['Pivot'], mode='lines',
        name='Pivot', line=dict(color='blue', width=2, dash='dash')
    ))
    return fig


def add_indicator_charts(fig, data, indicators):
    for indicator in indicators:
        # if indicator == 'SMA 20':
        #     fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
        # elif indicator == 'EMA 20':
        #     fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))
        # elif indicator == 'TDA':
        #     fig.add_trace(go.Scatter(x=data['Datetime'], y=data['TDA'], name='TDA'))
        # elif indicator == 'S&R':

        if indicator == 'S&R':
            fig = add_support_resistance(fig, data)
        else:
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data[indicator], name=indicator))
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
        height=700,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=70),
    )

    return fig
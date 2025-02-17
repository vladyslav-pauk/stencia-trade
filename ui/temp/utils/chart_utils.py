import plotly.graph_objects as go
import numpy as np


def create_chart(data, symbol):
    """
    Create a candlestick chart using Plotly.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    ))

    fig.update_layout(
        title=f"Stock Price: {symbol}",
        # xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=800,
        margin=dict(l=80, r=50, t=80, b=40)
    )
    return fig.to_html(full_html=False)


def add_tda_to_chart(fig, data, norms):
    """
    Add TDA norms as a time-series line to an existing Plotly figure.
    Arguments:
        fig: Existing Plotly figure (LPPLS chart).
        data: Pandas DataFrame with stock data.
        segment: Tuple representing start and end indices of the segment.
        norms: List of TDA norms for each sliding window in the segment.
    """
    # Prepare x-coordinates for TDA norms
    tda_x = data.index[segment[0]:segment[1] - len(norms) + 1]
    print(norms)

    # Add TDA norms as a line plot
    fig.add_trace(go.Scatter(
        x=tda_x,
        y=norms,
        mode="lines",
        name="TDA Norm",
        line=dict(color="purple", dash="dash")
    ))

    fig.update_layout(
        title="LPPLS with TDA Norm Overlay",
        yaxis_title="Price / TDA Norm",
    )
    return fig


def add_segments_with_lppls_to_chart(data, symbol, segments, lppls_fits):
    """
    Add interchanging shaded regions for segments and overlay multiple LPPLS fits with unified legend entries.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    ))

    # Add segments as shaded regions with unified legend entry
    for idx, (start, end) in enumerate(segments):
        segment_x = data.index[start:end + 1]
        segment_y = [data['High'].max()] * len(segment_x)

        fig.add_trace(go.Scatter(
            x=list(segment_x) + list(segment_x[::-1]),
            y=[data['Low'].min()] * len(segment_x) + list(segment_y[::-1]),
            fill='toself',
            fillcolor="rgba(173, 216, 230, 0.3)" if idx % 2 == 0 else "rgba(0, 0, 0, 0)",
            line=dict(width=0),
            name="Segments" if idx == 0 else None,
            legendgroup="Segments",
            showlegend=idx == 0
        ))

    # Add LPPLS fits for each segment with unified legend entry
    for idx, (segment, params) in enumerate(lppls_fits):
        t = np.arange(segment[0], segment[1] + 1)
        log_prices = np.log(data['Close'].values[segment[0]:segment[1] + 1])
        tc, m, omega, A, B, C1, C2 = params

        trend = A + B * (tc - t) ** m
        oscillations = C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) + \
                       C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
        fitted = trend + oscillations

        fig.add_trace(go.Scatter(
            x=data.index[segment[0]:segment[1] + 1],
            y=np.exp(fitted),
            mode="lines",
            name="LPPLS Fit" if idx == 0 else None,
            line=dict(color="orange", dash="dot"),
            legendgroup="LPPLS",
            showlegend=idx == 0
        ))

    # Update chart layout
    fig.update_layout(
        title=f"Stock Price: {symbol}",
        # xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=800
    )
    return fig.to_html(full_html=False)


def add_lppls_to_chart(data, symbol, segment, params):
    """
    Overlay LPPLS fit on the chart.
    """
    t = np.arange(segment[0], segment[1] + 1)
    log_prices = np.log(data['Close'].values[segment[0]:segment[1] + 1])
    tc, m, omega, A, B, C1, C2 = params

    trend = A + B * (tc - t) ** m
    oscillations = C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) + \
                   C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
    fitted = trend + oscillations

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    ))

    fig.add_trace(go.Scatter(
        x=data.index[segment[0]:segment[1] + 1],
        y=np.exp(fitted),
        mode="lines",
        name="LPPLS Fit",
        line=dict(color="orange", dash="dot")
    ))

    fig.update_layout(
        title=f"Stock Price: {symbol} (with LPPLS)",
        # xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=800
    )
    return fig.to_html(full_html=False)
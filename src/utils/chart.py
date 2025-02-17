import plotly.graph_objects as go
import plotly.express as px

def create_chart(data, ticker, time_period, chart_type):
    """
    Create a Plotly chart based on the given data and parameters.
    """
    range_breaks = []
    if time_period == '3mo':
        range_breaks = [dict(bounds=["fri", "sun"])]
    elif time_period == '1wk':
        range_breaks = [dict(bounds=[16, 9], pattern="hour")]

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
    fig.update_layout(
        title=f'{ticker} {time_period.upper()} Chart',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        height=800,
        xaxis=dict(
            rangeslider=dict(visible=True),  # Adds zoom slider
            type="date",
            rangebreaks=range_breaks
        ),
        yaxis=dict(
            automargin=True,  # Ensures margins auto-adjust
            rangemode="normal",  # Automatically adjusts Y-axis range
            fixedrange=False  # Allows zooming in
        )
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
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data[indicator.replace(' ', '_')], name=indicator))
    return fig
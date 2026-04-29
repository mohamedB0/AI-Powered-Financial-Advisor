import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any


def candlestick_chart(rows: List[Dict[str, Any]], ticker: str) -> go.Figure:
    """
    Create a candlestick chart from price history data.
    Expects rows with: date, open, high, low, close, volume
    """
    if not rows:
        return go.Figure()

    from decimal import Decimal

    def to_float(val):
        if isinstance(val, Decimal):
            return float(val)
        return val

    df = {
        'date': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    }

    for r in rows:
        df['date'].append(r.get('date', r.get('Date')))
        df['open'].append(to_float(r.get('open', r.get('Open', 0))))
        df['high'].append(to_float(r.get('high', r.get('High', 0))))
        df['low'].append(to_float(r.get('low', r.get('Low', 0))))
        df['close'].append(to_float(r.get('close', r.get('Close', 0))))
        df['volume'].append(int(to_float(r.get('volume', r.get('Volume', 0)))))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('', 'Volume')
    )

    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=ticker,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volume',
            marker_color='#90caf9'
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_white',
        height=500,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        title=f'{ticker} Price Chart',
        yaxis_title='Price (USD)',
        xaxis_title='Date'
    )

    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)

    return fig


def revenue_bar_chart(rows: List[Dict[str, Any]], company: str) -> go.Figure:
    """
    Create a bar chart for financial metrics (revenue, net income).
    """
    if not rows:
        return go.Figure()

    from decimal import Decimal
    def to_float(val):
        if isinstance(val, Decimal):
            return float(val)
        return val

    fig = go.Figure()

    years = [str(r.get('year', '')) for r in rows]
    revenues = [to_float(r.get('revenue', 0) or 0) / 1e9 for r in rows]
    net_incomes = [to_float(r.get('net_income', 0) or 0) / 1e9 for r in rows]

    fig.add_trace(go.Bar(x=years, y=revenues, name='Revenue (B)', marker_color='#26a69a'))
    fig.add_trace(go.Bar(x=years, y=net_incomes, name='Net Income (B)', marker_color='#42a5f5'))

    fig.update_layout(
        template='plotly_white',
        height=500,
        title=f'{company} Financials',
        xaxis_title='Year',
        yaxis_title='Amount (Billions USD)',
        barmode='group'
    )

    return fig


def macro_line_chart(rows: List[Dict[str, Any]], indicator: str, country: str = "") -> go.Figure:
    """
    Create a line chart for macro indicators over time.
    """
    if not rows:
        return go.Figure()

    from decimal import Decimal
    def to_float(val):
        if isinstance(val, Decimal):
            return float(val)
        return val

    dates = [r.get('date') for r in rows]
    values = [to_float(r.get('value', 0)) for r in rows]

    title = f"{indicator} Over Time"
    if country:
        title = f"{indicator} - {country}"

    fig = go.Figure(go.Scatter(x=dates, y=values, mode='lines+markers', name=indicator))
    fig.update_layout(
        template='plotly_white',
        height=400,
        title=title,
        xaxis_title='Date',
        yaxis_title='Value'
    )

    return fig


def movers_bar(rows: List[Dict[str, Any]], metric: str = 'pct_change') -> go.Figure:
    """
    Create a horizontal bar chart for top movers.
    """
    if not rows:
        return go.Figure()

    tickers = [r.get('ticker', '') for r in rows]
    values = [float(r.get(metric, 0)) for r in rows]

    fig = go.Figure(go.Bar(x=values, y=tickers, orientation='h', marker_color='#26a69a'))
    fig.update_layout(
        template='plotly_white',
        height=300,
        title=f'Top Movers by {metric}',
        xaxis_title=metric,
        yaxis_title='Ticker'
    )

    return fig
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime


class Visualization:
    # Modern stock market color scheme
    COLOR_BULLISH = '#00CC96'  # Green
    COLOR_BEARISH = '#EF553B'  # Red
    COLOR_NEUTRAL = '#636EFA'  # Blue
    COLOR_BACKGROUND = '#1E1E1E'  # Dark background
    COLOR_GRID = '#2A2A2A'  # Grid color

    @staticmethod
    def create_stock_chart(df, predictions=None):
        """
        Create an interactive stock chart with technical indicators
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price', 'RSI', 'MACD', 'Volume'),
            row_width=[0.2, 0.2, 0.2, 0.4]
        )

        # Price data
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ), row=1, col=1
        )

        # Moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FF7F0E', width=1), name='MA20'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#1F77B4', width=1), name='MA50'),
            row=1, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], line=dict(color=Visualization.COLOR_NEUTRAL, width=1), name='RSI'),
            row=2, col=1
        )
        # Add RSI markers
        fig.add_hline(y=70, line_dash="dash", line_color=Visualization.COLOR_BEARISH, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=Visualization.COLOR_BULLISH, row=2, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], line=dict(color=Visualization.COLOR_NEUTRAL, width=1), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color=Visualization.COLOR_BULLISH, width=1),
                       name='Signal'),
            row=3, col=1
        )
        # MACD histogram
        colors = np.where(df['MACD_Histogram'] < 0, Visualization.COLOR_BEARISH, Visualization.COLOR_BULLISH)
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color=colors),
            row=3, col=1
        )

        # Volume
        colors = np.where(df['Close'] > df['Open'], Visualization.COLOR_BULLISH, Visualization.COLOR_BEARISH)
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
            row=4, col=1
        )

        # Update layout
        fig.update_layout(
            title='Technical Analysis',
            template='plotly_dark',
            height=800,
            showlegend=True,
            plot_bgcolor=Visualization.COLOR_BACKGROUND,
            paper_bgcolor=Visualization.COLOR_BACKGROUND,
            font=dict(color='white')
        )

        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)

        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='RSI', row=2, col=1)
        fig.update_yaxes(title_text='MACD', row=3, col=1)
        fig.update_yaxes(title_text='Volume', row=4, col=1)

        return fig

    @staticmethod
    def create_sentiment_chart(sentiment_scores):
        """
        Create a sentiment analysis chart
        """
        # Count sentiment labels
        labels = [s.get('label', 'NEUTRAL') for s in sentiment_scores]
        positive_count = labels.count('POSITIVE')
        negative_count = labels.count('NEGATIVE')
        neutral_count = len(labels) - positive_count - negative_count

        # Create pie chart
        fig = px.pie(
            values=[positive_count, negative_count, neutral_count],
            names=['Positive', 'Negative', 'Neutral'],
            title='News Sentiment Distribution',
            color=['Positive', 'Negative', 'Neutral'],
            color_discrete_map={
                'Positive': Visualization.COLOR_BULLISH,
                'Negative': Visualization.COLOR_BEARISH,
                'Neutral': Visualization.COLOR_NEUTRAL
            }
        )

        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor=Visualization.COLOR_BACKGROUND,
            paper_bgcolor=Visualization.COLOR_BACKGROUND,
            font=dict(color='white')
        )

        return fig

    @staticmethod
    def create_prediction_gauge(prediction, confidence):
        """
        Create a prediction gauge chart
        """
        # Determine color based on prediction
        if prediction == "Bullish":
            color = Visualization.COLOR_BULLISH
        else:
            color = Visualization.COLOR_BEARISH

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Prediction: {prediction}", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': Visualization.COLOR_BEARISH},
                    {'range': [30, 70], 'color': Visualization.COLOR_NEUTRAL},
                    {'range': [70, 100], 'color': Visualization.COLOR_BULLISH}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence}}))

        fig.update_layout(
            height=400,
            template='plotly_dark',
            plot_bgcolor=Visualization.COLOR_BACKGROUND,
            paper_bgcolor=Visualization.COLOR_BACKGROUND,
            font=dict(color='white')
        )

        return fig


def format_large_number(num):
    """
    Format large numbers for display (e.g., 1,000,000 -> 1M)
    """
    if num is None:
        return "N/A"

    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"
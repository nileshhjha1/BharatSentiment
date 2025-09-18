import streamlit as st
import pandas as pd
import numpy as np
from data_acquisition import DataAcquirer
from models import TechnicalModel, SentimentModel, FusionModel
from utils import Visualization, format_large_number
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="BharatSentiment - Indian Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00CC96;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #636EFA;
        border-bottom: 1px solid #2A2A2A;
        padding-bottom: 0.5rem;
    }
    .stock-card {
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .positive {
        color: #00CC96;
    }
    .negative {
        color: #EF553B;
    }
    .disclaimer {
        background-color: #2A2A2A;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_acquired' not in st.session_state:
    st.session_state.data_acquired = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# App title
st.markdown('<h1 class="main-header">BharatSentiment</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #636EFA;">Multi-Modal Indian Stock Predictor</h3>',
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
stocks_list = ["RELIANCE", "TATASTEEL", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "SBIN", "HINDUNILVR", "BAJFINANCE",
               "KOTAKBANK"]
selected_stock = st.sidebar.selectbox("Select Stock", stocks_list)
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
news_api_key = st.sidebar.text_input("News API Key (Optional)", type="password")
analyze_button = st.sidebar.button("Analyze Stock", type="primary")

# Disclaimer
st.sidebar.markdown("""
<div class="disclaimer">
    <strong>Important Disclaimer:</strong> This application is for educational and research purposes only. 
    The predictions made by the AI model are not financial advice and should not be considered as a recommendation 
    to buy, sell, or hold any security. The stock market is subject to significant risks, and past performance 
    is not indicative of future results.
</div>
""", unsafe_allow_html=True)

# Main content
if analyze_button:
    with st.spinner("Fetching data and analyzing..."):
        # Initialize data acquirer
        acquirer = DataAcquirer()

        # Get stock data
        df, pe_ratio, pb_ratio, market_cap = acquirer.get_stock_data(selected_stock, period)

        if df is not None:
            st.session_state.df = df
            st.session_state.pe_ratio = pe_ratio
            st.session_state.pb_ratio = pb_ratio
            st.session_state.market_cap = market_cap
            st.session_state.data_acquired = True

            # Get news sentiment
            headlines, sentiment_scores = acquirer.get_news_sentiment(selected_stock, news_api_key)
            st.session_state.headlines = headlines
            st.session_state.sentiment_scores = sentiment_scores

            # Initialize models
            technical_model = TechnicalModel()
            sentiment_model = SentimentModel()
            fusion_model = FusionModel()

            # Prepare technical data
            X_tech, y_tech = technical_model.prepare_data(df)

            if len(X_tech) > 0:
                # Make prediction (in a real app, this would use a trained model)
                tech_prediction = np.mean(y_tech[-10:])  # Simplified for demo

                # Analyze sentiment
                analyzed_sentiment = sentiment_model.analyze_sentiment(headlines)

                # Prepare fundamental data
                fundamental_data = {
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'market_cap': market_cap
                }

                # Fusion model prediction (simplified for demo)
                fusion_features = fusion_model.prepare_features(
                    tech_prediction, analyzed_sentiment, fundamental_data
                )

                # Simulate prediction (in a real app, this would use a trained model)
                bull_prob = 0.6 if tech_prediction > df['Close'].iloc[-1] else 0.4
                bear_prob = 1 - bull_prob

                # Adjust based on sentiment
                positive_sentiment = sum(1 for s in analyzed_sentiment if s.get('label') == 'POSITIVE') / len(
                    analyzed_sentiment)
                bull_prob = min(1.0, bull_prob + (positive_sentiment * 0.2))
                bear_prob = 1 - bull_prob

                st.session_state.bull_prob = bull_prob
                st.session_state.bear_prob = bear_prob
                st.session_state.analysis_done = True
            else:
                st.error("Insufficient data for analysis. Please select a longer time period.")
        else:
            st.error("Failed to fetch stock data. Please try again later.")

# Display results if data is available
if st.session_state.data_acquired:
    df = st.session_state.df
    pe_ratio = st.session_state.pe_ratio
    pb_ratio = st.session_state.pb_ratio
    market_cap = st.session_state.market_cap

    # Stock overview section
    st.markdown('<h2 class="sub-header">Stock Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_close
        change_percent = (price_change / prev_close) * 100

        st.metric(
            label="Current Price",
            value=f"₹{current_price:.2f}",
            delta=f"{change_percent:.2f}%"
        )
    with col2:
        st.metric(label="P/E Ratio", value=f"{pe_ratio:.2f}" if pe_ratio else "N/A")
    with col3:
        st.metric(label="P/B Ratio", value=f"{pb_ratio:.2f}" if pb_ratio else "N/A")
    with col4:
        st.metric(label="Market Cap", value=format_large_number(market_cap) if market_cap else "N/A")

    # Technical chart
    st.plotly_chart(Visualization.create_stock_chart(df), use_container_width=True)

    # Display analysis results if available
    if st.session_state.analysis_done:
        st.markdown('<h2 class="sub-header">AI Analysis Results</h2>', unsafe_allow_html=True)

        bull_prob = st.session_state.bull_prob
        bear_prob = st.session_state.bear_prob
        headlines = st.session_state.headlines
        sentiment_scores = st.session_state.sentiment_scores

        # Prediction gauge
        if bull_prob > bear_prob:
            prediction = "Bullish"
            confidence = bull_prob * 100
        else:
            prediction = "Bearish"
            confidence = bear_prob * 100

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(Visualization.create_prediction_gauge(prediction, confidence), use_container_width=True)

        with col2:
            st.plotly_chart(Visualization.create_sentiment_chart(sentiment_scores), use_container_width=True)

            # Display key metrics
            st.info(f"""
            **Analysis Summary:**
            - **Technical Indicators:** {'Bullish' if bull_prob > 0.6 else 'Bearish' if bear_prob > 0.6 else 'Neutral'}
            - **News Sentiment:** {sum(1 for s in sentiment_scores if s.get('label') == 'POSITIVE') / len(sentiment_scores) * 100:.1f}% Positive
            - **Fundamental Strength:** {'Strong' if pe_ratio and pe_ratio < 25 else 'Average' if pe_ratio and pe_ratio < 40 else 'Weak' if pe_ratio else 'N/A'}
            """)

        # News sentiment analysis
        st.markdown('<h3 class="sub-header">Recent News Sentiment</h3>', unsafe_allow_html=True)

        for i, (headline, sentiment) in enumerate(zip(headlines, sentiment_scores)):
            sentiment_label = sentiment.get('label', 'NEUTRAL')
            sentiment_score = sentiment.get('score', 0.5)

            if sentiment_label == 'POSITIVE':
                icon = "🟢"
                color_class = "positive"
            elif sentiment_label == 'NEGATIVE':
                icon = "🔴"
                color_class = "negative"
            else:
                icon = "🔵"
                color_class = ""

            st.markdown(f"""
            <div class="stock-card">
                <p>{icon} <span class="{color_class}">{headline}</span> (Confidence: {sentiment_score:.2f})</p>
            </div>
            """, unsafe_allow_html=True)

            if i >= 4:  # Limit to 5 headlines
                break

# Initial state message
if not st.session_state.data_acquired:
    st.info("""
    Welcome to BharatSentiment! 

    This application uses machine learning and deep learning techniques to analyze Indian stocks by combining:
    - Technical indicators from price data
    - Sentiment analysis from news and social media
    - Fundamental analysis of company metrics

    **To get started:**
    1. Select a stock from the sidebar
    2. Choose a time period for analysis
    3. Click the "Analyze Stock" button
    """)

    # Sample chart for illustration
    st.plotly_chart(Visualization.create_stock_chart(pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [95, 96, 97, 98, 99],
        'Close': [102, 103, 101, 105, 107],
        'Volume': [1000000, 1200000, 800000, 1500000, 2000000],
        'MA20': [100, 100.5, 101, 101.5, 102],
        'RSI': [45, 50, 55, 60, 65],
        'MA50': [99, 99.5, 100, 100.5, 101],  # Added MA50
        'MACD': [0.1, 0.2, 0.3, 0.4, 0.5],
        'MACD_Signal': [0.05, 0.15, 0.25, 0.35, 0.45],
        'MACD_Histogram': [0.05, 0.05, 0.05, 0.05, 0.05],
        'BB_Middle': [100, 101, 102, 103, 104]
    })), use_container_width=True)
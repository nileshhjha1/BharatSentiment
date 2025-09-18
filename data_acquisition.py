import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time

# Download NLTK data (run once)
nltk.download('vader_lexicon')


class DataAcquirer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def get_stock_data(self, symbol, period="1y"):
        """
        Fetch historical stock data from Yahoo Finance
        """
        try:
            # For NSE symbols, add .NS suffix
            if not symbol.endswith('.NS'):
                symbol += '.NS'

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            # Calculate additional technical indicators
            df = self.calculate_technical_indicators(df)

            # Get fundamental data
            info = ticker.info
            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            market_cap = info.get('marketCap', None)

            return df, pe_ratio, pb_ratio, market_cap
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None, None, None, None

    def calculate_technical_indicators(self, df):
        """
        Calculate common technical indicators
        """
        # Moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        return df

    def get_news_sentiment(self, query, api_key=None, days=7):
        """
        Get news sentiment for a stock (simulated if no API key)
        """
        # Simulate news data if no API key provided
        if not api_key:
            return self.simulate_news_sentiment(query)

        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            # NewsAPI endpoint
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_date.strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={api_key}"

            response = requests.get(url)
            articles = response.json().get('articles', [])

            sentiments = []
            headlines = []

            for article in articles[:10]:  # Limit to 10 articles
                title = article.get('title', '')
                headlines.append(title)
                sentiment = self.sia.polarity_scores(title)
                sentiments.append(sentiment)

            return headlines, sentiments
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self.simulate_news_sentiment(query)

    def simulate_news_sentiment(self, query):
        """
        Simulate news sentiment for demonstration purposes
        """
        # Sample news headlines with varied sentiment
        sample_news = [
            f"{query} reports strong quarterly results with profit increase of 20%",
            f"Analysts upgrade {query} to buy rating citing growth potential",
            f"{query} faces regulatory challenges in new market",
            f"{query} announces new partnership to expand operations",
            f"Competition intensifies for {query} in core markets",
            f"{query} CEO expresses confidence in long-term strategy",
            f"Supply chain issues affecting {query} production capacity",
            f"{query} dividend announcement exceeds expectations",
            f"Market volatility impacts {query} stock performance",
            f"{query} innovation recognized with industry award"
        ]

        sentiments = []
        for headline in sample_news:
            sentiment = self.sia.polarity_scores(headline)
            sentiments.append(sentiment)

        return sample_news, sentiments


# Example usage
if __name__ == "__main__":
    acquirer = DataAcquirer()
    df, pe, pb, mc = acquirer.get_stock_data("RELIANCE")
    print(f"PE Ratio: {pe}, PB Ratio: {pb}, Market Cap: {mc}")
    print(df.tail())
# BharatSentiment AI 📈

A sophisticated multi-modal stock prediction application focused on the Indian market (NSE), leveraging Technical Analysis, News Sentiment Analysis, and Fundamental Data through a fusion AI model.

![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-TensorFlow%20%7C%20Scikit--learn-FF6F00?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Transformers%20%7C%20NLTK-green?style=for-the-badge)

## ✨ Features

*   **Multi-Modal Analysis**: Combines three distinct data sources for robust predictions.
    *   **Technical Analysis**: Utilizes LSTM neural networks on historical price data and indicators (RSI, MACD, Moving Averages, Bollinger Bands).
    *   **Sentiment Analysis**: Employs a pre-trained `distilbert` model from Hugging Face to analyze news headlines for market sentiment.
    *   **Fundamental Analysis**: Incorporates key valuation metrics like P/E Ratio, P/B Ratio, and Market Cap.
*   **Fusion AI Model**: A `RandomForestClassifier` that intelligently weights the predictions from each modality (Technical, Sentiment, Fundamental) to generate a final bull/bear probability.
*   **Interactive Dashboard**: A beautiful, responsive Streamlit web app with interactive Plotly charts for visual analysis.
*   **Real-time Data**: Fetches live stock data from Yahoo Finance (`yfinance`).
*   **News Integration**: Can pull real news headlines using the NewsAPI (with a provided key) or uses a realistic simulator for demonstration.

## 🚀 Live Demo

A live demo of the application is hosted on Streamlit Community Cloud:  
*(Link to be added after deployment)*

## 📸 Screenshots

*(You can add screenshots of your app here later)*
| Dashboard Overview | Technical Analysis | Sentiment Gauge |
| :---: | :---: | :---: |
| ![WhatsApp Image 2025-09-18 at 11 16 26_a0779e39](https://github.com/user-attachments/assets/9ea6224d-134b-41f2-873b-edada0e48290) | ![WhatsApp Image 2025-09-18 at 11 16 49_907f0522](https://github.com/user-attachments/assets/30f3ec39-c48d-42d6-9783-dceaaf4cf202) | ![WhatsApp Image 2025-09-18 at 11 17 35_4954e491](https://github.com/user-attachments/assets/d983001f-39be-4e93-8fcb-0c1a01f46a4f)|

## 🛠️ Installation & Setup

Follow these steps to run **BharatSentiment AI** locally.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/BharatSentiment-AI.git
    cd BharatSentiment-AI
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    # Using conda
    conda create -n bharatsentiment python=3.10
    conda activate bharatsentiment

    # Or using venv
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Set up NewsAPI**
    *   Get a free API key from [NewsAPI.org](https://newsapi.org/).
    *   You can enter this key directly in the app's sidebar when running, or set it as an environment variable.

5.  **Run the Application**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser at `http://localhost:8501`.

## 📊 Usage

1.  **Select a Stock**: Choose from a list of popular NSE stocks (e.g., RELIANCE, INFY, TATASTEEL).
2.  **Choose a Period**: Select the historical time period for analysis (1 month to 2 years).
3.  **Enter API Key (Optional)**: For live news analysis, enter your NewsAPI key in the sidebar.
4.  **Analyze**: Click the "Analyze Stock" button.
5.  **Interpret Results**:
    *   View the stock's current price, P/E ratio, and other metrics.
    *   Explore interactive technical charts.
    *   See the AI's final prediction (Bullish/Bearish) with a confidence gauge.
    *   Review the sentiment breakdown of recent news headlines.

## 🧠 Project Architecture
BharatSentiment-AI/
│
├── app.py # Main Streamlit application
├── data_acquisition.py # Fetches stock & news data, calculates indicators
├── models.py # ML models (Technical LSTM, Sentiment, Fusion)
├── utils.py # Visualization functions and helpers
├── requirements.txt # Python dependencies
│
└── .idea/ # PyCharm project files (can be ignored by Git) 

**Data Flow**:
1.  **Data Acquisition**: `yfinance` → OHLCV + Fundamentals → Technical Indicators calculated.
2.  **Technical Model**: Prepared data fed into an LSTM network for price trend prediction.
3.  **Sentiment Model**: News headlines are processed by a transformer model for sentiment scores.
4.  **Fusion Model**: Technical prediction, sentiment scores, and fundamental data are features for a final classifier.
5.  **Visualization**: Results are displayed via Plotly charts in the Streamlit UI.

## ⚠️ Important Disclaimer

**This application is for EDUCATIONAL and RESEARCH purposes ONLY.**

The predictions generated by this AI model are **not financial advice** and should **not** be construed as a recommendation to buy, sell, or hold any security or investment product. The stock market is inherently risky, and past performance is not indicative of future results. Always conduct your own due diligence and consult with a qualified financial advisor before making any investment decisions.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/BharatSentiment-AI/issues).

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

*   Stock data provided by [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`.
*   Sentiment analysis powered by [Hugging Face Transformers](https://huggingface.co/).
*   UI built with [Streamlit](https://streamlit.io/).
*   Visualizations created with [Plotly](https://plotly.com/).
     

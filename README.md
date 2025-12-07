# Financial News Sentiment and Stock Movement Prediction Using Transformer Models and Machine Learning

This repository contains an end-to-end system for predicting short-term stock price movements using:

- Transformer-based sentiment analysis (FinBERT)
- Technical indicators derived from OHLCV stock data
- Supervised machine learning models
- An interactive Streamlit dashboard
- An LLM assistant for interpretability and user-driven insights

The project integrates NLP, time-series feature engineering, explainable ML, and interactive visualization into a coherent financial prediction pipeline.

---

## 1. Project Overview

Financial markets respond quickly to new information, making sentiment extracted from financial news a valuable predictive signal.  
This project explores whether combining FinBERT-derived sentiment metrics with traditional technical indicators improves forecasting of next-day stock direction.

The system includes:

- Data preprocessing
- FinBERT sentiment scoring
- Hybrid feature engineering
- Classification modeling
- SHAP interpretability
- Streamlit tools for analysis and simulation
- A generative LLM assistant for explanations

---

## 2. System Architecture
```bash
Raw News (FNSPID) ──► Preprocessing ──► FinBERT Sentiment
                                   │
Stock Prices (OHLCV) ──► Technical Indicators
                                   │
                     ──────────────┴──────────────
                     ►     Hybrid Feature Set     ◄
                     ──────────────┬──────────────
                                   │
                              ML Models
                                   │
        ┌──────────────────────────┴─────────────────────────┐
        │                                                    │
Streamlit Insights Dashboard                          LLM Assistant
```

---

## 3. Data Sources

### 3.1 Financial News (FNSPID dataset)
- Over 13,000 financial news articles  
- Includes titles, summaries, timestamps, raw text, and ticker labels  
- Filtered to four supported tickers: **AAPL, MSFT, AMZN, GOOG**

### 3.2 Stock Price Data
Daily OHLCV data collected for the same tickers.

### 3.3 Date Alignment
All timestamps were converted to UTC and aggregated at daily granularity.  
Modeling window: **2023-09-01 to 2023-12-15**

---

## 4. Sentiment Extraction with FinBERT

The project uses the **ProsusAI/FinBERT** model (HuggingFace) to classify articles as Positive, Negative, or Neutral.

Process:

- Prefer Textrank summaries when available (reduces token length)
- Batch size: 32
- Truncation: 512 tokens
- Store sentiment label + confidence score

Output: `news_with_sentiment.csv`

---

## 5. Feature Engineering

### 5.1 Sentiment Features
Daily aggregated features:
- Mean sentiment score  
- Weighted sentiment  
- Positive/negative article counts  
- Sentiment confidence  

### 5.2 Technical Indicators
Derived from OHLCV data:
- Daily returns  
- Rolling volatility  
- SMA, EMA  
- RSI, MACD  

### 5.3 Hybrid Dataset
Sentiment + technical indicators combined into a unified feature set for modeling.

---

## 6. Predictive Modeling

Target variable:  
**1 = UP** (next-day close > current close)  
**0 = DOWN**

Models implemented:
- Logistic Regression  
- Random Forest  
- XGBoost  
- Simple ensemble  

Outputs include predicted direction, probability of upward movement (P(UP)), and confidence scores.

---

## 7. Explainability with SHAP

SHAP values allow decomposition of predictions into additive feature contributions.

Key insights:
- Rolling volatility is one of the strongest predictors  
- Mean sentiment and positive article count influence upward predictions  
- Sentiment interacts nonlinearly with technical indicators  

This improves interpretability and trust in the model.

---

## 8. Streamlit Dashboard

### 8.1 Insights Dashboard
Displays:
- Predictions for a selected date
- Probability of UP
- Price context
- Rolling volatility
- Sentiment metrics

### 8.2 Market Reaction Simulator
Allows users to simulate:
- Sentiment shocks  
- Volatility shocks  
- Hypothetical price movements  

### 8.3 News Browser
Shows all articles for a date with their FinBERT sentiment.

### 8.4 LLM Assistant
- Generates summaries  
- Provides explanations of predictions  
- Helps users interpret sentiment and model behavior  

---

## 9. Installation

### Clone Repository

```bash
git clone https://github.com/your-username/financial-news-sentiment.git
cd financial-news-sentiment
```

### Create Virtual Environment
```bash
python -m venv .venv
```

### Activation:

Windows:    .\.venv\Scripts\activate
Mac/Linux:  source .venv/bin/activate

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit App
```bash
streamlit run app/app.py
```

## 10. Repository Structure
```bash
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── news_with_sentiment.csv
│   └── testing_predictions_clean.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_finbert_sentiment.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_shap_interpretability.ipynb
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── insights.py
│   │   ├── simulator.py
│   │   ├── news_browser.py
│   │   └── llm_assistant.py
│   └── utils/
│       ├── preprocess.py
│       ├── loaders.py
│       ├── feature_engineering.py
│       ├── model_utils.py
│       └── finbert_client.py
│
├── README.md
└── requirements.txt
```

11. References
```bash
Araci, D. (2019). FinBERT: Financial Sentiment Analysis with BERT.
Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market.
ProsusAI. FinBERT documentation, HuggingFace.
```
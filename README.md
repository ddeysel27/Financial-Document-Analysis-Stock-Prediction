📊 Generative AI for Financial Document Analysis

Course Project — Fall 2025

📌 Overview

This project explores the intersection of financial news, market data, and AI.
We combine:

Transformers (FinBERT) for financial sentiment analysis

Machine Learning for stock price prediction

Generative AI (LLMs) for investor-friendly summaries

Dashboards (Python / PowerBi) for visualization and insights

🔄 Project Pipeline

Financial News Sentiment → Extract tone from market articles using FinBERT

Stock Price Prediction → Train ML models on sentiment + price data

Investor Summaries → Generate natural-language insights using LLMs (Hugging Face)

Analytics Dashboard → Visualize sentiment, predictions, and performance

📂 Project Structure
genai-financial-doc-analysis/
│
├── data/                  # Datasets (ignored in Git)
│   ├── raw/               # Original datasets (Hugging Face, Kaggle, yfinance)
│   └── processed/         # Cleaned / merged data
│
├── notebooks/             # Jupyter notebooks (week-by-week pipeline)
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_transformer_sentiment.ipynb
│   ├── 04_ml_prediction.ipynb
│   ├── 05_llm_summary_generation.ipynb
│   ├── 06_dashboard_analysis.ipynb
│   └── 07_final_demo.ipynb
│
├── scripts/               # Modular Python scripts
│   ├── data_utils.py
│   ├── sentiment_utils.py
│   ├── ml_utils.py
│   └── summary_utils.py
│
├── reports/               # Documentation
│   ├── proposal.md
│   ├── interim_report.md
│   └── final_report.md
│
├── results/               # Outputs (charts, logs, screenshots)
│
├── requirements.txt       # Core dependencies
├── dev-requirements.txt   # Jupyter + dev tools
├── README.md              # Project overview
└── .gitignore             # Ignore venv, data, cache, etc.


⚙️ Setup Instructions
1. Clone Repository
git clone https://github.com/your-username/genai-financial-doc-analysis.git
cd genai-financial-doc-analysis

2. Create Virtual Environment
python -m venv .venv


Activate it:

Windows (PowerShell):

.\.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

3. Install Dependencies

Core packages:

pip install -r requirements.txt


Development tools (Jupyter, ipykernel):

pip install -r dev-requirements.txt

4. Register Kernel (Jupyter / VS Code)
python -m ipykernel install --user --name=genai-financial-doc-analysis --display-name "Python (.venv) GenAI"

📊 Data Sources

Financial News Sentiment Dataset: Kaggle

Stock Market Data: Collected via yfinance

Processed News Dataset: Filtered subset from Hugging Face FNSPID

🚀 Roadmap

Week 1–2: Data collection & preprocessing

Week 3: Transformer sentiment analysis (FinBERT)

Week 4–5: Stock price prediction (Logistic Regression, XGBoost)

Week 6: LLM investor summary generation (Watsonx.ai)

Week 7: Analytics dashboard (Watson Studio)

Week 8: Final report & presentation

🛠️ Tech Stack

Language: Python 3.10+

Libraries: pandas, yfinance, scikit-learn, matplotlib, seaborn, transformers, torch

📜 License

This project is released under the MIT License.
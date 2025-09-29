📊 Generative AI for Financial Document Analysis

Course Project — Fall 2025

📌 Overview

This project combines transformers (FinBERT), machine learning, and generative AI (LLMs) on IBM Cloud to analyze financial news and market data.

Pipeline:

Financial News Sentiment (transformers)

Stock Price Prediction (machine learning)

Investor Summaries (LLMs with Watsonx.ai)

Analytics Dashboard (Watson Studio / Python)

📂 Project Structure
genai-financial-doc-analysis/
│
├── data/                  # Datasets
│   ├── raw/               # Original datasets (Kaggle, yfinance)
│   └── processed/         # Cleaned/merged data
│
├── notebooks/             # Jupyter notebooks (Week-by-week pipeline)
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

Core project packages:

pip install -r requirements.txt


Dev tools (Jupyter, ipykernel):

pip install -r dev-requirements.txt

4. Register Kernel (for Jupyter/VS Code)
python -m ipykernel install --user --name=genai-financial-doc-analysis --display-name "Python (.venv) GenAI"

📊 Data Sources

Financial News Sentiment Dataset (Kaggle):
https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news

Stock Market Data (Yahoo Finance):
Collected via yfinance
.

🚀 Roadmap

Week 1–2: Data collection & preprocessing

Week 3: Transformer sentiment analysis (FinBERT)

Week 4–5: ML prediction (logistic regression, XGBoost)

Week 6: LLM investor summary generation (Watsonx.ai)

Week 7: Analytics dashboard (Watson Studio)

Week 8: Final report & presentation

🛠️ Tech Stack

Python 3.10+

Libraries: pandas, yfinance, scikit-learn, matplotlib, seaborn, transformers, torch

IBM Cloud Services: Watsonx.ai, Watson Machine Learning, Watson Studio

📜 License

MIT License. See LICENSE
 for details.
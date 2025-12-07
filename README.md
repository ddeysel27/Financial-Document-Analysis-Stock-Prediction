Generative AI for Financial Document Analysis

Overview

This project examines how financial news sentiment and market data can be combined with AI models to improve stock movement prediction and investor insights. It includes:

Transformer-based sentiment analysis using FinBERT

Machine learning models for stock direction prediction

Generative AI models for producing investor-friendly summaries

Dashboards (Python and Power BI) for data visualization and performance reporting

Project Pipeline

Financial News Sentiment
Extract sentiment scores from market articles using FinBERT.

Stock Price Prediction
Train ML models using combined sentiment features and historical price data.

Investor Summary Generation
Use large language models (Hugging Face) to generate natural-language insights.

Analytics Dashboard
Visualize sentiment trends, predictions, and model performance.

Setup Instructions
1. Clone the Repository
git clone https://github.com/your-username/genai-financial-doc-analysis.git
cd genai-financial-doc-analysis

2. Create a Virtual Environment
python -m venv .venv


Activate the environment:

Windows (PowerShell):

.\.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

3. Install Dependencies

Core packages:

pip install -r requirements.txt


Development tools (Jupyter, ipykernel):

pip install -r dev-requirements.txt

4. Register the Kernel (optional)
python -m ipykernel install --user --name=genai-financial-doc-analysis --display-name "Python (.venv) GenAI"

Data Sources

Financial News Sentiment Dataset: Kaggle

Stock Market Data: Collected using yfinance

Processed News Dataset: Filtered subset from Hugging Face FNSPID

Roadmap

Week 1–2: Data collection and preprocessing

Week 3: Transformer-based sentiment analysis (FinBERT)

Week 4–5: Stock direction prediction (Logistic Regression, XGBoost)

Week 6: LLM-generated investor summaries

Week 7: Analytics dashboard development

Week 8: Final report

Tech Stack

Language: Python 3.10+

Libraries: pandas, yfinance, scikit-learn, matplotlib, seaborn, transformers, torch

License

This project is released under the MIT License.
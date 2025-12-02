import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="News Articles", layout="wide")
st.title("News Articles & Summaries")
st.write("Browse news articles aligned with your model's test period.")

# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------
# 1. Load the testing window (from model)
try:
    test_df = pd.read_csv("data/testing_predictions_clean.csv")
    test_df["Date"] = pd.to_datetime(test_df["Date"], errors="coerce")
except:
    st.error("Could not load testing_predictions_clean.csv")
    st.stop()

test_min = test_df["Date"].min().date()
test_max = test_df["Date"].max().date()

# 2. Load processed news
try:
    news_df = pd.read_csv(
        "C:/Users/User/Desktop/Data Science/Fall 2025/Generative AI/Project/genai-financial-doc-analysis/data/processed/news_with_sentiment.csv"
    )
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
except Exception as e:
    st.error(f"Could not load news dataset: {e}")
    st.stop()

# ------------------------------------------------------------
# VALIDATE COLUMNS
# ------------------------------------------------------------
required_cols = [
    "Date", "Article_title", "Stock_symbol", "Url", "Publisher",
    "Author", "Article", "Lsa_summary", "Luhn_summary",
    "Textrank_summary", "Lexrank_summary", "Sentiment_score"
]

missing = [c for c in required_cols if c not in news_df.columns]
if missing:
    st.error(f"Missing columns in news file: {missing}")
    st.stop()

# ------------------------------------------------------------
# FILTER NEWS TO TEST WINDOW ONLY
# ------------------------------------------------------------
news_df = news_df[
    (news_df["Date"].dt.date >= test_min) &
    (news_df["Date"].dt.date <= test_max)
].copy()

if news_df.empty:
    st.warning("⚠ No news articles fall inside the model's test window.")
    st.stop()

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("🔎 Filter News")

# Only show tickers that exist BOTH in news & model test data
valid_tickers = sorted(
    list(set(news_df["Stock_symbol"].unique()).intersection(set(test_df["Ticker"].unique())))
)

selected_ticker = st.sidebar.selectbox("Select Ticker", valid_tickers)

df_ticker = news_df[news_df["Stock_symbol"] == selected_ticker]

# Show only dates inside the test window
dates = sorted(df_ticker["Date"].dt.date.unique().tolist())
selected_date = st.sidebar.selectbox("Select Date", dates)

df_day = df_ticker[df_ticker["Date"].dt.date == selected_date]

if df_day.empty:
    st.info("No articles found for this date.")
    st.stop()

st.subheader(f"Articles for **{selected_ticker}** on **{selected_date}**")

# ------------------------------------------------------------
# SENTIMENT BADGE HELPER
# ------------------------------------------------------------
def sentiment_badge(score):
    score_fmt = f"{score:.2f}"

    if score > 0.05:
        color = "#4CAF50"
        label = "Positive"
    elif score < -0.05:
        color = "#F44336"
        label = "Negative"
    else:
        color = "#9E9E9E"
        label = "Neutral"

    return f"""
        <span style="
            background-color:{color};
            color:white;
            padding:3px 8px;
            border-radius:6px;
            font-size:0.75rem;
        ">{label}: {score_fmt}</span>
    """

# ------------------------------------------------------------
# DISPLAY ARTICLES
# ------------------------------------------------------------
for idx, row in df_day.iterrows():

    st.markdown("---")

    badge = sentiment_badge(row["Sentiment_score"])

    title_html = f"""
        <div style="display:flex; align-items:center; gap:10px;">
            <h3 style="margin:0;">📝 {row['Article_title']}</h3>
            {badge}
        </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)

    # Metadata
    meta = []
    if pd.notna(row["Publisher"]):
        meta.append(f"**Publisher:** {row['Publisher']}")
    if pd.notna(row["Author"]):
        meta.append(f"**Author:** {row['Author']}")
    if pd.notna(row["Url"]):
        meta.append(f"[🔗 Read Original]({row['Url']})")

    if meta:
        st.markdown(" • ".join(meta))

    # Full article text
    with st.expander("Full Article"):
        st.write(row["Article"])

    # Summary expanders
    st.markdown("#### Summaries")

    if pd.notna(row["Lsa_summary"]):
        with st.expander("- LSA Summary"):
            st.write(row["Lsa_summary"])

    if pd.notna(row["Luhn_summary"]):
        with st.expander("- Luhn Summary"):
            st.write(row["Luhn_summary"])

    if pd.notna(row["Textrank_summary"]):
        with st.expander("- TextRank Summary"):
            st.write(row["Textrank_summary"])

    if pd.notna(row["Lexrank_summary"]):
        with st.expander("- LexRank Summary"):
            st.write(row["Lexrank_summary"])

st.markdown("---")
st.success(f"✔ Showing articles from **{test_min} → {test_max}** (model test window)")

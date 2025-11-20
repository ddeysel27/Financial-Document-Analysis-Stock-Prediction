import streamlit as st
import pandas as pd
from datetime import datetime

# ================================================
#   PAGE CONFIG
# ================================================
st.set_page_config(page_title="News Articles", layout="wide")

st.title("📰 News Articles & Summaries")
st.write("Browse news articles linked to your stock prediction dataset.")

# ================================================
#   LOAD NEWS DATASET
# ================================================
try:
    news_df = pd.read_csv("C:/Users/User/Desktop/Data Science/Fall 2025/Generative AI/Project/genai-financial-doc-analysis/data/raw/FNSPID/news_filtered.csv")  # <--- CHANGE to your filename
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
except Exception as e:
    st.error(f"Could not load news dataset: {e}")
    st.stop()

# Ensure correct column names exist
required_cols = [
    "Date", "Article_title", "Stock_symbol", "Url", "Publisher",
    "Author", "Article", "Lsa_summary", "Luhn_summary",
    "Textrank_summary", "Lexrank_summary"
]

missing = [c for c in required_cols if c not in news_df.columns]
if missing:
    st.error(f"Missing columns in news file: {missing}")
    st.stop()

# ================================================
#   SIDEBAR FILTERS
# ================================================
st.sidebar.header("🔎 Filter News")

tickers = sorted(news_df["Stock_symbol"].dropna().unique().tolist())
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

# Filter by ticker first
df_ticker = news_df[news_df["Stock_symbol"] == selected_ticker]

dates = sorted(df_ticker["Date"].dt.date.unique().tolist())
selected_date = st.sidebar.selectbox("Select Date", dates)

# Filter by selected date
df_day = df_ticker[df_ticker["Date"].dt.date == selected_date]

if df_day.empty:
    st.info("No articles found for this date.")
    st.stop()

# ================================================
#   DISPLAY ARTICLES
# ================================================
st.subheader(f"🗞 Articles for **{selected_ticker}** on **{selected_date}**")

for idx, row in df_day.iterrows():
    st.markdown("---")

    st.markdown(f"### 📝 {row['Article_title']}")
    
    # Meta info
    meta = []
    if pd.notna(row["Publisher"]):
        meta.append(f"**Publisher:** {row['Publisher']}")
    if pd.notna(row["Author"]):
        meta.append(f"**Author:** {row['Author']}")
    if pd.notna(row["Url"]):
        meta.append(f"[Read Original]({row['Url']})")

    if meta:
        st.markdown(" • ".join(meta))

    # Full article content
    with st.expander("📄 Full Article"):
        st.write(row["Article"])

    # Summaries section
    st.markdown("#### 🔍 Summaries")

    if pd.notna(row["Lsa_summary"]):
        with st.expander("🧠 LSA Summary"):
            st.write(row["Lsa_summary"])

    if pd.notna(row["Luhn_summary"]):
        with st.expander("📚 Luhn Summary"):
            st.write(row["Luhn_summary"])

    if pd.notna(row["Textrank_summary"]):
        with st.expander("🕸 TextRank Summary"):
            st.write(row["Textrank_summary"])

    if pd.notna(row["Lexrank_summary"]):
        with st.expander("📈 LexRank Summary"):
            st.write(row["Lexrank_summary"])

st.markdown("---")
st.success("✔ Articles Loaded Successfully")

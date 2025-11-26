import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# =========================================================
#                   PAGE SETUP
# =========================================================
st.set_page_config(page_title="LLM Chatbot", layout="wide")
st.title("🤖 Chat With The Local LLaMA Model")
st.write("Ask financial, prediction, or sentiment-based questions. The AI uses your dataset for accurate analysis.")

# =========================================================
#           LOAD BOTH DATASETS (Prediction + Sentiment)
# =========================================================
try:
    pred_df = pd.read_csv("data/testing_predictions_clean.csv")
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])
except Exception as e:
    st.error(f"Could not load predictions dataset: {e}")
    st.stop()

try:
    sent_df = pd.read_csv("data/testing_data.csv")
    sent_df["Date"] = pd.to_datetime(sent_df["Date"])
except Exception as e:
    st.error(f"Could not load sentiment dataset: {e}")
    st.stop()

# ---- Merge on Date + Ticker ----
merged_df = pd.merge(
    pred_df,
    sent_df,
    how="left",
    on=["Date", "Ticker"]
)

# =========================================================
#        RETRIEVAL: FIND THE RIGHT ROWS FOR THE QUESTION
# =========================================================
def retrieve_context(user_msg):
    """
    Dynamically extracts rows from merged_df based on ticker, dates,
    and relative phrases like 'past 7 days'.
    Returns a small subset of relevant context in text form.
    """
    msg = user_msg.lower()
    tickers = merged_df["Ticker"].unique().tolist()

    # ---- Detect ticker ----
    found_ticker = None
    for t in tickers:
        if t.lower() in msg:
            found_ticker = t

    # ---- Detect date references ----
    extracted_date = None

    if "past 7 days" in msg:
        extracted_date = merged_df["Date"].max() - timedelta(days=7)
    elif "yesterday" in msg:
        extracted_date = merged_df["Date"].max() - timedelta(days=1)

    # ---- Manual YYYY-MM-DD detection ----
    for word in user_msg.split():
        try:
            extracted_date = datetime.strptime(word, "%Y-%m-%d")
        except:
            pass

    subset = merged_df.copy()

    if found_ticker:
        subset = subset[subset["Ticker"] == found_ticker]

    if extracted_date:
        subset = subset[subset["Date"] >= extracted_date]

    # ---- Fallback ----
    if subset.empty:
        subset = merged_df.tail(5)

    # ---- Limit to last 10 rows to avoid overwhelming prompt ----
    subset = subset.tail(10)

    return subset.to_string(index=False)


# =========================================================
#        OLLAMA CALL (STREAMING SAFE)
# =========================================================
def ask_ollama(prompt, model="llama3.2"):
    """
    Sends prompt to local Ollama LLaMA model.
    Handles streaming JSON responses safely.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            stream=True
        )

        full_reply = ""

        for line in response.iter_lines():
            if not line:
                continue

            try:
                json_obj = json.loads(line.decode("utf-8"))
            except Exception as e:
                return f"⚠️ JSON Parsing Error: {e}\nRaw: {line}"

            full_reply += json_obj.get("response", "")

            if json_obj.get("done"):
                break

        return full_reply.strip()

    except Exception as e:
        return f"⚠️ Error contacting Ollama: {e}"


# =========================================================
#                 CHAT HISTORY UI
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous messages
for speaker, text in st.session_state.history:
    st.chat_message(speaker).write(text)


# =========================================================
#                 CHAT INPUT (MAIN LOGIC)
# =========================================================
user_input = st.chat_input("Ask something about stocks, predictions, sentiment...")

if user_input:
    # Show user msg
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)

    # Retrieve dataset context
    context = retrieve_context(user_input)

    # Build prompt
    final_prompt = f"""
You are a financial analysis assistant with access to the user's structured stock dataset.
This dataset includes OHLC data, model predictions, prediction error, volatility,
and sentiment (avg_sentiment_score, article_count).

Use BOTH the dataset context below AND your own reasoning.

DATA CONTEXT:
{context}

USER QUESTION:
{user_input}

TASK:
Analyze market behavior, sentiment, and prediction accuracy.
Compare sentiment to price direction when relevant.
Explain your reasoning clearly and cite specific numbers from the DATA CONTEXT.
Provide a helpful and concise financial answer.
"""

    with st.spinner("Thinking..."):
        ai_response = ask_ollama(final_prompt)

    st.session_state.history.append(("assistant", ai_response))
    st.chat_message("assistant").write(ai_response)

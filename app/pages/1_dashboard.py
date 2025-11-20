import streamlit as st
import pandas as pd

st.title("📅 Single-Day Prediction Analysis")

# -------------------------------------------------------
# Load testing data
# -------------------------------------------------------
df = pd.read_csv("data/testing_predictions_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])

# -------------------------------------------------------
# Sidebar — Ticker selection
# -------------------------------------------------------
tickers = sorted(df["Ticker"].unique().tolist())
ticker = st.sidebar.selectbox("Select Ticker", tickers)

df_ticker = df[df["Ticker"] == ticker].copy()

# -------------------------------------------------------
# Sidebar — Date selection
# -------------------------------------------------------
available_dates = df_ticker["Date"].dt.date.unique().tolist()
selected_date = st.sidebar.selectbox("Select Date", available_dates)

# Filter for the selected date
row = df_ticker[df_ticker["Date"].dt.date == selected_date]

if row.empty:
    st.warning("No data found for this date.")
    st.stop()

actual = float(row["Close"].iloc[0])
pred = float(row["Prediction"].iloc[0])
error = abs(pred - actual)
error_pct = (error / actual) * 100

# -------------------------------------------------------
# Direction correctness logic (safe iloc-based method)
# -------------------------------------------------------

# Get positional index of the selected row
row_pos = row.index[0]                      # this is the actual index label
pos = df_ticker.index.get_loc(row_pos)      # convert label → position

# If it's the first row, no direction possible
if pos == 0:
    prev_close = None
    actual_dir = "N/A"
    pred_dir = "N/A"
    dir_correct = False
else:
    prev_close = float(df_ticker["Close"].iloc[pos - 1])
    actual_dir = "UP" if actual > prev_close else "DOWN"
    pred_dir   = "UP" if pred > prev_close else "DOWN"
    dir_correct = actual_dir == pred_dir

# -------------------------------------------------------
# Display Insights
# -------------------------------------------------------
st.subheader(f"📊 Insights for {ticker} on {selected_date}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Actual Close", f"${actual:,.2f}")
col2.metric("Predicted Close", f"${pred:,.2f}")
col3.metric("Error ($)", f"{error:,.2f}")
col4.metric("Error (%)", f"{error_pct:.2f}%")

# Directional prediction
st.markdown("### 🔍 Directional Accuracy")

if prev_close is not None:
    st.write(f"**Previous Close:** ${prev_close:,.2f}")
    st.write(f"📈 Actual movement: **{actual_dir}**")
    st.write(f"🤖 Predicted movement: **{pred_dir}**")

    if dir_correct:
        st.success("✅ Prediction correctly predicted the direction of movement.")
    else:
        st.error("❌ Prediction got the direction wrong.")
else:
    st.info("Directional accuracy unavailable for the first date.")

# -------------------------------------------------------
# Optional: Mini price chart around selected date
# -------------------------------------------------------
st.markdown("### 📉 Price Context (±3 Days)")

context = df_ticker.copy()
context["date_only"] = context["Date"].dt.date

# Find row index of selected date
pos_list = context.index[context["date_only"] == selected_date].tolist()

if len(pos_list) == 0:
    st.warning("No context available for this date.")
else:
    pos = pos_list[0]
    start = max(context.index.min(), pos - 3)
    end = min(context.index.max(), pos + 3)

    chart_data = (
        context.loc[start:end]
        .set_index("Date")[["Close", "Prediction"]]
    )

    st.line_chart(chart_data)

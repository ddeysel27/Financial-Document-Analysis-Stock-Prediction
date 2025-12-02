import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==============================================================
# 1) Load Predictions for Test Window (Directional Model)
# ==============================================================

@st.cache_data
def load_predictions(path: str = "data/testing_predictions_clean.csv") -> pd.DataFrame:
    """
    Load the final test predictions from CSV and ensure
    the Date column is parsed + sorted by Ticker and Date.
    """
    df_ = pd.read_csv(path)
    df_["Date"] = pd.to_datetime(df_["Date"])
    df_ = df_.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Ensure the core columns exist
    required_cols = {"Date", "Close", "Ticker", "Prediction", "Ensemble_Prob"}
    missing = required_cols - set(df_.columns)
    if missing:
        st.error(f"Missing required columns in CSV: {missing}")
        st.stop()

    # If Return not present, compute it once here (by ticker)
    if "Return" not in df_.columns:
        df_["Return"] = (
            df_.groupby("Ticker")["Close"].pct_change()
        )

    return df_


df = load_predictions()

# ==============================================================
# 2) Sidebar Controls
# ==============================================================

st.title("Single-Day Directional Prediction & Range Analysis")

tickers = sorted(df["Ticker"].unique().tolist())
ticker = st.sidebar.selectbox("Select Ticker", tickers)

# Subset to this ticker only
df_ticker = df[df["Ticker"] == ticker].copy().reset_index(drop=True)

available_dates = df_ticker["Date"].dt.date.unique().tolist()
selected_date = st.sidebar.selectbox("Select Date (Single-Day View)", available_dates)

# ==============================================================
# 3) Extract Single-Day Record
# ==============================================================

row_df = df_ticker[df_ticker["Date"].dt.date == selected_date]

if row_df.empty:
    st.warning("No data available for this date.")
    st.stop()

row = row_df.iloc[0]  # Series for convenience

actual_close = float(row["Close"])
pred_class = int(row["Prediction"])         # 1 = UP, 0 = DOWN
prob_up = float(row["Ensemble_Prob"])      # probability that price goes UP

# Use stored Return when possible; fallback if NaN
if not pd.isna(row.get("Return", np.nan)):
    actual_ret = float(row["Return"])
else:
    idx_label = row_df.index[0]
    pos = df_ticker.index.get_loc(idx_label)
    if pos == 0:
        actual_ret = np.nan
    else:
        prev_close_tmp = float(df_ticker["Close"].iloc[pos - 1])
        actual_ret = (actual_close - prev_close_tmp) / prev_close_tmp

# --------------------------------------------------------------
# Compute actual vs predicted direction for this day
# --------------------------------------------------------------
# Position in full ticker slice
idx_label = row_df.index[0]
pos = df_ticker.index.get_loc(idx_label)

if pos == 0:
    prev_close = None
    actual_dir = "N/A"
    dir_correct = False
else:
    prev_close = float(df_ticker["Close"].iloc[pos - 1])
    actual_dir = "UP" if actual_close > prev_close else "DOWN"

pred_dir = "UP" if pred_class == 1 else "DOWN"
dir_correct = (prev_close is not None) and (actual_dir == pred_dir)

# Calibration error = |probability - true_label|
if prev_close is None:
    calib_error = np.nan
else:
    true_label = 1 if actual_dir == "UP" else 0
    calib_error = abs(prob_up - true_label)
st.markdown("---")

# ==============================================================
# 4) Single-Day Overview Metrics
# ==============================================================

st.subheader(f"Single-Day Insights for {ticker} on {selected_date}")

c1, c2, c3, c4 = st.columns(4)

# Actual close
c1.metric(
    "Actual Close",
    f"${actual_close:,.2f}",
    help="Observed market closing price on the selected day."
)

# Previous close
if prev_close is None:
    prev_text = "N/A"
else:
    prev_text = f"${prev_close:,.2f}"

c2.metric(
    "Previous Close",
    prev_text,
    help="Closing price on the day before the selected date."
)

# Actual daily return (%)
if np.isnan(actual_ret):
    ret_text = "N/A"
else:
    ret_text = f"{actual_ret * 100:,.2f}%"

c3.metric(
    "Actual Daily Return",
    ret_text,
    help="Percentage change from the previous closing price to today's close."
)

# Model probability of an UP move
c4.metric(
    "Model P(UP)",
    f"{prob_up * 100:,.1f}%",
    help="Model's estimated probability that the next day's price will move UP."
)


st.markdown("---")
# ==============================================================
# 5) Directional Prediction Explanation
# ==============================================================

st.markdown("### Directional Prediction")
st.caption(
    "The model predicts whether the next closing price will move UP or DOWN "
    "relative to the previous day and provides a probability for that UP move."
)

if prev_close is None:
    st.info("Directional accuracy cannot be computed for the first available date.")
else:
    st.write(f"📈 **Actual Direction:** {actual_dir}")
    st.write(
        f"🤖 **Predicted Direction:** {pred_dir} "
        f"(P(UP) = {prob_up * 100:,.1f}%)"
    )

    if dir_correct:
        st.success("✅ The model predicted the correct direction for this day.")
    else:
        st.error("❌ The model predicted the wrong direction for this day.")

    if not np.isnan(calib_error):
        st.caption(
            f"Calibration error for this day is **{calib_error:.3f}** "
            f"(0 = perfect, 0.5 ≈ random guess in a binary setting)."
        )

st.markdown("---")
# ==============================================================
# 6) Price Context (±3 Days)
# ==============================================================
import plotly.graph_objects as go
import pandas as pd

# -----------------------------
# 1. CLEAN DATE COLUMN
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).copy()

# -----------------------------
# 2. FILTER BY TICKER FIRST
# -----------------------------
df_tkr = df[df["Ticker"] == ticker].copy()

# Remove bad rows
df_tkr = df_tkr.dropna(subset=["Open", "High", "Low", "Close"])

# Ensure timezones removed
if df_tkr["Date"].dt.tz is not None:
    df_tkr["Date"] = df_tkr["Date"].dt.tz_localize(None)

# Sort by date
df_tkr = df_tkr.sort_values("Date").reset_index(drop=True)

# -----------------------------
# 3. FIX selected_date
# -----------------------------
selected_date = pd.to_datetime(selected_date)

# -----------------------------
# 4. WINDOW SELECTION
# -----------------------------
window = 3
start_date = selected_date - pd.Timedelta(days=window)
end_date   = selected_date + pd.Timedelta(days=window)

price_window = df_tkr[
    (df_tkr["Date"] >= start_date) &
    (df_tkr["Date"] <= end_date)
].copy()

if price_window.empty:
    st.warning("No price data in this window.")
else:

    # -----------------------------
    # 5. CANDLESTICK FIGURE
    # -----------------------------
    fig_candle = go.Figure()

    fig_candle.add_trace(go.Candlestick(
        x=price_window["Date"],
        open=price_window["Open"],
        high=price_window["High"],
        low=price_window["Low"],
        close=price_window["Close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        name="Candles"
    ))

    # -----------------------------
    # 6. TRUE vertical line over candle range
    # -----------------------------
    y_min = price_window["Low"].min()
    y_max = price_window["High"].max()

    fig_candle.add_shape(
        type="line",
        x0=selected_date,
        x1=selected_date,
        y0=y_min,
        y1=y_max,
        line=dict(color="yellow", width=2, dash="dash"),
        xref="x",
        yref="y"
    )

    fig_candle.add_annotation(
        x=selected_date,
        y=y_max,
        text="Selected",
        showarrow=False,
        font=dict(color="yellow"),
        yshift=8
    )

    # -----------------------------
    # 7. LAYOUT
    # -----------------------------
    fig_candle.update_layout(
        title=f"Candlestick Chart (±{window} Days Around {selected_date.date()})",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=350,
        margin=dict(l=10, r=10, t=50, b=20),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig_candle, use_container_width=True)

st.markdown("---")
# ==============================================================
# 7) Volatility Context (7-Day Rolling Std of Returns)
# ==============================================================

import plotly.express as px
import pandas as pd

st.markdown("### 💵 Volatility in Dollar Terms (Last 7 Days)")
st.caption(
    "Shows how much the stock moved each day in dollar terms. "
    "Useful for understanding market turbulence around the selected date."
)

# Ensure correct datetime
df_daily = df_ticker.copy()
df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce")
df_daily = df_daily.sort_values("Date").reset_index(drop=True)

# Compute daily dollar volatility
df_daily["DollarVol"] = (df_daily["Close"] - df_daily["Close"].shift(1)).abs()

# Convert selected_date properly
selected_date = pd.to_datetime(selected_date).date()

# Filter last 7 days BEFORE selected date
window_data = df_daily[df_daily["Date"].dt.date < selected_date].tail(7).copy()

if window_data.empty:
    st.info("Not enough data to compute 7-day volatility window.")
else:
    # Metric for the most recent day in window
    recent_vol = window_data["DollarVol"].iloc[-1]

    st.metric(
        "Most Recent Daily Dollar Volatility",
        f"${recent_vol:,.2f}",
        help="Absolute difference between today's close and previous day's close."
    )

    # Bar chart for the last 7 days
    fig_vol = px.bar(
        window_data,
        x="Date",
        y="DollarVol",
        title="Daily Dollar Volatility (|Close - Previous Close|)",
        labels={"DollarVol": "Dollar Volatility ($)", "Date": "Date"},
        template="plotly_dark",
    )

    fig_vol.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="Volatility ($)",
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(fig_vol, use_container_width=True)


# ==============================================================
# 8) Qualitative Confidence from Probability
# ==============================================================

st.markdown("### Qualitative Confidence")
st.caption(
    "Translates the UP probability into a simple High / Medium / Low "
    "confidence tag for easier interpretation."
)

if prob_up >= 0.7 or prob_up <= 0.3:
    conf_label = "🟢 High Confidence"
elif 0.45 <= prob_up <= 0.55:
    conf_label = "🟠 Low Confidence"
else:
    conf_label = "🟡 Medium Confidence"

st.metric(conf_label, f"P(UP) = {prob_up * 100:,.1f}%")

st.markdown("---")

# ==============================================================
# 9) Recent Directional Performance (Last 7 Records)
# ==============================================================

st.markdown("### Recent Directional Performance (Last 7 Records)")
st.caption(
    "Shows the last few predictions for this ticker, including actual market moves, "
    "predicted direction, and whether the model was correct."
)

# -------------------------------------------------------------------
# Step 1: Prepare dataframe sorted by date
# -------------------------------------------------------------------
perf_df = df_ticker.copy().sort_values("Date").reset_index(drop=True)

# Previous close (needed for returns)
perf_df["PrevClose"] = perf_df["Close"].shift(1)
perf_df = perf_df.dropna(subset=["PrevClose"]).copy()

# -------------------------------------------------------------------
# Step 2: Compute actual + predicted directions
# -------------------------------------------------------------------
perf_df["ActualDir"] = np.where(perf_df["Close"] > perf_df["PrevClose"], 1, 0)
perf_df["PredDir"] = perf_df["Prediction"].astype(int)
perf_df["Correct"] = perf_df["ActualDir"] == perf_df["PredDir"]

# Labels for tooltip readability
perf_df["ActualDirectionLabel"] = perf_df["ActualDir"].map({1: "Up", 0: "Down"})
perf_df["PredDirectionLabel"] = perf_df["PredDir"].map({1: "Up", 0: "Down"})
perf_df["CorrectLabel"] = perf_df["Correct"].map({True: "Correct", False: "Incorrect"})

# -------------------------------------------------------------------
# Step 3: Compute returns + rounded values
# -------------------------------------------------------------------
# Ensure Return column exists
if "Return" not in perf_df.columns or perf_df["Return"].isna().all():
    perf_df["Return"] = (
        (perf_df["Close"] - perf_df["PrevClose"]) / perf_df["PrevClose"]
    )

perf_df["DailyReturnPct"] = perf_df["Return"] * 100.0
perf_df["DailyReturnPctRounded"] = perf_df["DailyReturnPct"].round(2)

# Dollar movement
perf_df["ActualMovement"] = perf_df["Close"] - perf_df["PrevClose"]
perf_df["ActualMovementRounded"] = perf_df["ActualMovement"].round(2)

# -------------------------------------------------------------------
# Step 4: Slice the last 7 rows up to selected date
# -------------------------------------------------------------------
perf_df = perf_df[perf_df["Date"] <= row["Date"]].tail(7)

if perf_df.empty:
    st.info("Not enough observations to evaluate recent performance.")
else:

    # ---------------------------------------------------------------
    # Step 5: Display recent accuracy metric
    # ---------------------------------------------------------------
    recent_acc = perf_df["Correct"].mean() * 100.0
    st.metric("Last-7 Directional Accuracy", f"{recent_acc:.1f}%")

    # ---------------------------------------------------------------
    # Step 6: Plot scatter chart with detailed tooltips
    # ---------------------------------------------------------------
    scatter = (
        alt.Chart(perf_df)
        .mark_circle(size=90)
        .encode(
            x="Date:T",
            y=alt.Y("DailyReturnPct:Q", title="Daily Return (%)"),
            color=alt.Color(
                "Correct:N",
                scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#d62728"]),
                legend=alt.Legend(title="Prediction Correct?")
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("DailyReturnPctRounded:Q", title="Daily Return (%)"),
                alt.Tooltip("ActualMovementRounded:Q", title="Price Change ($)"),
                alt.Tooltip("ActualDirectionLabel:N", title="Actual Direction"),
                alt.Tooltip("PredDirectionLabel:N", title="Predicted Direction"),
                alt.Tooltip("CorrectLabel:N", title="Outcome")
            ]
        )
        .properties(height=250)
    )

    st.altair_chart(scatter, use_container_width=True)


st.markdown("---")

# ==============================================================
# 10) Multi-Day Accuracy & Simple Strategy Evaluation
# ==============================================================

st.markdown("## Multi-Day Accuracy Analysis")
st.caption(
    "Choose a date range to evaluate the model's directional accuracy and compare a "
    """Strategy Cumulative Return
    
    How much you would have earned by only entering trades on days where 
    the model predicted the stock would go UP. If the model predicted DOWN,
    you sit out that day (0% return).

    Buy-and-Hold Cumulative Return
    How much you would earn by simply buying the stock at the start of the
    date range and holding it continuously.

    Why Strategy Can Beat Buy-and-Hold
    Because the model lets you avoid down days by sitting in cash."""
)

r1, r2 = st.columns(2)
min_d = df_ticker["Date"].dt.date.min()
max_d = df_ticker["Date"].dt.date.max()

range_start = r1.date_input(
    "Start Date",
    value=min_d,
    min_value=min_d,
    max_value=max_d,
)

range_end = r2.date_input(
    "End Date",
    value=max_d,
    min_value=min_d,
    max_value=max_d,
)

if range_start > range_end:
    st.error("Start Date must be earlier than or equal to End Date.")
else:
    mask = (
        (df_ticker["Date"].dt.date >= range_start) &
        (df_ticker["Date"].dt.date <= range_end)
    )
    range_df = df_ticker[mask].sort_values("Date").reset_index(drop=True)

    if len(range_df) < 3:
        st.warning("Not enough data in the selected range to compute metrics.")
    else:
        range_df["PrevClose"] = range_df["Close"].shift(1)
        range_df = range_df.dropna(subset=["PrevClose"]).copy()

        # Directions
        range_df["ActualDir"] = np.where(range_df["Close"] > range_df["PrevClose"], 1, 0)
        range_df["PredDir"] = range_df["Prediction"].astype(int)
        range_df["Correct"] = range_df["ActualDir"] == range_df["PredDir"]

        # Returns inside the range
        if "Return" not in range_df.columns or range_df["Return"].isna().all():
            range_df["Return"] = (
                (range_df["Close"] - range_df["PrevClose"]) / range_df["PrevClose"]
            )

        # Confusion matrix
        tp = ((range_df["ActualDir"] == 1) & (range_df["PredDir"] == 1)).sum()
        tn = ((range_df["ActualDir"] == 0) & (range_df["PredDir"] == 0)).sum()
        fp = ((range_df["ActualDir"] == 0) & (range_df["PredDir"] == 1)).sum()
        fn = ((range_df["ActualDir"] == 1) & (range_df["PredDir"] == 0)).sum()

        dir_acc = range_df["Correct"].mean() * 100.0

        # Simple strategy: only hold when model predicts UP
        range_df["StrategyReturn"] = np.where(range_df["PredDir"] == 1,
                                              range_df["Return"], 0.0)
        range_df["BuyHoldReturn"] = range_df["Return"]

        range_df["StrategyCurve"] = (1 + range_df["StrategyReturn"]).cumprod() - 1
        range_df["BuyHoldCurve"] = (1 + range_df["BuyHoldReturn"]).cumprod() - 1

        st.markdown("### Range-Level Metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Directional Accuracy", f"{dir_acc:.2f}%")
        m2.metric("True Positives (UP & correct)", str(tp))
        m3.metric("False Positives (Predicted UP, actually DOWN)", str(fp))
        m4.metric("False Negatives (Predicted DOWN, actually UP)", str(fn))

        st.caption(f"Range evaluated from **{range_start}** to **{range_end}**.")

        #----------------------------------------------------------
        # Cumulative Return Metrics (%)
        #----------------------------------------------------------
        final_strategy_ret = float(range_df["StrategyCurve"].iloc[-1])
        final_buyhold_ret = float(range_df["BuyHoldCurve"].iloc[-1])

        m5, m6 = st.columns(2)
        m5.metric("📈 Strategy Cumulative Return",
                f"{final_strategy_ret * 100:,.2f}%")

        m6.metric("💼 Buy-and-Hold Cumulative Return",
                f"{final_buyhold_ret * 100:,.2f}%")


        # Cumulative return comparison
        st.markdown("### 💹 Cumulative Return: Strategy vs Buy-and-Hold")

        curve_chart = alt.Chart(range_df).transform_fold(
            ["StrategyCurve", "BuyHoldCurve"],
            as_=["Series", "Value"]
        ).mark_line().encode(
            x="Date:T",
            y=alt.Y("Value:Q", title="Cumulative Return"),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=["StrategyCurve", "BuyHoldCurve"],
                    range=["#ff7f0e", "#1f77b4"]
                ),
                legend=alt.Legend(title="Series")
            ),
            tooltip=["Date:T", "Series:N", "Value:Q"]
        ).properties(height=260)

        st.altair_chart(curve_chart, use_container_width=True)

st.success("Insights and directional analysis loaded successfully.")

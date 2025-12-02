import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --------------------------------------------------
# Load test window predictions
# --------------------------------------------------
df = pd.read_csv("data/testing_predictions_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

st.title("🧪 Market Reaction Simulator")
st.write(
    "Explore how shocks to news sentiment, volatility, and price might affect the stock. "
    "This simulator uses your **directional ensemble probability** to build a "
    "model-implied expected price, then runs Monte-Carlo simulations around it."
)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
tickers = sorted(df["Ticker"].unique().tolist())
ticker = st.sidebar.selectbox("Select Ticker", tickers)

df_ticker = df[df["Ticker"] == ticker].copy()

available_dates = df_ticker["Date"].dt.date.unique().tolist()
base_date = st.sidebar.selectbox("Select Base Date", available_dates)

row = df_ticker[df_ticker["Date"].dt.date == base_date]
if row.empty:
    st.error("No data available for this date.")
    st.stop()

row = row.iloc[0]

actual_price = float(row["Close"])          # true close on that date
p_up = float(row["Ensemble_Prob"])          # model probability price will go UP

# --------------------------------------------------
# Historical volatility (daily returns)
# --------------------------------------------------
df_ticker = df_ticker.sort_values("Date").reset_index(drop=True)
df_ticker["Ret"] = df_ticker["Close"].pct_change()
hist_vol = df_ticker["Ret"].std()

# Fallback if volatility cannot be computed
if np.isnan(hist_vol) or hist_vol == 0:
    hist_vol = 0.02  # 2% daily vol as a safe default

# --------------------------------------------------
# Model-implied baseline expected price
# --------------------------------------------------
# Intuition:
#   If P(UP) = 0.5  → expected return = 0
#   If P(UP) > 0.5 → small positive drift
#   If P(UP) < 0.5 → small negative drift
expected_return = (p_up - 0.5) * 2 * hist_vol
baseline_price = actual_price * (1 + expected_return)

# --------------------------------------------------
# Scenario controls
# --------------------------------------------------
st.subheader(f"Scenario Controls for {ticker} on {base_date}")

col1, col2 = st.columns(2)

sentiment_shock = col1.slider(
    "News Sentiment Shock",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="−1 = very negative news, +1 = very positive news."
)

price_shock_pct = col2.slider(
    "Price Shock (%)",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Immediate price jump/drop from earnings, guidance, macro surprise, etc."
)

vol_shock_pct = col1.slider(
    "Volatility Shock (%)",
    min_value=-50,
    max_value=200,
    value=0,
    step=5,
    help="Changes overall volatility relative to recent history."
)

scenario_type = col2.selectbox(
    "Scenario Type",
    [
        "Custom Scenario",
        "Very Positive Earnings Surprise",
        "Mild Positive News",
        "Neutral Macro Day",
        "Mild Negative Macro Shock",
        "Severe Negative News",
    ],
)

scenario_info = {
    "Custom Scenario": "Fully user-defined scenario.",
    "Very Positive Earnings Surprise": "Strong beat and bullish guidance.",
    "Mild Positive News": "Favorable analyst notes or sector tailwinds.",
    "Neutral Macro Day": "No major catalysts; normal market noise.",
    "Mild Negative Macro Shock": "Slightly worse-than-expected macro data.",
    "Severe Negative News": "Major earnings miss, regulatory hit, or systemic risk.",
}

st.info(f"Scenario Selected: **{scenario_type}** — {scenario_info[scenario_type]}")

# --------------------------------------------------
# How the scenario modifies the price
# --------------------------------------------------
st.markdown("### How This Simulator Adjusts the Price")
st.caption(
    "We start from the model-implied expected price (using P(UP) and historical volatility), "
    "then adjust it with sentiment, price, and volatility shocks. "
    "This is for intuition only — not trading advice."
)

# Small weight: full ±1 sentiment shock ≈ ±3% extra return
sentiment_weight = 0.03
sentiment_return_adj = sentiment_weight * sentiment_shock

# Price shock as direct percentage
price_return_adj = price_shock_pct / 100.0

# Total scenario return vs *actual* close
scenario_return = expected_return + sentiment_return_adj + price_return_adj
scenario_price = actual_price * (1 + scenario_return)

# Volatility under scenario
vol_multiplier = 1 + vol_shock_pct / 100.0
scenario_vol = max(0.0001, hist_vol * vol_multiplier)

# --------------------------------------------------
# Monte-Carlo simulation around scenario price
# --------------------------------------------------
N = 1000
simulated_prices = np.random.normal(
    loc=scenario_price,
    scale=scenario_price * scenario_vol,
    size=N,
)
simulated_prices = np.clip(simulated_prices, 0.01, None)

mean_price = simulated_prices.mean()
p5 = np.percentile(simulated_prices, 5)
p95 = np.percentile(simulated_prices, 95)
prob_above_actual = (simulated_prices > actual_price).mean() * 100

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------
st.markdown("### Baseline vs Scenario Summary")

m1, m2, m3 = st.columns(3)
m1.metric("Actual Close", f"${actual_price:,.2f}")
m2.metric("Model-Implied Baseline Price", f"${baseline_price:,.2f}")
m3.metric("Scenario Price", f"${scenario_price:,.2f}")

r1, r2, r3 = st.columns(3)
r1.metric("Scenario Return vs Actual", f"{scenario_return * 100:,.2f}%")
r2.metric("Simulated Mean Price", f"${mean_price:,.2f}")
r3.metric("Pr(Price > Actual)", f"{prob_above_actual:,.1f}%")

st.caption(
    f"Baseline drift is driven by P(UP) = {p_up:.2f} and historical volatility "
    f"({hist_vol*100:.2f}% daily). Scenario adds sentiment, price and volatility shocks."
)

# --------------------------------------------------
# Simulated distribution chart
# --------------------------------------------------
st.markdown("### 📊 Simulated Price Distribution")
st.caption(
    "Histogram of simulated closing prices under the chosen scenario. "
    "Blue line = model-implied baseline price, orange line = scenario price."
)

sim_df = pd.DataFrame({"SimulatedPrice": simulated_prices})

hist = alt.Chart(sim_df).mark_bar(opacity=0.7).encode(
    x=alt.X("SimulatedPrice:Q", bin=alt.Bin(maxbins=40), title="Simulated Price"),
    y=alt.Y("count():Q", title="Frequency"),
)

baseline_rule = alt.Chart(pd.DataFrame({"value": [baseline_price]})).mark_rule(
    color="#1f77b4", strokeWidth=2
).encode(x="value:Q")

scenario_rule = alt.Chart(pd.DataFrame({"value": [scenario_price]})).mark_rule(
    color="#ff7f0e", strokeWidth=2, strokeDash=[5, 3]
).encode(x="value:Q")

st.altair_chart(hist + baseline_rule + scenario_rule, use_container_width=True)

# --------------------------------------------------
# Narrative explanation
# --------------------------------------------------
st.markdown("### 🧾 Scenario Narrative")

if scenario_return > 0.03:
    move_desc = "a **strong bullish** move"
elif scenario_return > 0.01:
    move_desc = "a **moderately bullish** reaction"
elif scenario_return > -0.01:
    move_desc = "a **neutral** reaction"
elif scenario_return > -0.03:
    move_desc = "a **mildly bearish** pullback"
else:
    move_desc = "a **strong bearish** move"

st.write(
    f"- Actual close on **{base_date}**: **${actual_price:,.2f}**.\n"
    f"- Model-implied baseline price (using P(UP) and volatility): "
    f"**${baseline_price:,.2f}**.\n"
    f"- Under your scenario, the expected price is **${scenario_price:,.2f}**, "
    f"which implies {scenario_return*100:,.2f}% move vs the actual close.\n"
    f"- Simulated outcomes mostly fall between **${p5:,.2f}** and **${p95:,.2f}**, "
    f"with an average of **${mean_price:,.2f}**.\n"
    f"- The probability of closing **above** the actual price is "
    f"**{prob_above_actual:.1f}%**.\n"
    f"- Overall this scenario corresponds to {move_desc} for **{ticker}**."
)

st.info(
    "This simulator is purely educational and for exploration — it is not financial advice "
    "or a recommendation to trade."
)

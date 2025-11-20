# start 2020-09-12import pandas as pd
import pandas as pd
# ----------------------------------------
# LOAD FULL MERGED DATA
# ----------------------------------------
df = pd.read_csv("data/processed/stocks_news_merged.csv")

# Ensure date column is datetime
df["Date"] = pd.to_datetime(df["Date"])


# Sort by date in ascending order
df = df.sort_values("Date").reset_index(drop=True)

# ----------------------------------------
# DEFINE TEST WINDOW (LAST 60 DAYS)
# ----------------------------------------
N_DAYS = 60

# Handle if dataset has fewer than 60 rows
if len(df) < N_DAYS:
    raise ValueError(f"Dataset has only {len(df)} rows. Need at least 60.")

train_df = df.iloc[:-N_DAYS]
test_df  = df.iloc[-N_DAYS:]

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

print("\n📌 Test Window:")
print(f"Start: {test_df['Date'].iloc[0].date()}")
print(f"End  : {test_df['Date'].iloc[-1].date()}")

# Save splits
train_df.to_csv("data/training_data.csv", index=False)
test_df.to_csv("data/testing_data.csv", index=False)

# ----------------------------------------
# BUILD testing_predictions_clean.csv
# ----------------------------------------
# Use next-day actual close as prediction placeholder
test_df["Prediction"] = test_df["Close"].shift(-1)

# Drop last row (no next-day close)
testing_pred_clean = test_df[["Date", "Close", "Ticker", "Prediction"]].dropna()

# Save clean version
testing_pred_clean.to_csv("data/testing_predictions_clean.csv", index=False)

print("\nSaved: data/testing_predictions_clean.csv")
print(testing_pred_clean.head())


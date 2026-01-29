import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

def fetch_data():
    df = yf.download(
        "BBCA.JK",
        period="90d",      # ⛔ yfinance membatasi data intraday
        interval="1h"
    )
    df.reset_index(inplace=True)
    df.to_csv("data/raw.csv", index=False)
    print("✅ Data ingestion selesai")

if __name__ == "__main__":
    fetch_data()
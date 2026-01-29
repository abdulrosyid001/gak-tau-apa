import pandas as pd

def preprocess():
    df = pd.read_csv("data/raw.csv")

    df = df[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["lag_1"] = df["Close"].shift(1)
    df.dropna(inplace=True)

    df.to_csv("data/processed.csv", index=False)
    print("✅ Preprocessing selesai")

if __name__ == "__main__":
    preprocess()
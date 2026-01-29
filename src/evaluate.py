import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os

def evaluate():
    df = pd.read_csv("data/processed.csv")

    X = df[["lag_1"]]
    y = df["Close"]

    split = int(len(df) * 0.8)
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    metrics_row = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mae": mae
    }])

    if os.path.exists("data/metrics.csv"):
        metrics_row.to_csv("data/metrics.csv", mode="a", header=False, index=False)
    else:
        metrics_row.to_csv("data/metrics.csv", index=False)

    print(f"📈 MAE: {mae:.4f}")

if __name__ == "__main__":
    evaluate()
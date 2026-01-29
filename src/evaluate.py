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

    today = datetime.now().strftime("%Y-%m-%d")

    new_row = pd.DataFrame([{
        "date": today,
        "mae": mae
    }])

    file_path = "data/metrics.csv"

    # =============================
    # HANDLE METRICS FILE SAFELY
    # =============================
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)

        # 🔧 FIX jika file lama tidak punya header
        if "date" not in existing.columns:
            existing.columns = ["date", "mae"]

        # ❌ Cegah duplikasi tanggal
        if today in existing["date"].astype(str).values:
            print(f"⚠️ Metrics for {today} already exists. Skipped.")
            return

        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(file_path, index=False)

    else:
        new_row.to_csv(file_path, index=False)

    print(f"📈 MAE ({today}): {mae:.4f}")

if __name__ == "__main__":
    evaluate()
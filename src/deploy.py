import pandas as pd
import shutil
import os

def deploy():
    if not os.path.exists("data/metrics.csv"):
        print("❌ Tidak ada metrik")
        return

    df = pd.read_csv("data/metrics.csv")

    if len(df) < 2:
        print("ℹ️ Model pertama, langsung deploy")
        shutil.copy("models/model.pkl", "models/production_model.pkl")
        return

    latest = df.iloc[-1]["mae"]
    previous = df.iloc[-2]["mae"]

    if latest < previous:
        shutil.copy("models/model.pkl", "models/production_model.pkl")
        print("🚀 Model baru DEPLOYED")
    else:
        print("⛔ Model lebih buruk, tidak deploy")

if __name__ == "__main__":
    deploy()
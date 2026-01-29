import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

os.makedirs("models", exist_ok=True)

def train():
    df = pd.read_csv("data/processed.csv")

    X = df[["lag_1"]]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model training selesai")

if __name__ == "__main__":
    train()
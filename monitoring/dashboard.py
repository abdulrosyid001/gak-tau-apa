import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="BBCA Forecast Monitoring", layout="wide")

st.title("📈 BBCA Stock Forecast – Model Monitoring Dashboard")

if not os.path.exists("data/metrics.csv"):
    st.warning("Belum ada data metrik.")
    st.stop()

df = pd.read_csv("data/metrics.csv")
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# METRIC SUMMARY
# =========================
latest_mae = df.iloc[-1]["mae"]

st.metric(
    label="Latest MAE",
    value=f"{latest_mae:.4f}"
)

# =========================
# PERFORMANCE OVER TIME
# =========================
st.subheader("Model Performance Over Time")

st.line_chart(
    df.set_index("Date")[["mae"]]
)

# =========================
# RAW METRICS TABLE
# =========================
with st.expander("Lihat raw metrics"):
    st.dataframe(df)
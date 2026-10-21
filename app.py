"""
Optional Streamlit dashboard for the Hadoop fraud-detection project.

Usage (on your own computer):
    1. Unzip fraud_results.zip (downloaded from the Colab notebook) into a folder named  outputs/  next to this file.
    2. pip install streamlit pandas matplotlib
    3. streamlit run app.py
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

OUT = os.environ.get("FRAUD_OUTPUTS", "outputs")

st.set_page_config(page_title="Fraud Detection - Hadoop", layout="wide")
st.title("Distributed Credit Card Fraud Detection")
st.caption("Hadoop MapReduce statistics + machine-learning detector. Predictions are experimental, "
           "not real financial decisions.")

if not os.path.exists(f"{OUT}/summary.json"):
    st.error(f"No results found in '{OUT}/'. Run the Colab notebook first and unzip fraud_results.zip there.")
    st.stop()

summary = json.load(open(f"{OUT}/summary.json"))
by_class = pd.read_csv(f"{OUT}/hadoop_by_class.csv")
by_amount = pd.read_csv(f"{OUT}/hadoop_by_amount.csv")
by_hour = pd.read_csv(f"{OUT}/hadoop_by_hour.csv")
metrics = pd.read_csv(f"{OUT}/model_metrics.csv")
top = pd.read_csv(f"{OUT}/top_suspicious.csv")

# --- headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", f"{summary['total_transactions']:,}")
c2.metric("Fraud cases", f"{summary['fraud_transactions']:,}")
c3.metric("Fraud rate", f"{summary['fraud_rate_percent']:.3f} %")
c4.metric("Hadoop = pandas?", "Yes" if summary["hadoop_matches_pandas"] else "NO")

t1, t2, t3 = st.tabs(["Hadoop statistics", "Model evaluation", "Suspicious transactions"])

with t1:
    a, b = st.columns(2)
    with a:
        st.subheader("Fraud vs genuine (Hadoop job 'class')")
        st.dataframe(by_class[["label", "count", "sum", "mean_amount", "min", "max"]], use_container_width=True)
        st.subheader("Fraud rate by amount range")
        fig, ax = plt.subplots()
        ax.bar(by_amount["bucket"].str[3:], by_amount["fraud_rate_%"], color="#f58518")
        ax.set_ylabel("fraud rate (%)"); ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
    with b:
        st.subheader("Fraud cases by hour since start")
        fig, ax = plt.subplots()
        ax.bar(by_hour["bucket"].astype(int), by_hour["fraud"], color="#e45756")
        ax.set_xlabel("hour (mod 24)"); ax.set_ylabel("fraud cases")
        st.pyplot(fig)
        r = summary["rule_detector"]
        st.subheader("Rule-based detector (run inside MapReduce)")
        st.write(f"Rule: V14 < {r['threshold_V14']} OR V12 < {r['threshold_V12']}")
        st.write(f"TP={r['TP']}  FP={r['FP']}  FN={r['FN']}  TN={r['TN']}")
        st.write(f"Precision {r['precision']:.3f} | Recall {r['recall']:.3f} | F1 {r['f1']:.3f}")
    st.write(f"Hadoop 'class' job: {summary['hadoop_class_job_seconds']} s  |  pandas: {summary['pandas_seconds']} s")

with t2:
    st.subheader("Model metrics on the held-out test set")
    st.dataframe(metrics.round(4), use_container_width=True)
    if os.path.exists(f"{OUT}/dashboard.png"):
        st.image(f"{OUT}/dashboard.png", caption="Full dashboard exported from the notebook")

with t3:
    st.subheader("Top 20 most suspicious test transactions (Random Forest)")
    st.dataframe(top.round(4), use_container_width=True)

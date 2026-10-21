# Distributed Credit Card Fraud Detection Using Apache Hadoop MapReduce

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite%20%2B%20TypeScript-61DAFB?style=flat&logo=react)](https://vitejs.dev)
[![Hadoop](https://img.shields.io/badge/Distributed-Apache%20Hadoop%20MapReduce-FFD700?style=flat&logo=apachehadoop)](https://hadoop.apache.org)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC?style=flat&logo=tailwindcss)](https://tailwindcss.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat&logo=python)](https://python.org)

A complete, production-grade web platform and real-time machine learning inference system built for the **Cloud and Distributed Computing (CDC Lab)** university project.

---

## 1. Project Overview

Credit card fraud represents an extreme needle-in-a-haystack problem: out of **284,807 transactions** in the benchmark Kaggle/ULB dataset, only **492 are fraudulent (0.1727%)**.

This platform combines two complementary distributed & machine-learning paradigms:
1. **Apache Hadoop MapReduce (Hadoop Streaming)**: Ingests the raw dataset on HDFS, executing distributed Python mappers, local combiners, and reducers to generate class distributions, hourly fraud volumes, amount bucketing, and evaluate a domain heuristic rule detector (`V14 < -4 OR V12 < -4`).
2. **Supervised Machine Learning (Random Forest & Logistic Regression)**: Trains class-weighted, stratified models achieving **84.2% precision, 81.6% recall, and 0.8466 PR-AUC**.
3. **Live Web Interface**: Offers interactive single-transaction scoring with explainable AI (Top 5 Signals), batch CSV screening, MapReduce pipeline visualization, an in-browser MapReduce simulator, and dark/light themes.

---

## 2. System Architecture

```
[Kaggle CreditCard CSV (284,807 rows)]
                   │
                   ▼
       [HDFS Distributed Storage]
       (Split into 128 MB blocks)
                   │
                   ▼
     [Hadoop Streaming MapReduce]
   ┌────────────────────────────────┐
   │ mapper.py  (STDOUT key-values) │
   │ reducer.py (Local Combiner)    │
   │ Shuffle & Sort (Group by key)  │
   │ reducer.py (Global Reducer)    │
   └────────────────┬───────────────┘
                    │
                    ▼
           [Data Layer: data/]
   ├── hadoop_by_class.csv
   ├── hadoop_by_amount.csv
   ├── hadoop_by_hour.csv
   ├── summary.json
   ├── model_metrics.csv
   ├── top_suspicious.csv
   ├── fraud_model.joblib (Trained Random Forest)
   └── demo_transactions.csv
                    │
                    ▼
     [FastAPI High-Performance API] (Port 8000)
    (Pydantic validation, joblib inference, explainability)
                    │
                    ▼
     [React + Vite + TypeScript UI] (Port 5173)
  (Tailwind CSS, Recharts, Interactive Simulator, Risk Gauge)
```

---

## 3. Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pandas, Scikit-Learn 1.8.0, Joblib, Pytest, Python-Multipart.
- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, Recharts, Lucide Icons, React Router.
- **Distributed Computing**: Apache Hadoop 3.3.6 (Hadoop Streaming with Python mapper/reducer and combiner).
- **Design System**: Fintech dark mode (`#060913` navy) with electric teal accents (`#06b6d4`), responsive layouts, and accessible ARIA risk palettes.

---

## 4. Quick Start (Run Locally)

### Prerequisites
- Python 3.11 or 3.12 installed
- Node.js (v18+) and npm installed

### Step 1: Clone or Open Workspace
```bash
cd "d:\8th Semester\CDC\LAB\Project"
```

### Step 2: Start the Backend (Port 8000)
```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
*Backend documentation is live at `http://127.0.0.1:8000/docs`.*

### Step 3: Start the Frontend (Port 5173)
In a new terminal:
```powershell
cd frontend
npm.cmd run dev
```
*Open your browser at `http://127.0.0.1:5173/`.*

---

## 5. Running Automated Backend Tests

The backend includes a comprehensive `pytest` test suite verifying health status, Hadoop summary metrics, sample retrieval, validation errors, and model scoring accuracy:

```powershell
python -m pytest backend/tests -v
```

All 13 unit and integration tests pass in under 1 second.

---

## 6. How to Replace Data/Model Files

When you re-run the `Fraud_Detection_Hadoop_Colab.ipynb` notebook in Google Colab:
1. At the end of the notebook execution, download the generated `fraud_results.zip`.
2. Extract the files directly into the `data/` folder of this project:
   - `summary.json`
   - `hadoop_by_class.csv`
   - `hadoop_by_amount.csv`
   - `hadoop_by_hour.csv`
   - `model_metrics.csv`
   - `top_suspicious.csv`
   - `dashboard.png`
   - `fraud_model.joblib`
   - `demo_transactions.csv`
3. Restart the backend server (`uvicorn app.main:app --reload`). The application will instantly reload all metrics without any code changes.

---

## 7. Backend API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health status, model loaded flag, and model file name |
| `GET` | `/api/summary` | Overall dataset totals, fraud counts, timing, and rule detector stats |
| `GET` | `/api/stats/class` | MapReduce class output (counts, sum, min, max, mean per class) |
| `GET` | `/api/stats/amount` | MapReduce transaction amount buckets (genuine, fraud, fraud rate %) |
| `GET` | `/api/stats/hour` | MapReduce hourly distribution (0-23 hours mod 24) |
| `GET` | `/api/metrics` | Model comparison table (Random Forest, Logistic Regression, Rule Detector) |
| `GET` | `/api/suspicious` | Top suspicious cases identified during model testing |
| `GET` | `/api/samples` | List of 100 test samples (with ID and actual ground-truth Class) |
| `GET` | `/api/samples/{id}` | Retrieve individual sample by ID |
| `GET` | `/api/feature-importance` | Top 10 Random Forest feature importances |
| `POST` | `/api/predict` | Single transaction inference (computes probability, risk tier, and top 5 signals) |
| `POST` | `/api/predict-batch` | Batch CSV upload (scores up to 5,000 rows with aggregate metrics) |
| `GET` | `/api/dashboard-image` | Serves `dashboard.png` exported from the Colab notebook |

---

## 8. How to Explain This in the Viva (Exam Q&A)

Here are the key questions your examiner is likely to ask, along with clear technical answers:

### Q1: Why did you use Hadoop MapReduce for credit card fraud detection?
> **Answer:** Credit card transaction networks (e.g., Visa, Mastercard) process tens of thousands of transactions per second, generating terabytes of raw logs daily. Storing these logs in HDFS allows linear, parallel disk scans across independent DataNodes. We used **Hadoop Streaming** with Python mappers to process transactions concurrently, extracting statistical aggregates (hourly patterns, amount distributions) and running rule-based filters without loading the entire dataset into memory.

### Q2: Why is the Combiner so critical in your Hadoop Streaming job?
> **Answer:** In MapReduce, the **Shuffle & Sort** stage transfers intermediate data over the network from mappers to reducers. If 284,807 transactions each emit a key-value record, 284,807 lines travel across the network. Our reducer logic (accumulating count, sum, min, max) is **associative and commutative**, meaning we can run `reducer.py` directly as a **Combiner** locally on each mapper node. The combiner pre-aggregates intermediate records into a single line per key before sending it across the network, reducing cluster network traffic by over 95%.

### Q3: Why is Accuracy a misleading metric for this dataset?
> **Answer:** Out of 284,807 transactions, 284,315 are genuine (99.827%) and only 492 are fraudulent (0.1727%). A dummy classifier that simply predicts every transaction is genuine achieves **99.83% accuracy**, but catches **0 frauds**. In imbalanced domains, accuracy is deceptive. We must evaluate models based on **Precision** (avoiding false alarms), **Recall** (catching real frauds), **F1-Score**, and **PR-AUC** (Precision-Recall Area Under Curve).

### Q4: Why was Pandas faster than Hadoop in your Colab run?
> **Answer:** In our Colab run, Pandas finished in **2.63 seconds** while Hadoop took **13.71 seconds**. This is because Hadoop was executed in **pseudo-distributed mode on a single machine** with a relatively small 150 MB dataset. Hadoop incurs fixed startup latency: launching the JVM, initializing HDFS block readers, and spawning Python streaming subprocesses. In a real-world multi-node cluster with 500 GB or 5 TB of transactions, Pandas would crash with an `OutOfMemoryError`, whereas Hadoop scales horizontally across dozens of machines.

### Q5: How does your Live Fraud Checker explain "Why" a transaction was flagged?
> **Answer:** Rather than acting as an unexplainable black box, our backend computes the **Top 5 Risk Signals** for each transaction. It calculates:
> $$\text{Score}_i = \text{FeatureImportance}_i \times \left|\frac{x_i - \text{Median}_{\text{genuine}, i}}{\sigma_{\text{genuine}, i}}\right|$$
> This multiplies the global importance of each feature by the transaction's standardized standard-deviation distance from the genuine baseline. For example, if $V_{14}$ is $-6.2$ (over $4\sigma$ below normal), it ranks as the top risk signal because $V_{14}$ has high predictive importance and shows anomalous deviation.

### Q6: What are the differences between your Rule-Based Detector and Random Forest?
> **Answer:**
> - **Hadoop Rule Detector (`V14 < -4 OR V12 < -4`)**: Fast, zero-training heuristic computed directly in MapReduce. It caught **403 out of 492 frauds** (81.9% recall), but produced 652 false alarms (precision: 38.2%).
> - **Random Forest (Supervised ML)**: Learns complex multi-dimensional decision boundaries across all 30 features with balanced subsampling. It achieved **84.2% precision**, **81.6% recall**, and an **F1-score of 82.9%**, eliminating the vast majority of false positives.

---

## 9. Academic Disclaimer

This web application and its machine learning predictions are strictly experimental university research outputs developed for the **Cloud and Distributed Computing** course. All predictions are generated for educational demonstration and must **never** be used for real financial authorizations or commercial banking decisions.

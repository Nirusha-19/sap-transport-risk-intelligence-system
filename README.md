---
title: SAP Transport Risk Intelligence System
emoji: 🚀
colorFrom: red
colorTo: purple
sdk: docker
app_port: 8501
tags:
  - streamlit
  - machine-learning
  - deep-learning
  - tabnet
  - shap
  - mlflow
  - firebase
  - groq
pinned: false
short_description: SAP Transport Risk Analysis using ML, SHAP and RAG
license: mit
---

# 🚀 SAP Transport Risk Intelligence System

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/nirusha-mr/sap-transport-risk-intelligence-system)

---

## 🎯 What Is This?

This is an intelligent risk analysis system for SAP transports, built to classify, explain, and act on deployment risk before it reaches Production.

In large enterprise SAP environments, a single bad transport in any of the four core modules can corrupt data, break pricing, or cause costly downtime:

- **FI (Finance)** handles the General Ledger, accounts payable, and financial reporting. A faulty deployment can corrupt financial records and cause audit failures.
- **MM (Materials Management)** controls procurement, inventory, and pricing procedures. A faulty deployment can break supplier pricing and stock management.
- **SD (Sales and Distribution)** manages customer orders and pricing conditions. A faulty deployment can cause incorrect billing and order processing failures.
- **HR (Human Resources)** handles payroll calculations and GDPR-sensitive employee records. A faulty deployment can result in incorrect salary processing and compliance violations.

Manual risk review is slow, inconsistent, and doesn't scale. This system catches that risk early, automatically.

---

## 🔍 What It Does

- **Classifies** the risk level of every SAP transport as LOW, MEDIUM, or HIGH — 5 ML models are trained and compared, and the best one is auto-selected based on F1 and CV-F1 score (best achieved: 0.9513)
- **Explains** each prediction using SHAP, showing which features drove the risk up or down
- **Flags** unusual transports using Isolation Forest anomaly detection
- **Answers questions** about transport risk using Groq LLaMA 3.3 70B backed by RAG, grounded in a curated SAP knowledge base
- **Saves** every prediction, AI response, and chat session to Firebase for audit and history
- **Calculates** business impact — estimated annual savings, ROI percentage, payback period, and SLA compliance

---

## 📊 Model Performance

Five models were trained on **10,000 SAP transport records** with SMOTE class balancing, and tracked using MLflow:

| Model | F1 Score | Type |
|-------|:--------:|------|
| **CatBoost ✅ Winner** | **0.9513** | Gradient Boosting |
| TabNet | 0.9507 | Deep Learning |
| LightGBM | 0.9502 | Gradient Boosting |
| RandomForest | 0.9426 | Ensemble |
| XGBoost | 0.9417 | Gradient Boosting |

The best model is selected automatically based on F1 + CV-F1 score after each training run. SHAP explanations use the best tree-based model (CatBoost) since TabNet is not compatible with SHAP TreeExplainer.

---

## 🖥️ 10 Dashboard Tabs

**📊 Dashboard**
Choose your data source — upload a CSV file or load directly from Firebase. Once loaded, the dashboard shows the overall risk distribution across all transports, summary metric cards (total transports, HIGH / MEDIUM / LOW / Critical counts), per-module risk gauges showing the risk breakdown for FI, MM, SD, and HR, confidence score distribution, individual graph downloads, and a live monitor toggle.

---

**🔍 Anomaly Detection**
Isolation Forest scans all transports and flags statistically unusual ones. Displays three summary numbers — total anomalies detected, normal transports, and anomaly rate. Results are shown in a table with Transport ID, Module, Transport Stage, Predicted Risk, Confidence %, and Anomaly flag. An anomaly scatter plot is displayed alongside an AI Explain Anomalies option.

---

**📄 Data Explorer**
Browse and filter the full dataset by Transport ID, Module (FI, MM, SD, HR), Predicted Risk, Objects Changed, Transport Stage, and more. Results are color-coded by risk level. Both the full dataset and filtered results can be downloaded as CSV or synced directly to Firebase.

---

**🤖 AI Insights**
Select any transport and click **Generate AI Insight** to get a detailed explanation of why that transport was classified at its risk level, what the business impact could be, and what actions to take. Additional options include **Summarise All Risks** for a full batch summary, **Generate Executive Email** to draft a ready-to-send stakeholder email, and a **7-Day Risk Trend Forecast**. There is also an **Ask AI** chat window which generates a structured, grounded response. Chat history is saved to Firebase per session.

---

**⚡ Manual Prediction**
Select the transport attributes — Module, Stage, and Status from dropdown menus, and adjust Objects Changed, Lines Changed, Conflicts, History Failures and more, then click redict Risk. The result is shown instantly with the predicted risk level and confidence score, a probability breakdown chart across all three classes, a SHAP Feature Impact bar chart showing which features increased or decreased the risk, and an Input Feature Values chart. The prediction is saved to Firebase automatically and an AI Explanation is generated below.

---

**🔄 What-If Analysis**
Compare two risk scenarios side by side — set the attributes for Scenario A (current state) and Scenario B (improved state), and the system instantly predicts the risk level and confidence score for both. This helps identify which attributes have the most impact on bringing a transport from HIGH risk down to LOW risk.

---

**📜 Prediction History**
A complete log of every prediction made, saved to Firebase with timestamp, transport details, predicted risk level, confidence score, and more.

---

**💰 Business Impact & ROI**
Enter cost and prevention parameters to quantify the financial value of SAP transport risk prevention. Outputs include total annual savings, ROI percentage, payback period, incidents prevented, downtime prevented, net annual savings, and SLA compliance score — shown through an annual savings vs cost breakdown graph, a potential savings by module graph across FI, MM, SD, and HR, and a current risk distribution pie chart.

---

**🔬 SHAP Explainability**
Shows which features drive each risk prediction using CatBoost. A global feature importance chart ranks all features across 100 records, and a single record explanation shows individual feature contributions — red pushes toward HIGH risk, green pushes away.

---

**📈 Model Comparison**
A live leaderboard showing Test F1 and CV F1 scores for all 5 trained models. All models were trained with SMOTE balancing and tracked with MLflow.

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Web App | Streamlit |
| Deep Learning | PyTorch · TabNet |
| Gradient Boosting | XGBoost · LightGBM · CatBoost |
| Classical ML | scikit-learn — RandomForest · Isolation Forest |
| Explainability | SHAP |
| LLM | Groq API — LLaMA 3.3 70B |
| RAG | TF-IDF + Cosine Similarity |
| Cloud Storage | Firebase |
| Experiment Tracking | MLflow |
| Visualisation | Plotly · Matplotlib · Seaborn |
| PDF Reports | ReportLab |
| Deployment | Docker · HuggingFace Spaces |

---

## 📁 Project Structure

sap-transport-risk-intelligence-system/

├── app_firebase.py          ← Main Streamlit application (10 tabs)

├── model/

│   ├── model.pkl            ← Best model (CatBoost)

│   ├── shap_explainer.pkl   ← SHAP explainer (CatBoost)

│   ├── model_comparison.pkl ← All 5 model results

│   └── train_model.py       ← Training pipeline

├── utils/

│   ├── firebase_helper.py   ← Firebase CRUD operations

│   ├── ml_helper.py         ← Preprocessing + SHAP utilities

│   ├── rag_helper.py        ← RAG engine + knowledge base

│   └── gauge.py             ← Risk gauge component

├── scripts/

│   └── generate_dataset.py  ← Synthetic dataset generation

├── data/

│   └── sap_transport_dataset.csv

├── Dockerfile

└── requirements.txt

---

## 💡 What Makes This Different

❌ Most SAP risk tools → rule-based, no explanation
✅ This system → ML + Deep Learning + Explainable AI

❌ Generic LLM chatbot → hallucinations, no SAP context
✅ RAG pipeline → grounded in curated SAP policy

❌ Single model → no confidence in results
✅ 5 models compared → MLflow tracked, best auto-selected

❌ Local predictions → no audit trail
✅ Firebase cloud → every prediction, insight, chat saved

---

## 🚀 Run Locally

```bash
git clone https://huggingface.co/spaces/nirusha-mr/sap-transport-risk-intelligence-system
cd sap-transport-risk-intelligence-system
pip install -r requirements.txt
streamlit run app_firebase.py
```

Retrain all 5 models from scratch:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 model/train_model.py
```

---

## 🔑 Secrets Required

| Secret | Where to Get It |
|--------|----------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `FIREBASE_KEY_BASE64` | Firebase Console → Service Accounts → base64 encode the JSON |

---

## 👩‍💻 Author

**Nirusha Mantralaya Ramesh**
- 🐙 GitHub: [Nirusha-19](https://github.com/Nirusha-19)

---

## 📄 License

MIT — free to use, fork, and build upon.
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
from sklearn.ensemble import IsolationForest
import time
import uuid
import numpy as np

from utils.gauge import risk_gauge
from utils.ml_helper import preprocess, predict, get_shap_values, shap_summary_for_row
from utils.rag_helper import get_rag_engine
from utils.firebase_helper import (
    upload_dataframe_to_firestore,
    fetch_from_firestore,
    save_prediction,
    fetch_predictions,
    save_ai_insight,
    fetch_ai_insights,
    save_chat_history,
    fetch_chat_history,
    clear_collection
)

# ================================================================
# CONFIG
# ================================================================
load_dotenv()
st.set_page_config(page_title="SAP Risk AI", layout="wide", page_icon="🚀")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ================================================================
# LANDING PAGE HERO
# ================================================================
st.markdown("""
<div style='background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
     padding: 40px 30px; border-radius: 20px; text-align: center;
     margin-bottom: 25px; border: 1px solid #764ba2;
     box-shadow: 0 0 40px rgba(118,75,162,0.4)'>
    <div style='font-size:3em; margin-bottom:10px'>🚀</div>
    <h1 style='color: white; font-size: 2.4em; margin:0; letter-spacing:1px'>
        SAP Transport Risk Intelligence System
    </h1>
    <p style='color: #b39ddb; font-size: 1.0em; margin-top:10px'>
        Machine Learning &nbsp;•&nbsp; Deep Learning &nbsp;•&nbsp; Explainable AI &nbsp;•&nbsp; RAG-Powered LLM &nbsp;•&nbsp; Firebase Cloud &nbsp;•&nbsp; Anomaly Detection &nbsp;•&nbsp; MLflow Tracking
    </p>
    <div style='margin-top:15px; display:flex; justify-content:center; gap:8px; flex-wrap:nowrap'>
        <span style='background:#4a148c;color:white;padding:3px 10px;border-radius:20px;font-size:0.75em;white-space:nowrap'>🧠 TabNet Deep Learning</span>
        <span style='background:#1565c0;color:white;padding:3px 10px;border-radius:20px;font-size:0.75em;white-space:nowrap'>📊 5-Model MLflow Tracking</span>
        <span style='background:#00695c;color:white;padding:3px 10px;border-radius:20px;font-size:0.75em;white-space:nowrap'>🔍 SHAP Explainability</span>
        <span style='background:#e65100;color:white;padding:3px 10px;border-radius:20px;font-size:0.75em;white-space:nowrap'>🤖 Groq LLaMA 3.3 70B</span>
        <span style='background:#bf360c;color:white;padding:3px 10px;border-radius:20px;font-size:0.75em;white-space:nowrap'>🔥 Firebase Firestore</span>
        <span style='background:#37474f;color:white;padding:3px 10px;border-radius:20px;font-size:0.75em;white-space:nowrap'>⚡ Real-Time Monitoring</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# GROQ + RAG SETUP
# ================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)
rag          = get_rag_engine()

# ================================================================
# LOAD MODEL + SHAP + COMPARISON
# ================================================================
import os
base_path = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(base_path, "model/model.pkl"), "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"⚠ Model error: {e}")
    st.stop()

try:
    with open(os.path.join(base_path, "model/shap_explainer.pkl"), "rb") as f:
        shap_explainer = pickle.load(f)
except:
    shap_explainer = None

try:
    with open(os.path.join(base_path, "model/model_comparison.pkl"), "rb") as f:
        model_comparison = pickle.load(f)
except:
    model_comparison = None

try:
    with open(os.path.join(base_path, "model/best_model_name.txt"), "r") as f:
        best_model_name = f.read().strip()
except:
    best_model_name = "Unknown"
TREE_MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
shap_model_name = best_model_name if best_model_name in TREE_MODELS else "CatBoost"

# ================================================================
# HELPERS
# ================================================================
def save_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def ask_ai(prompt: str) -> str:
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {e}"

def ask_ai_rag(query: str, transport_ctx=None, shap_ctx=None) -> str:
    prompt = rag.build_prompt(query, transport_ctx, shap_ctx)
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        return res.choices[0].message.content
    except Exception as e:
        return rag.answer_no_llm(query)

def ask_ai_with_history(messages: list) -> str:
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {e}"

def get_confidence(df_proc):
    try:
        # Handle TabNet which needs float32 numpy array
        if hasattr(df_proc, 'values'):
            X = df_proc.values.astype(np.float32)
        else:
            X = np.array(df_proc).astype(np.float32)

        # TabNet predict_proba returns tuple sometimes
        result = model.predict_proba(X)
        if isinstance(result, tuple):
            proba = result[0]
        else:
            proba = result 

        # TabNet doesn't have classes_ attribute like sklearn
        if hasattr(model, 'classes_'):
            classes = model.classes_
        else:
            classes = np.array([0, 1, 2])

        return (proba.max(axis=1)*100).round(2), classes, proba
    except:
        return None, None, None

def generate_pdf(filtered_df, ai_summary, date_range=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("SAP Transport Risk Intelligence Report", styles['Title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    ))
    if date_range:
        story.append(Paragraph(f"Date Range: {date_range}", styles['Normal']))
    story.append(Spacer(1, 20))

    high  = (filtered_df['Predicted Risk']=="HIGH").sum()
    med   = (filtered_df['Predicted Risk']=="MEDIUM").sum()
    low   = (filtered_df['Predicted Risk']=="LOW").sum()
    total = len(filtered_df)

    story.append(Paragraph("Risk Summary", styles['Heading1']))
    story.append(Spacer(1, 8))
    s_data = [
        ['Metric','Value'],
        ['Total Transports', str(total)],
        ['HIGH Risk', str(high)],
        ['MEDIUM Risk', str(med)],
        ['LOW Risk', str(low)],
        ['High Risk %', f"{round(high/total*100,2)}%" if total else "0%"],
    ]
    st_ = Table(s_data, colWidths=[3*inch,3*inch])
    st_.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#302b63')),
        ('TEXTCOLOR', (0,0),(-1,0),colors.white),
        ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',     (0,0),(-1,-1),'CENTER'),
        ('GRID',      (0,0),(-1,-1),1,colors.black),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#ecf0f1'),colors.white]),
    ]))
    story.append(st_)
    story.append(Spacer(1,20))

    story.append(Paragraph("Module-wise Risk Breakdown", styles['Heading1']))
    story.append(Spacer(1,8))
    m_data = [['Module','HIGH','MEDIUM','LOW','Total']]
    for mod in filtered_df['module'].unique():
        mdf = filtered_df[filtered_df['module']==mod]
        m_data.append([
            mod,
            str((mdf['Predicted Risk']=="HIGH").sum()),
            str((mdf['Predicted Risk']=="MEDIUM").sum()),
            str((mdf['Predicted Risk']=="LOW").sum()),
            str(len(mdf))
        ])
    mt = Table(m_data, colWidths=[1.5*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch])
    mt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0,0),(-1,0),colors.white),
        ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',     (0,0),(-1,-1),'CENTER'),
        ('GRID',      (0,0),(-1,-1),1,colors.black),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#fadbd8'),colors.white]),
    ]))
    story.append(mt)
    story.append(Spacer(1,20))

    critical = filtered_df[
        (filtered_df['transport_stage']=="Production") &
        (filtered_df['Predicted Risk']=="HIGH")
    ]
    if not critical.empty:
        story.append(Paragraph(
            "Critical Transports (HIGH Risk in Production)", styles['Heading1']
        ))
        story.append(Spacer(1,8))
        c_data = [['Transport ID','Module','Stage','Risk']]
        for _, row in critical.head(20).iterrows():
            c_data.append([
                str(row['transport_id']), str(row['module']),
                str(row['transport_stage']), str(row['Predicted Risk'])
            ])
        ct = Table(c_data, colWidths=[2*inch,1.5*inch,1.5*inch,1.5*inch])
        ct.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0,0),(-1,0),colors.white),
            ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
            ('ALIGN',     (0,0),(-1,-1),'CENTER'),
            ('GRID',      (0,0),(-1,-1),1,colors.black),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#fadbd8'),colors.white]),
        ]))
        story.append(ct)
        story.append(Spacer(1,20))

    story.append(Paragraph("AI-Generated Risk Analysis", styles['Heading1']))
    story.append(Spacer(1,8))
    clean = ai_summary.replace('*','').replace('#','')
    for line in clean.split('\n'):
        if line.strip():
            story.append(Paragraph(line.strip(), styles['Normal']))
            story.append(Spacer(1,5))

    doc.build(story)
    buf.seek(0)
    return buf

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown("## 📂 Data Source")
data_source = st.sidebar.radio(
    "Choose Source", ["📁 Upload CSV", "🔥 Load from Firebase"]
)

df = pd.DataFrame()

if data_source == "📁 Upload CSV":
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV file(s)", type=["csv"], accept_multiple_files=True
    )
    if uploaded_files:
        df = pd.concat(
            [pd.read_csv(f) for f in uploaded_files], ignore_index=True
        )
        st.sidebar.success(f"✅ {len(df)} records loaded")
        if st.sidebar.button("🔥 Save to Firebase"):
            with st.spinner("Uploading to Firebase Firestore..."):
                try:
                    count = upload_dataframe_to_firestore(df)
                    st.sidebar.success(f"✅ {count} records saved to Firebase!")
                except Exception as e:
                    st.sidebar.error(f"Firebase Error: {e}")

elif data_source == "🔥 Load from Firebase":
    if st.sidebar.button("🔄 Fetch from Firebase"):
        with st.spinner("Fetching from Firebase..."):
            try:
                df = fetch_from_firestore()
                if df.empty:
                    st.sidebar.warning("No data found. Upload CSV first.")
                else:
                    st.sidebar.success(f"🔥 {len(df)} records fetched!")
                    st.session_state['firebase_df'] = df
            except Exception as e:
                st.sidebar.error(f"Firebase Error: {e}")
    if 'firebase_df' in st.session_state:
        df = st.session_state['firebase_df']
        st.sidebar.info(f"🔥 Using {len(df)} Firebase records")

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚡ Live Monitor")
live_mode    = st.sidebar.toggle("🔴 Enable Live Monitor", value=False)
refresh_rate = 30
if live_mode:
    refresh_rate = st.sidebar.slider("Refresh every (seconds)", 10, 120, 30)
    st.sidebar.success(f"Auto-refreshing every {refresh_rate}s")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔥 Firebase Admin")
if st.sidebar.button("🗑 Clear Predictions History"):
    try:
        n = clear_collection("predictions")
        st.sidebar.success(f"✅ Cleared {n} predictions")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ================================================================
# MAIN APP
# ================================================================
if not df.empty:

    if live_mode:
        st.markdown(f"""
        <div style='background:#1a1a2e;border:2px solid #e74c3c;
             border-radius:10px;padding:10px 20px;
             display:flex;align-items:center;gap:10px;margin-bottom:15px'>
            <span style='color:#e74c3c;font-size:1.5em'>●</span>
            <span style='color:white;font-weight:bold'>LIVE MONITORING ACTIVE</span>
            <span style='color:#aaa;font-size:0.85em'>
                — Auto-refreshing every {refresh_rate}s
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.success(f"✅ {len(df)} transports loaded")
    st.dataframe(df.head(), use_container_width=True)

    df_original = df.copy()
    df_original = df_original.dropna(subset=['transport_id'])

    module_mapping = {0:'FI',1:'MM',2:'SD',3:'HR','0':'FI','1':'MM','2':'SD','3':'HR'}
    stage_mapping  = {0:'Development',1:'Quality',2:'Production',
                      '0':'Development','1':'Quality','2':'Production'}
    status_mapping = {0:'Approved',1:'Pending',2:'Rejected',
                      '0':'Approved','1':'Pending','2':'Rejected'}

    df_original['module']                = df_original['module'].replace(module_mapping)
    df_original['transport_stage']       = df_original['transport_stage'].replace(stage_mapping)
    df_original['change_request_status'] = df_original['change_request_status'].replace(status_mapping)
    df_original['module']                = df_original['module'].fillna('Unknown')
    df_original['transport_stage']       = df_original['transport_stage'].fillna('Unknown')
    df_original['change_request_status'] = df_original['change_request_status'].fillna('Unknown')
    df_original['objects_changed']   = pd.to_numeric(df_original.get('objects_changed',0),  errors='coerce').fillna(0)
    df_original['lines_changed']     = pd.to_numeric(df_original.get('lines_changed',0),    errors='coerce').fillna(0)
    df_original['conflicts']         = pd.to_numeric(df_original.get('conflicts',0),        errors='coerce').fillna(0)
    df_original['history_failures']  = pd.to_numeric(df_original.get('history_failures',0), errors='coerce').fillna(0)
    
    df_processed = preprocess(df_original.copy())
    df_original['Predicted Risk'] = predict(model, df_processed)

    X_conf = df_processed[['module','objects_changed','lines_changed',
                            'conflicts','history_failures',
                            'transport_stage','change_request_status']].values.astype(np.float32)
    conf_scores, classes, all_proba = get_confidence(
        pd.DataFrame(X_conf, columns=['module','objects_changed','lines_changed',
                                      'conflicts','history_failures',
                                      'transport_stage','change_request_status'])
    )
    df_original['Confidence %'] = conf_scores if conf_scores is not None else "N/A"
    
    numeric_cols = [c for c in
        ['objects_changed','lines_changed','conflicts','history_failures']
        if c in df_original.columns]

    if numeric_cols:
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(df_original[numeric_cols])
        df_original['Anomaly'] = iso.predict(df_original[numeric_cols])
        df_original['Anomaly'] = df_original['Anomaly'].map({1:"Normal",-1:"⚠ Anomaly"})

    # ---- FILTERS ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 Filters")
    tid = st.sidebar.text_input("🔎 Transport ID")

    module_options = [x for x in df_original['module'].unique()
                      if str(x) != 'nan' and x is not None]
    module_filter  = st.sidebar.multiselect(
        "📦 Module", module_options, default=module_options
    )
    risk_options = [x for x in df_original['Predicted Risk'].unique()
                    if str(x) != 'nan' and x is not None]
    risk_filter  = st.sidebar.multiselect(
        "⚠ Risk Level", risk_options, default=risk_options
    )

    st.sidebar.markdown("## 📅 Date Filter")
    date_range_label = "All Data"
    if 'date' in df_original.columns:
        df_original['date'] = pd.to_datetime(df_original['date'], errors='coerce')
        min_date = df_original['date'].min()
        max_date = df_original['date'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_from = st.sidebar.date_input("From", min_date)
            date_to   = st.sidebar.date_input("To",   max_date)
            date_range_label = f"{date_from} to {date_to}"
    else:
        st.sidebar.info("No 'date' column found")

    filtered_df = df_original.copy()
    if tid:
        filtered_df = filtered_df[
            filtered_df['transport_id'].astype(str).str.contains(tid)
        ]
    if module_filter:
        filtered_df = filtered_df[filtered_df['module'].isin(module_filter)]
    if risk_filter:
        filtered_df = filtered_df[filtered_df['Predicted Risk'].isin(risk_filter)]
    if 'date' in df_original.columns and pd.notna(df_original['date'].min()):
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(date_from)) &
            (filtered_df['date'] <= pd.Timestamp(date_to))
        ]

    # ================================================================
    # TABS
    # ================================================================
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
        "📊 Dashboard",
        "🔍 Anomaly Detection",
        "📄 Data Explorer",
        "🤖 AI Insights",
        "⚡ Manual Prediction",
        "🔄 What-If Analysis",
        "📜 Prediction History",
        "💰 Business Impact & ROI",
        "🔬 SHAP Explainability",
        "📈 Model Comparison",
    ])

    cmap = {"HIGH":"#ff4d4d","MEDIUM":"#f1c40f","LOW":"#2ecc71"}

    # ============================================================
    # TAB 1 — DASHBOARD
    # ============================================================
    with tab1:
        st.subheader("📊 Business Dashboard")

        high_n  = (filtered_df['Predicted Risk']=="HIGH").sum()
        med_n   = (filtered_df['Predicted Risk']=="MEDIUM").sum()
        low_n   = (filtered_df['Predicted Risk']=="LOW").sum()
        total_n = len(filtered_df)
        critical = filtered_df[
            (filtered_df['transport_stage']=="Production") &
            (filtered_df['Predicted Risk']=="HIGH")
        ]
        anomaly_n = (filtered_df['Anomaly']=="⚠ Anomaly").sum() \
                    if 'Anomaly' in filtered_df.columns else 0

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("📦 Total",       total_n)
        c2.metric("🔴 HIGH Risk",   high_n,
                  delta=f"{round(high_n/total_n*100,1)}%" if total_n else "0%")
        c3.metric("🟡 MEDIUM Risk", med_n)
        c4.metric("🟢 LOW Risk",    low_n)
        c5.metric("🚨 Critical",    len(critical))

        st.divider()

        if not critical.empty:
            st.markdown(f"""
            <div style='background:linear-gradient(90deg,#c0392b,#e74c3c);
                 color:white;padding:15px 20px;border-radius:12px;
                 font-size:1.1em;font-weight:bold;margin-bottom:10px'>
                🚨 CRITICAL ALERT — {len(critical)} HIGH-RISK transports in PRODUCTION.
                Immediate action required!
            </div>""", unsafe_allow_html=True)
            with st.expander("👁 View Critical Transports"):
                st.dataframe(
                    critical[['transport_id','module','transport_stage',
                               'Predicted Risk','Confidence %']],
                    use_container_width=True
                )
        elif high_n > 0:
            st.warning(f"⚠ {high_n} HIGH-risk transports detected")
        else:
            st.success("✅ All transports within acceptable risk levels")

        if anomaly_n > 0:
            st.warning(f"🔍 {anomaly_n} anomalous transports detected by AI")

        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #4caf50;border-radius:8px;
             padding:8px 16px;display:inline-block;margin:8px 0'>
            <span style='color:#4caf50'>🔥 Firebase Connected</span>
            <span style='color:#aaa;font-size:0.85em'>
              &nbsp;— Data persisted to Firestore cloud database
            </span>
        </div>""", unsafe_allow_html=True)

        st.divider()

        if total_n > 0:
            pct   = int((high_n/total_n)*100)
            color = "#ff4d4d" if pct>70 else "#f1c40f" if pct>40 else "#2ecc71"
            st.markdown(risk_gauge(pct, color), unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📦 Module Risk Indicators")
        modules = filtered_df['module'].unique()
        cols    = st.columns(len(modules))
        for i, mod in enumerate(modules):
            mdf  = filtered_df[filtered_df['module']==mod]
            hc   = (mdf['Predicted Risk']=="HIGH").sum()
            tot  = len(mdf)
            pmod = int((hc/tot)*100) if tot>0 else 0
            cmod = "#ff4d4d" if pmod>70 else "#f1c40f" if pmod>40 else "#2ecc71"
            with cols[i]:
                st.markdown(f"**{mod}**")
                st.markdown(risk_gauge(pmod, cmod), unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Analytics")
        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)

        with col1:
            fig1 = px.bar(filtered_df, x="Predicted Risk",
                          title="Risk Distribution",
                          color="Predicted Risk", color_discrete_map=cmap)
            st.plotly_chart(fig1, use_container_width=True)
            mf1 = plt.figure()
            sns.countplot(x='Predicted Risk', data=filtered_df)
            st.download_button("📥 Risk Chart", save_chart(mf1), "risk.png")

        with col2:
            fig2 = px.bar(filtered_df, x="module", color="Predicted Risk",
                          title="Module vs Risk", color_discrete_map=cmap)
            st.plotly_chart(fig2, use_container_width=True)
            mf2 = plt.figure()
            sns.countplot(x='module', hue='Predicted Risk', data=filtered_df)
            st.download_button("📥 Module Chart", save_chart(mf2), "module.png")

        with col3:
            fig3 = px.histogram(filtered_df, x="module", color="Predicted Risk",
                                barmode="group", title="Clustered Risk Analysis",
                                color_discrete_map=cmap)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.histogram(filtered_df, x="module", color="Predicted Risk",
                                barmode="stack", title="Stacked Risk Composition",
                                color_discrete_map=cmap)
            st.plotly_chart(fig4, use_container_width=True)

        if 'Confidence %' in filtered_df.columns and \
           filtered_df['Confidence %'].dtype != object:
            st.markdown("### 🎯 Confidence Distribution")
            fig_c = px.histogram(filtered_df, x='Confidence %',
                                 color='Predicted Risk', nbins=20,
                                 title="Model Confidence Distribution",
                                 color_discrete_map=cmap)
            st.plotly_chart(fig_c, use_container_width=True)

        st.download_button(
            "📥 Download Analytics CSV",
            data=filtered_df.to_csv(index=False),
            file_name="sap_analytics.csv", mime="text/csv"
        )

        if live_mode:
            st.info(f"🔄 Next refresh in {refresh_rate} seconds...")
            time.sleep(refresh_rate)
            st.rerun()

    # ============================================================
    # TAB 2 — ANOMALY DETECTION
    # ============================================================
    with tab2:
        st.subheader("🔍 AI Anomaly Detection")
        st.caption("Isolation Forest ML detects unusual transport patterns automatically")

        if 'Anomaly' in filtered_df.columns:
            a_cnt = (filtered_df['Anomaly']=="⚠ Anomaly").sum()
            n_cnt = (filtered_df['Anomaly']=="Normal").sum()

            c1,c2,c3 = st.columns(3)
            c1.metric("⚠ Anomalies",  a_cnt)
            c2.metric("✅ Normal",     n_cnt)
            c3.metric("Anomaly Rate",  f"{round(a_cnt/len(filtered_df)*100,1)}%")

            if a_cnt > 0:
                st.error(f"🚨 {a_cnt} anomalous transports detected!")
                anomalies = filtered_df[filtered_df['Anomaly']=="⚠ Anomaly"]
                st.dataframe(
                    anomalies[['transport_id','module','transport_stage',
                               'Predicted Risk','Confidence %','Anomaly']],
                    use_container_width=True
                )

                if numeric_cols and len(numeric_cols) >= 2:
                    fig_a = px.scatter(
                        filtered_df, x=numeric_cols[0], y=numeric_cols[1],
                        color='Anomaly', symbol='Anomaly',
                        title="Anomaly Scatter Plot",
                        color_discrete_map={"Normal":"#2ecc71","⚠ Anomaly":"#ff4d4d"}
                    )
                    st.plotly_chart(fig_a, use_container_width=True)

                if st.button("🤖 AI Explain Anomalies"):
                    prompt = f"""
                    Anomaly detection found {a_cnt} unusual SAP transports out of {len(filtered_df)}.
                    Anomalous stats: {anomalies[numeric_cols].describe().to_string() if numeric_cols else 'N/A'}
                    Normal stats: {filtered_df[filtered_df['Anomaly']=='Normal'][numeric_cols].describe().to_string() if numeric_cols else 'N/A'}
                    Explain: 1. What makes these anomalous 2. Business risk 3. Recommended actions
                    """
                    with st.spinner("AI analyzing anomalies..."):
                        st.write(ask_ai(prompt))
            else:
                st.success("✅ No anomalies detected in the dataset")

    # ============================================================
    # TAB 3 — DATA EXPLORER 
    # ============================================================
    with tab3:
        st.subheader("📄 Data Explorer")
        st.info("💡 Rows color-coded by Predicted Risk level.")

        # Color ONLY based on Predicted Risk column
        def highlight_risk(row):
            risk = row.get('Predicted Risk', '')
            if risk == 'HIGH':
                color = 'background-color:#ff4d4d;color:white'
            elif risk == 'MEDIUM':
                color = 'background-color:#fff3cd;color:black'
            elif risk == 'LOW':
                color = 'background-color:#d4edda;color:black'
            else:
                color = ''
            return [color] * len(row)

        # Show only relevant columns
        display_cols = [c for c in [
            'transport_id','module','objects_changed','lines_changed',
            'conflicts','history_failures','transport_stage',
            'change_request_status','Predicted Risk','Confidence %','Anomaly'
        ] if c in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button("📥 Full Data", df_original.to_csv(index=False), "full.csv")
        with c2:
            st.download_button("📥 Filtered Data", filtered_df.to_csv(index=False), "filtered.csv")
        with c3:
            if st.button("🔥 Sync Filtered to Firebase"):
                with st.spinner("Syncing..."):
                    try:
                        count = upload_dataframe_to_firestore(
                            filtered_df, collection="filtered_exports"
                        )
                        st.success(f"✅ {count} records synced!")
                    except Exception as e:
                        st.error(f"Sync error: {e}")

    # ============================================================
    # TAB 4 — AI INSIGHTS (RAG works silently behind the scenes)
    # ============================================================
    with tab4:
        st.subheader("🧠 AI Insights")
        st.caption(f"Powered by Groq LLaMA 3.3 70B | Best Model: {best_model_name}")

        if not filtered_df.empty:
            high_c  = (filtered_df['Predicted Risk']=="HIGH").sum()
            med_c   = (filtered_df['Predicted Risk']=="MEDIUM").sum()
            low_c   = (filtered_df['Predicted Risk']=="LOW").sum()
            total_c = len(filtered_df)

            c1,c2,c3 = st.columns(3)
            c1.metric("🔴 HIGH",   high_c)
            c2.metric("🟡 MEDIUM", med_c)
            c3.metric("🟢 LOW",    low_c)

            if 'ai_summary' not in st.session_state:
                st.session_state.ai_summary = ""

            # Single Transport Analysis
            st.markdown("### 🔍 Analyze Specific Transport")
            t_ids  = filtered_df['transport_id'].astype(str).tolist()
            sel_id = st.selectbox("Select Transport ID", t_ids)
            row    = filtered_df[
                filtered_df['transport_id'].astype(str)==sel_id
            ].iloc[0]

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ID",     str(row['transport_id']))
            c2.metric("Module", row['module'])
            c3.metric("Stage",  row['transport_stage'])
            c4.metric("Risk",   row['Predicted Risk'])

            transport_ctx = {
                "module":           row['module'],
                "risk":             row['Predicted Risk'],
                "stage":            row['transport_stage'],
                "lines_changed":    row.get('lines_changed','N/A'),
                "conflicts":        row.get('conflicts','N/A'),
                "history_failures": row.get('history_failures','N/A'),
            }

            shap_ctx = None
            if shap_explainer is not None:
                try:
                    row_proc = preprocess(pd.DataFrame([row.to_dict()]))
                    shap_ctx = shap_summary_for_row(
                        shap_explainer, row_proc, row['Predicted Risk']
                    )
                except:
                    pass

            query = (f"Explain why transport {row['transport_id']} "
                     f"in module {row['module']} is {row['Predicted Risk']} risk.")

            if st.button("🔍 Generate AI Insight"):
                with st.spinner("AI analyzing transport..."):
                    # RAG works silently behind the scenes
                    insight = ask_ai_rag(query, transport_ctx, shap_ctx)
                st.write(insight)
                try:
                    save_ai_insight(row['transport_id'], insight)
                    st.caption("✅ Insight saved to Firebase")
                except Exception as e:
                    st.caption(f"⚠ Could not save: {e}")

            with st.expander("📚 View Saved AI Insights from Firebase"):
                try:
                    insights = fetch_ai_insights()
                    if insights:
                        for tid_key, data in insights.items():
                            st.markdown(f"**Transport:** {tid_key}")
                            st.write(data.get('insight',''))
                            st.markdown(f"*Saved: {data.get('timestamp','')}*")
                            st.divider()
                    else:
                        st.info("No saved insights yet")
                except Exception as e:
                    st.error(f"Firebase fetch error: {e}")

            st.divider()

            col1,col2 = st.columns(2)
            with col1:
                if st.button("📋 Summarize All Risks"):
                    prompt = f"""
                    SAP Transport Dataset:
                    Total={total_c}, HIGH={high_c}, MEDIUM={med_c}, LOW={low_c}
                    Module breakdown:
                    {filtered_df.groupby(['module','Predicted Risk']).size().to_string()}
                    1) Overall summary 2) Riskiest modules 3) Top 3 recommendations
                    """
                    with st.spinner("AI analyzing..."):
                        result = ask_ai(prompt)
                    st.session_state.ai_summary = result
                    st.write(result)

            with col2:
                if st.button("📧 Generate Executive Email"):
                    top_mod = filtered_df.groupby('module')['Predicted Risk'].apply(
                        lambda x: (x=="HIGH").sum()
                    ).idxmax() if high_c > 0 else "N/A"
                    prompt = f"""
                    Write a professional executive email to the CTO:
                    - Total SAP Transports: {total_c}
                    - HIGH Risk: {high_c} ({round(high_c/total_c*100,1) if total_c else 0}%)
                    - MEDIUM Risk: {med_c}, LOW Risk: {low_c}
                    - Most at-risk module: {top_mod}
                    Format: Subject, greeting, bullet points, action items, sign-off.
                    """
                    with st.spinner("Drafting executive email..."):
                        email = ask_ai(prompt)
                    st.text_area("📧 Copy & send:", email, height=350)

            st.divider()
            st.markdown("### 🔮 7-Day Risk Trend Forecast")
            if st.button("📈 Predict Next 7 Days"):
                prod_count = len(filtered_df[filtered_df['transport_stage']=="Production"])
                top_module = filtered_df.groupby('module').size().idxmax()
                anomaly_c  = (filtered_df.get('Anomaly','Normal')=="⚠ Anomaly").sum()
                prompt = f"""
                Based on SAP transport data:
                HIGH={high_c}/{total_c}, Module={top_module},
                Production={prod_count}, Anomalies={anomaly_c}
                Predict 7-day risk trend as:
                - Day-by-day table (Day 1-7, Risk Level, Key Driver)
                - Overall 7-day outlook
                - Top action for this week
                """
                with st.spinner("Forecasting..."):
                    st.write(ask_ai(prompt))

            st.divider()
            st.markdown("### 📄 Export PDF Report")
            if st.button("🔄 Generate AI Summary for PDF"):
                with st.spinner("Generating..."):
                    prompt = f"""
                    SAP Risk Executive Report:
                    Total={total_c}, HIGH={high_c}, MEDIUM={med_c}, LOW={low_c}
                    Modules: {filtered_df.groupby(['module','Predicted Risk']).size().to_string()}
                    Write: executive summary, key risks, recommendations.
                    """
                    st.session_state.ai_summary = ask_ai(prompt)
                st.success("✅ Ready — click Download PDF")

            if st.session_state.ai_summary:
                pdf_buf = generate_pdf(
                    filtered_df, st.session_state.ai_summary, date_range_label
                )
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_buf,
                    file_name="sap_risk_report.pdf",
                    mime="application/pdf"
                )

            st.divider()
            st.subheader("💬 Ask AI")
            st.caption(f"Session: {st.session_state.session_id} — Chat history saved to Firebase")

            if 'chat_history' not in st.session_state:
                try:
                    saved = fetch_chat_history(st.session_state.session_id)
                    st.session_state.chat_history = saved if saved else [{
                        "role":"system",
                        "content":"You are an expert SAP Basis consultant and transport risk analyst. Answer clearly and professionally."
                    }]
                except:
                    st.session_state.chat_history = [{
                        "role":"system",
                        "content":"You are an expert SAP Basis consultant and transport risk analyst."
                    }]

            for msg in st.session_state.chat_history[1:]:
                if msg['role']=='user':
                    st.chat_message("user").write(msg['content'])
                else:
                    st.chat_message("assistant").write(msg['content'])

            query_chat = st.chat_input("Ask anything about SAP transport risks…")
            if query_chat:
                st.session_state.chat_history.append({"role":"user","content":query_chat})
                st.chat_message("user").write(query_chat)

                with st.spinner("AI thinking..."):
                    # RAG silently enriches the prompt
                    rag_prompt = rag.build_prompt(query_chat)
                    messages_with_rag = st.session_state.chat_history[:-1] + [
                        {"role":"user","content":rag_prompt}
                    ]
                    reply = ask_ai_with_history(messages_with_rag)

                st.session_state.chat_history.append({"role":"assistant","content":reply})
                st.chat_message("assistant").write(reply)

                try:
                    save_chat_history(st.session_state.session_id, st.session_state.chat_history)
                except:
                    pass

            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = [{
                    "role":"system",
                    "content":"You are an expert SAP Basis consultant."
                }]
                st.rerun()

    # ============================================================
    # TAB 5 — MANUAL PREDICTION 
    # ============================================================
    with tab5:
        st.subheader("⚡ Real-Time Risk Prediction")

        c1,c2 = st.columns(2)
        with c1:
            module    = st.selectbox("Module",           ["FI","MM","SD","HR"])
            objects   = st.number_input("Objects Changed", 1, 10, 3)
            lines     = st.number_input("Lines Changed",   1, 500, 100)
            conflicts = st.number_input("Conflicts",       0, 5, 1)
        with c2:
            failures  = st.number_input("Failures",        0, 5, 1)
            stage     = st.selectbox("Stage",             ["Development","Quality","Production"])
            status    = st.selectbox("Status",            ["Approved","Pending","Rejected"])

        if st.button("🔮 Predict Risk"):
            new_df = pd.DataFrame([{
                "module":module,"objects_changed":objects,
                "lines_changed":lines,"conflicts":conflicts,
                "history_failures":failures,
                "transport_stage":stage,"change_request_status":status
            }])
            processed = preprocess(new_df.copy())
            pred      = predict(model, processed)[0]
            conf, cls, proba_vals = get_confidence(processed)
            confidence = round(float(conf[0]),2) if conf is not None else None

            rc = cmap.get(pred,"#aaa")
            st.markdown(f"""
            <div style='background:{rc};
                 color:{"white" if pred=="HIGH" else "black"};
                 padding:20px;border-radius:15px;text-align:center;
                 font-size:1.8em;font-weight:bold;margin:10px 0'>
                {pred} RISK
                {f"— {confidence}% Confident" if confidence else ""}
            </div>""", unsafe_allow_html=True)

            if confidence and cls is not None:
                prob_df = pd.DataFrame({
                    'Risk Level': cls,
                    'Probability %': (proba_vals[0]*100).round(2)
                })
                fig_pb = px.bar(
                    prob_df, x='Risk Level', y='Probability %',
                    title="🎯 Prediction Probability Breakdown",
                    color='Risk Level', color_discrete_map=cmap
                )
                st.plotly_chart(fig_pb, use_container_width=True)

            pct   = 90 if pred=="HIGH" else 60 if pred=="MEDIUM" else 30
            color = "#ff4d4d" if pred=="HIGH" else "#f1c40f" if pred=="MEDIUM" else "#2ecc71"
            st.markdown(risk_gauge(pct, color), unsafe_allow_html=True)

            # SHAP bar
            if shap_explainer is not None:
                try:
                    shap_vals = shap_summary_for_row(shap_explainer, processed, pred)
                    feat_names = list(shap_vals.keys())
                    vals       = list(shap_vals.values())
                    colors_shap = ["#ff4d4d" if v > 0 else "#2ecc71" for v in vals]
                    fig_shap = go.Figure(go.Bar(
                        x=vals, y=feat_names, orientation='h',
                        marker_color=colors_shap
                    ))
                    fig_shap.update_layout(
                        title="SHAP Feature Impact (red=increases risk)",
                        xaxis_title="SHAP value"
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except:
                    pass

            fig_inp = px.bar(
                x=["Objects","Lines","Conflicts","Failures"],
                y=[objects,lines,conflicts,failures],
                title="Input Feature Values",
                color_discrete_sequence=[rc]
            )
            st.plotly_chart(fig_inp, use_container_width=True)

            if stage=="Production" and pred=="HIGH":
                st.error("🚨 DO NOT DEPLOY — HIGH-RISK PRODUCTION TRANSPORT!")

            # Save to Firebase
            pred_data = {
                "transport_id": f"MANUAL_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "module": module, "objects_changed": objects,
                "lines_changed": lines, "conflicts": conflicts,
                "history_failures": failures, "transport_stage": stage,
                "change_request_status": status, "predicted_risk": pred,
                "confidence": float(confidence) if confidence else None,
                "timestamp": datetime.datetime.now().isoformat()
            }
            try:
                save_prediction(pred_data)
                st.success("✅ Prediction saved to Firebase!")
            except Exception as e:
                st.warning(f"⚠ Could not save to Firebase: {e}")

            # AI explanation auto-generates (no nested button)
            st.markdown("### 🤖 AI Explanation")
            with st.spinner("AI analyzing prediction..."):
                prompt_exp = f"""
                SAP transport predicted as {pred} risk ({confidence}% confidence).
                Module={module}, Objects Changed={objects}, Lines Changed={lines},
                Conflicts={conflicts}, History Failures={failures},
                Stage={stage}, Status={status}.
                Explain:
                1. Why this specific risk level was predicted
                2. Business impact if deployed
                3. Top 3 recommended actions
                """
                st.write(ask_ai(prompt_exp))

    # ============================================================
    # TAB 6 — WHAT-IF ANALYSIS
    # ============================================================
    with tab6:
        st.subheader("🔄 What-If Analysis")
        st.caption("Compare two scenarios to measure impact of changes")

        st.markdown("### 📌 Scenario A — Current State")
        a1,a2 = st.columns(2)
        with a1:
            a_mod  = st.selectbox("Module (A)",   ["FI","MM","SD","HR"], key="am")
            a_obj  = st.number_input("Objects (A)", 1,10,5,              key="ao")
            a_lin  = st.number_input("Lines (A)",   1,500,200,           key="al")
            a_con  = st.number_input("Conflicts (A)",0,5,3,              key="ac")
        with a2:
            a_fail = st.number_input("Failures (A)",0,5,2,               key="af")
            a_stg  = st.selectbox("Stage (A)",["Development","Quality","Production"], key="as")
            a_stat = st.selectbox("Status (A)",["Approved","Pending","Rejected"],     key="ast")

        st.markdown("### 🎯 Scenario B — Improved State")
        b1,b2 = st.columns(2)
        with b1:
            b_mod  = st.selectbox("Module (B)",   ["FI","MM","SD","HR"], key="bm")
            b_obj  = st.number_input("Objects (B)", 1,10,2,              key="bo")
            b_lin  = st.number_input("Lines (B)",   1,500,50,            key="bl")
            b_con  = st.number_input("Conflicts (B)",0,5,0,              key="bc")
        with b2:
            b_fail = st.number_input("Failures (B)",0,5,0,               key="bf")
            b_stg  = st.selectbox("Stage (B)",["Development","Quality","Production"], key="bs")
            b_stat = st.selectbox("Status (B)",["Approved","Pending","Rejected"],     key="bst")

        if st.button("🔄 Compare Scenarios"):
            df_a = pd.DataFrame([{"module":a_mod,"objects_changed":a_obj,
                "lines_changed":a_lin,"conflicts":a_con,"history_failures":a_fail,
                "transport_stage":a_stg,"change_request_status":a_stat}])
            df_b = pd.DataFrame([{"module":b_mod,"objects_changed":b_obj,
                "lines_changed":b_lin,"conflicts":b_con,"history_failures":b_fail,
                "transport_stage":b_stg,"change_request_status":b_stat}])

            pred_a = predict(model, preprocess(df_a.copy()))[0]
            pred_b = predict(model, preprocess(df_b.copy()))[0]
            ca,_,_ = get_confidence(preprocess(df_a.copy()))
            cb,_,_ = get_confidence(preprocess(df_b.copy()))
            conf_a = round(float(ca[0]),2) if ca is not None else "N/A"
            conf_b = round(float(cb[0]),2) if cb is not None else "N/A"

            col1,col2 = st.columns(2)
            with col1:
                st.markdown("#### 📌 Scenario A")
                rc_a = cmap.get(pred_a,"#aaa")
                st.markdown(f"""
                <div style='background:{rc_a};
                     color:{"white" if pred_a=="HIGH" else "black"};
                     padding:20px;border-radius:12px;text-align:center;
                     font-size:1.5em;font-weight:bold'>
                     {pred_a} RISK<br>
                     <span style='font-size:0.6em'>Confidence: {conf_a}%</span>
                </div>""", unsafe_allow_html=True)
                pct_a = 90 if pred_a=="HIGH" else 60 if pred_a=="MEDIUM" else 30
                st.markdown(risk_gauge(pct_a, rc_a), unsafe_allow_html=True)

            with col2:
                st.markdown("#### 🎯 Scenario B")
                rc_b = cmap.get(pred_b,"#aaa")
                st.markdown(f"""
                <div style='background:{rc_b};
                     color:{"white" if pred_b=="HIGH" else "black"};
                     padding:20px;border-radius:12px;text-align:center;
                     font-size:1.5em;font-weight:bold'>
                     {pred_b} RISK<br>
                     <span style='font-size:0.6em'>Confidence: {conf_b}%</span>
                </div>""", unsafe_allow_html=True)
                pct_b = 90 if pred_b=="HIGH" else 60 if pred_b=="MEDIUM" else 30
                st.markdown(risk_gauge(pct_b, rc_b), unsafe_allow_html=True)

            risk_order = {"HIGH":3,"MEDIUM":2,"LOW":1}
            delta = risk_order.get(pred_a,2) - risk_order.get(pred_b,2)
            if delta > 0:
                st.success(f"✅ Risk REDUCED from {pred_a} → {pred_b}")
            elif delta < 0:
                st.error(f"⚠ Risk INCREASED from {pred_a} → {pred_b}")
            else:
                st.info(f"➡ Risk unchanged — both predict {pred_a}")

            comp_df = pd.DataFrame({
                'Parameter':  ['Objects','Lines','Conflicts','Failures'],
                'Scenario A': [a_obj,a_lin,a_con,a_fail],
                'Scenario B': [b_obj,b_lin,b_con,b_fail]
            })
            fig_cmp = px.bar(
                comp_df.melt(id_vars='Parameter',var_name='Scenario',value_name='Value'),
                x='Parameter', y='Value', color='Scenario', barmode='group',
                title="Parameter Comparison A vs B",
                color_discrete_map={"Scenario A":"#ff4d4d","Scenario B":"#2ecc71"}
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            with st.spinner("AI comparing scenarios..."):
                prompt = f"""
                Compare two SAP transport scenarios:
                A: Module={a_mod}, Objects={a_obj}, Lines={a_lin},
                Conflicts={a_con}, Failures={a_fail}, Stage={a_stg}
                → Predicted: {pred_a} ({conf_a}% confidence)
                B: Module={b_mod}, Objects={b_obj}, Lines={b_lin},
                Conflicts={b_con}, Failures={b_fail}, Stage={b_stg}
                → Predicted: {pred_b} ({conf_b}% confidence)
                1. Why did risk change? 2. Which parameter had most impact?
                3. Is Scenario B safe? 4. Further improvements?
                """
                st.write(ask_ai(prompt))

    # ============================================================
    # TAB 7 — PREDICTION HISTORY
    # ============================================================
    with tab7:
        st.subheader("📜 Prediction History")
        st.caption("All manual predictions saved to Firebase Firestore")

        if st.button("🔄 Refresh from Firebase"):
            st.rerun()

        try:
            history_df = fetch_predictions()
            if not history_df.empty:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Predictions", len(history_df))
                c2.metric("HIGH Risk",
                    (history_df['predicted_risk']=="HIGH").sum()
                    if 'predicted_risk' in history_df.columns else 0)
                c3.metric("MEDIUM Risk",
                    (history_df['predicted_risk']=="MEDIUM").sum()
                    if 'predicted_risk' in history_df.columns else 0)
                c4.metric("LOW Risk",
                    (history_df['predicted_risk']=="LOW").sum()
                    if 'predicted_risk' in history_df.columns else 0)

                if 'predicted_risk' in history_df.columns:
                    def highlight_pred(row):
                        risk = row.get('predicted_risk', '')
                        if risk == 'HIGH':
                            return ['background-color:#ff4d4d;color:white']*len(row)
                        elif risk == 'MEDIUM':
                            return ['background-color:#fff3cd;color:black']*len(row)
                        elif risk == 'LOW':
                            return ['background-color:#d4edda;color:black']*len(row)
                        return ['']*len(row)

                    st.dataframe(
                        history_df.style.apply(highlight_pred, axis=1),
                        use_container_width=True
                    )
                    fig_h = px.bar(
                        history_df['predicted_risk'].value_counts().reset_index(),
                        x='predicted_risk', y='count',
                        title="Prediction History Distribution",
                        color='predicted_risk', color_discrete_map=cmap
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

                st.download_button(
                    "📥 Export Prediction History",
                    history_df.to_csv(index=False),
                    "prediction_history.csv"
                )

                if st.button("🤖 AI Analyze Prediction History"):
                    if 'predicted_risk' in history_df.columns:
                        prompt = f"""
                        Analyze SAP transport prediction history:
                        Total: {len(history_df)},
                        HIGH: {(history_df['predicted_risk']=="HIGH").sum()},
                        MEDIUM: {(history_df['predicted_risk']=="MEDIUM").sum()},
                        LOW: {(history_df['predicted_risk']=="LOW").sum()}
                        Identify patterns, trends, and risk insights.
                        """
                        with st.spinner("AI analyzing history..."):
                            st.write(ask_ai(prompt))
            else:
                st.info("No prediction history yet. Use Manual Prediction tab.")
        except Exception as e:
            st.error(f"Firebase fetch error: {e}")

    # ============================================================
    # TAB 8 — BUSINESS IMPACT & ROI 
    # ============================================================
    with tab8:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0f0c29,#302b63);
             padding:25px;border-radius:15px;margin-bottom:20px;
             border:1px solid #764ba2'>
            <h2 style='color:white;margin:0'>💰 Business Impact & ROI Calculator</h2>
            <p style='color:#b39ddb;margin-top:8px'>
                Quantify the financial value of AI-powered SAP transport risk prevention
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### 💸 Cost Parameters")
            cost_per_incident    = st.number_input(
                "Average cost per SAP incident ($)", 10000, 10000000, 50000, 10000)
            downtime_hours       = st.number_input(
                "Average downtime per incident (hours)", 1, 72, 4)
            hourly_business_loss = st.number_input(
                "Business loss per hour ($)", 1000, 1000000, 10000, 5000)
        with col2:
            st.markdown("#### 📊 Prevention Parameters")
            prevention_rate     = st.slider("AI Prevention Rate (%)", 50, 95, 60)
            transports_per_year = st.number_input(
                "Transports deployed per year", 100, 10000000, 1000, 100)
            implementation_cost = st.number_input(
                "Platform implementation cost ($)", 0, 5000000, 200000, 50000)

        st.divider()

        high_n_roi  = (filtered_df['Predicted Risk']=="HIGH").sum()
        med_n_roi   = (filtered_df['Predicted Risk']=="MEDIUM").sum()
        low_n_roi   = (filtered_df['Predicted Risk']=="LOW").sum()
        total_n_roi = len(filtered_df)

        high_risk_rate      = high_n_roi / total_n_roi if total_n_roi > 0 else 0
        annual_high_risk    = int(transports_per_year * high_risk_rate)
        incidents_prevented = int(annual_high_risk * prevention_rate / 100)
        downtime_prevented  = incidents_prevented * downtime_hours
        direct_savings      = incidents_prevented * cost_per_incident
        downtime_savings    = downtime_prevented * hourly_business_loss
        total_savings       = direct_savings + downtime_savings
        net_roi             = total_savings - implementation_cost
        roi_percentage      = ((net_roi / implementation_cost) * 100) if implementation_cost > 0 else 0
        payback_days        = int((implementation_cost / (total_savings / 365))) if total_savings > 0 else 0

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("💰 Total Annual Savings",  f"${total_savings:,.0f}", delta="Per Year")
        r2.metric("📈 ROI Percentage",         f"{roi_percentage:,.0f}%")
        r3.metric("⚡ Payback Period",          f"{payback_days} days")
        r4.metric("🛡 Incidents Prevented",    f"{incidents_prevented:,}")

        st.divider()
        col1,col2 = st.columns(2)
        with col1:
            savings_data = {
                'Category': ['Direct Incident Savings','Downtime Prevention','Implementation Cost'],
                'Amount ($)': [direct_savings, downtime_savings, -implementation_cost]
            }
            fig_roi = px.bar(savings_data, x='Category', y='Amount ($)',
                             title="Annual Savings vs Cost Breakdown",
                             color='Category',
                             color_discrete_sequence=['#2ecc71','#3498db','#e74c3c'])
            st.plotly_chart(fig_roi, use_container_width=True)
        with col2:
            impact_data = {
                'Risk Type': ['HIGH Risk','MEDIUM Risk','LOW Risk'],
                'Count':     [high_n_roi, med_n_roi, low_n_roi]
            }
            fig_impact = px.pie(impact_data, values='Count', names='Risk Type',
                                title="Current Risk Distribution",
                                color='Risk Type',
                                color_discrete_map={
                                    'HIGH Risk':'#ff4d4d',
                                    'MEDIUM Risk':'#f1c40f',
                                    'LOW Risk':'#2ecc71'
                                })
            st.plotly_chart(fig_impact, use_container_width=True)

        st.divider()
        st.markdown("### 🏢 Business Impact Summary")
        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #2ecc71;
                 border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2em'>🛡️</div>
                <h3 style='color:#2ecc71;margin:8px 0'>{incidents_prevented:,}</h3>
                <p style='color:#aaa;margin:0'>Production incidents prevented annually</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #3498db;
                 border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2em'>⏱️</div>
                <h3 style='color:#3498db;margin:8px 0'>{downtime_prevented:,} hrs</h3>
                <p style='color:#aaa;margin:0'>Downtime prevented annually</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #f1c40f;
                 border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2em'>💰</div>
                <h3 style='color:#f1c40f;margin:8px 0'>${net_roi:,.0f}</h3>
                <p style='color:#aaa;margin:0'>Net annual savings after costs</p>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📋 SLA Compliance Analysis")
        sla_breach = filtered_df[
            (filtered_df['Predicted Risk']=="HIGH") &
            (filtered_df['transport_stage']=="Production")
        ]
        compliance_score = round(
            (1 - len(sla_breach)/total_n_roi)*100, 1
        ) if total_n_roi > 0 else 100

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("📋 SLA Compliance Score",  f"{compliance_score}%")
        s2.metric("🚨 SLA Breach Risk",        len(sla_breach))
        s3.metric("✅ Compliant Transports",   total_n_roi - len(sla_breach))
        s4.metric("🎯 Target Compliance",       "95%",
                  delta=f"{round(compliance_score-95,1)}%")

        comp_color = "#2ecc71" if compliance_score >= 95 \
                     else "#f1c40f" if compliance_score >= 80 \
                     else "#ff4d4d"

        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid {comp_color};
             border-radius:12px;padding:20px;margin:15px 0'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
                <div>
                    <h3 style='color:white;margin:0'>SLA Compliance Status</h3>
                    <p style='color:#aaa;margin:5px 0'>
                        Based on HIGH risk transports in Production
                    </p>
                </div>
                <div style='font-size:2.5em;font-weight:bold;color:{comp_color}'>
                    {compliance_score}%
                </div>
            </div>
            <div style='background:#333;border-radius:10px;height:20px;margin-top:15px'>
                <div style='background:{comp_color};width:{min(compliance_score,100)}%;
                     height:20px;border-radius:10px'>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📦 Module-wise Business Impact")
        module_impact = []
        for mod in filtered_df['module'].unique():
            mdf       = filtered_df[filtered_df['module']==mod]
            mod_high  = (mdf['Predicted Risk']=="HIGH").sum()
            mod_total = len(mdf)
            mod_risk  = round(mod_high/mod_total*100,1) if mod_total>0 else 0
            mod_cost  = int(mod_high * prevention_rate/100 * cost_per_incident)
            module_impact.append({
                'Module': mod, 'Total': mod_total,
                'HIGH Risk': mod_high, 'Risk %': mod_risk,
                'Potential Savings ($)': mod_cost
            })
        impact_df = pd.DataFrame(module_impact).sort_values(
            'Potential Savings ($)', ascending=False
        )
        st.dataframe(impact_df, use_container_width=True)

        fig_mod = px.bar(
            impact_df, x='Module', y='Potential Savings ($)',
            title="Potential Savings by Module",
            color='Risk %',
            color_continuous_scale=['green','yellow','red']
        )
        st.plotly_chart(fig_mod, use_container_width=True)

        st.divider()
        if st.button("📊 Generate AI Business Impact Summary"):
            prompt = f"""
            Generate a professional business impact report for SAP Transport Risk AI Platform:
            Total Transports: {total_n_roi:,}, HIGH Risk: {high_n_roi:,},
            Critical (Production+HIGH): {len(sla_breach):,},
            SLA Compliance: {compliance_score}%
            Annual Incidents Prevented: {incidents_prevented:,},
            Total Annual Savings: ${total_savings:,.0f},
            ROI: {roi_percentage:,.0f}%, Payback: {payback_days} days,
            Downtime Prevented: {downtime_prevented:,} hours/year
            Module Risk: {impact_df[['Module','HIGH Risk','Risk %']].to_string()}
            Write: Executive Summary, Key Risk Findings, Financial Benefits,
            Strategic Recommendations, Conclusion with ROI justification.
            """
            with st.spinner("AI generating business report..."):
                report = ask_ai(prompt)
            st.write(report)
            st.session_state.ai_summary = report
            pdf_buf = generate_pdf(
                filtered_df, report,
                f"ROI Analysis — {datetime.datetime.now().strftime('%Y-%m-%d')}"
            )
            st.download_button(
                "📥 Download Business Impact PDF",
                data=pdf_buf,
                file_name="sap_business_impact_report.pdf",
                mime="application/pdf"
            )

    # ============================================================
    # TAB 9 — SHAP EXPLAINABILITY 
    # ============================================================
    with tab9:
        st.subheader("🔬 SHAP Explainability")
        st.caption("SHapley Additive exPlanations — understand what drives each prediction")

        if shap_explainer is None:
            st.warning("shap_explainer.pkl not found. Re-run train_model.py.")
        elif filtered_df.empty:
            st.info("No data after filters.")
        else:
            sample_size = min(100, len(filtered_df))
            sample_df   = filtered_df.head(sample_size).copy()
            proc_sample = preprocess(sample_df)

            st.markdown(f"Explaining {sample_size} records using **CatBoost** (best tree model)...")
            try:
                shap_vals, feat_names = get_shap_values(shap_explainer, proc_sample)

                if isinstance(shap_vals, list):
                    mean_abs = np.abs(shap_vals[2]).mean(axis=0)
                else:
                    mean_abs = np.abs(shap_vals[:,:,2]).mean(axis=0) \
                               if shap_vals.ndim==3 else np.abs(shap_vals).mean(axis=0)

                importance_df = pd.DataFrame({
                    "Feature":     feat_names,
                    "Mean |SHAP|": mean_abs
                }).sort_values("Mean |SHAP|", ascending=True)

                fig = px.bar(importance_df, x="Mean |SHAP|", y="Feature",
                             orientation='h',
                             title="Global Feature Importance (HIGH risk class)",
                             color="Mean |SHAP|",
                             color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Single Record Explanation (first filtered row)")
                row0      = filtered_df.iloc[[0]].copy()
                proc_row0 = preprocess(row0)
                shap_row  = shap_summary_for_row(
                    shap_explainer, proc_row0,
                    filtered_df.iloc[0]['Predicted Risk']
                )

                feat_sorted = sorted(shap_row.items(), key=lambda x: x[1])
                fnames  = [f[0] for f in feat_sorted]
                fvals   = [f[1] for f in feat_sorted]
                fcolors = ["#ff4d4d" if v > 0 else "#2ecc71" for v in fvals]

                wfig = go.Figure(go.Bar(
                    x=fvals, y=fnames, orientation='h', marker_color=fcolors
                ))
                wfig.update_layout(
                    title=f"Feature contributions → {filtered_df.iloc[0]['Predicted Risk']} risk",
                    xaxis_title="SHAP value (red=pushes toward HIGH, green=away)"
                )
                st.plotly_chart(wfig, use_container_width=True)

                st.info("""
                **How to read SHAP:**
                - 🔴 Red bars → feature increases risk prediction
                - 🟢 Green bars → feature decreases risk prediction
                - Longer bar = stronger influence on prediction
                """)
            except Exception as e:
                st.error(f"SHAP error: {e}")

    # ============================================================
    # TAB 10 — MODEL COMPARISON 
    # ============================================================
    with tab10:
        st.subheader("📈 Model Comparison")
        st.caption("5 Models trained with SMOTE balancing + MLflow tracking")

        if model_comparison is None:
            st.warning("model_comparison.pkl not found. Re-run train_model.py.")
        else:
            cdf = pd.DataFrame([
                {"Model": k, "F1 (test)": v["f1"], "CV F1": v["cv_f1"]}
                for k, v in model_comparison.items()
            ]).sort_values("F1 (test)", ascending=False).reset_index(drop=True)
            cdf["Rank"] = cdf.index + 1

            st.dataframe(cdf, use_container_width=True)

            fig = px.bar(cdf, x="Model", y=["F1 (test)","CV F1"],
                         barmode="group",
                         title="Model Performance Comparison (with SMOTE)",
                         color_discrete_map={"F1 (test)":"#4e8cff","CV F1":"#f1c40f"})
            st.plotly_chart(fig, use_container_width=True)

            best = cdf.iloc[0]
            st.success(f"🏆 Best Model: **{best['Model']}**  —  "
                       f"F1={best['F1 (test)']:.4f}  CV-F1={best['CV F1']:.4f}")

            st.markdown(f"""
            **What each model brings:**
            - **XGBoost** — gradient boosting king, industry standard
            - **LightGBM** — Microsoft's fast enterprise booster
            - **CatBoost** — native categorical support, perfect for SAP modules
            - **RandomForest** — classical ensemble, best SHAP explainability
            - **TabNet** — attention-based deep learning for tabular data

            All models trained on **SMOTE-balanced** data tracked with **MLflow**.
            Best model: **{best_model_name}** (auto-selected by F1 + CV-F1 score) | SHAP explanations: **{shap_model_name}**
            """)

# ================================================================
# EMPTY STATE
# ================================================================
else:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;
         background:linear-gradient(135deg,#1a1a2e,#16213e);
         border-radius:20px;border:1px dashed #764ba2;margin-top:20px'>
        <div style='font-size:4em'>📂</div>
        <h2 style='color:white;margin-top:15px'>
            Upload CSV or Load from Firebase
        </h2>
        <p style='color:#aaa;font-size:1.1em'>
            Choose a data source from the sidebar to begin
        </p>
        <div style='margin-top:20px;display:flex;
             justify-content:center;gap:30px;flex-wrap:wrap'>
            <div style='color:#764ba2'>
                <div style='font-size:2em'>📁</div>
                <div>Upload CSV</div>
            </div>
            <div style='color:#4caf50'>
                <div style='font-size:2em'>🔥</div>
                <div>Load from Firebase</div>
            </div>
        </div>
        <p style='color:#555;margin-top:25px;font-size:0.9em'>
            Required columns: transport_id, module, transport_stage,
            objects_changed, lines_changed, conflicts,
            history_failures, change_request_status
        </p>
    </div>
    """, unsafe_allow_html=True)

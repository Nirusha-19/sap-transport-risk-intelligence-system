import os, base64, json, tempfile

# Handle Firebase credentials for HuggingFace deployment
firebase_key_b64 = os.getenv("FIREBASE_KEY_BASE64")
if firebase_key_b64:
    key_data = json.loads(base64.b64decode(firebase_key_b64))
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as f:
        json.dump(key_data, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import datetime
import streamlit as st

# ---------------- INIT FIREBASE ----------------
def init_firebase():
    if not firebase_admin._apps:
        try:
            # Streamlit Cloud
            firebase_credentials = dict(st.secrets["firebase"])
            if "private_key" in firebase_credentials:
                firebase_credentials["private_key"] = (
                    firebase_credentials["private_key"]
                    .replace("\\n", "\n")
                )
            cred = credentials.Certificate(firebase_credentials)
        except Exception:
            # Local development or HuggingFace fallback
            firebase_path = os.getenv(
                "FIREBASE_KEY_PATH",
                "firebase_key.json"
            )
            # If GOOGLE_APPLICATION_CREDENTIALS is set use it
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                cred = credentials.Certificate(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                )
            else:
                cred = credentials.Certificate(firebase_path)

        firebase_admin.initialize_app(cred)

    return firestore.client()

# ---------------- UPLOAD CSV TO FIRESTORE ----------------
def upload_dataframe_to_firestore(df, collection="transports"):
    db = init_firebase()
    col_ref = db.collection(collection)

    existing = col_ref.get()
    for doc in existing:
        doc.reference.delete()

    batch = db.batch()
    count = 0

    for _, row in df.iterrows():
        doc_ref = col_ref.document(str(row['transport_id']))
        row_dict = {}
        for k, v in row.to_dict().items():
            if hasattr(v, 'item'):
                row_dict[k] = v.item()
            else:
                row_dict[k] = v
        batch.set(doc_ref, row_dict)
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()
    return count

# ---------------- FETCH FROM FIRESTORE ----------------
def fetch_from_firestore(collection="transports"):
    db = init_firebase()
    docs = db.collection(collection).get()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame([doc.to_dict() for doc in docs])

    module_mapping = {
        0: 'FI', 1: 'MM', 2: 'SD', 3: 'HR',
        '0': 'FI', '1': 'MM', '2': 'SD', '3': 'HR'
    }
    stage_mapping = {
        0: 'Development', 1: 'Quality', 2: 'Production',
        '0': 'Development', '1': 'Quality', '2': 'Production'
    }
    status_mapping = {
        0: 'Approved', 1: 'Pending', 2: 'Rejected',
        '0': 'Approved', '1': 'Pending', '2': 'Rejected'
    }

    if 'module' in df.columns:
        df['module'] = df['module'].replace(module_mapping)
    if 'transport_stage' in df.columns:
        df['transport_stage'] = df['transport_stage'].replace(stage_mapping)
    if 'change_request_status' in df.columns:
        df['change_request_status'] = df['change_request_status'].replace(status_mapping)

    return df

# ---------------- SAVE PREDICTION ----------------
def save_prediction(data: dict, collection="predictions"):
    db = init_firebase()
    clean = {}
    for k, v in data.items():
        if hasattr(v, 'item'):
            clean[k] = v.item()
        else:
            clean[k] = v
    db.collection(collection).add(clean)

# ---------------- FETCH PREDICTIONS ----------------
def fetch_predictions(collection="predictions"):
    db = init_firebase()
    docs = db.collection(collection).order_by(
        "timestamp",
        direction=firestore.Query.DESCENDING
    ).get()

    if not docs:
        return pd.DataFrame()

    return pd.DataFrame([doc.to_dict() for doc in docs])

# ---------------- SAVE AI INSIGHT ----------------
def save_ai_insight(transport_id, insight, collection="ai_insights"):
    db = init_firebase()
    db.collection(collection).document(str(transport_id)).set({
        "transport_id": str(transport_id),
        "insight": insight,
        "timestamp": datetime.datetime.now().isoformat()
    })

# ---------------- FETCH AI INSIGHTS ----------------
def fetch_ai_insights(collection="ai_insights"):
    db = init_firebase()
    docs = db.collection(collection).get()
    if not docs:
        return {}
    return {doc.id: doc.to_dict() for doc in docs}

# ---------------- SAVE CHAT HISTORY ----------------
def save_chat_history(session_id, messages, collection="chat_history"):
    db = init_firebase()
    db.collection(collection).document(session_id).set({
        "session_id": session_id,
        "messages": messages,
        "timestamp": datetime.datetime.now().isoformat()
    })

# ---------------- FETCH CHAT HISTORY ----------------
def fetch_chat_history(session_id, collection="chat_history"):
    db = init_firebase()
    doc = db.collection(collection).document(session_id).get()
    if doc.exists:
        return doc.to_dict().get("messages", [])
    return []

# ---------------- DELETE ALL IN COLLECTION ----------------
def clear_collection(collection):
    db = init_firebase()
    docs = db.collection(collection).get()
    for doc in docs:
        doc.reference.delete()
    return len(docs)
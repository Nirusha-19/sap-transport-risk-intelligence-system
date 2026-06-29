# ADDED: get_shap_values() and shap_summary_for_row()
# CHANGED: preprocess now returns a copy (safe for chaining)

import pickle
import pandas as pd
import numpy as np

FEATURES = ['module','objects_changed','lines_changed',
            'conflicts','history_failures',
            'transport_stage','change_request_status']

module_map  = {'FI': 0, 'MM': 1, 'SD': 2, 'HR': 3}
stage_map   = {'Development': 0, 'Quality': 1, 'Production': 2}
status_map  = {'Approved': 0, 'Pending': 1, 'Rejected': 2}
risk_map    = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()                                  # CHANGED: was mutating in place
    df['module']                = df['module'].map(module_map)
    df['transport_stage']       = df['transport_stage'].map(stage_map)
    df['change_request_status'] = df['change_request_status'].map(status_map)
    return df


def predict(model, df: pd.DataFrame):
    X = df[FEATURES].values
    preds = model.predict(X)
    if hasattr(preds, 'flatten'):
        preds = preds.flatten()
    return [risk_map[int(p)] for p in preds]


def get_shap_values(explainer, df_processed: pd.DataFrame):
    """Return raw SHAP values array + feature names."""
    X = df_processed[FEATURES].values
    shap_vals = explainer.shap_values(X)
    return shap_vals, FEATURES


def shap_summary_for_row(explainer, row_processed: pd.DataFrame,
                          predicted_class: str) -> dict:
    class_idx = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}[predicted_class]
    X = row_processed[FEATURES].values
    shap_vals = explainer.shap_values(X)

    try:
        if isinstance(shap_vals, list):
            vals = shap_vals[class_idx][0]
        elif shap_vals.ndim == 3:
            vals = shap_vals[0, :, class_idx]
        else:
            vals = shap_vals[0]
    except Exception:
        vals = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[0][0]

    return dict(zip(FEATURES, vals))
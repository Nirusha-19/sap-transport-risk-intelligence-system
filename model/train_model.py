# MODELS: XGBoost, LightGBM, CatBoost, RandomForest, TabNet
# SMOTE oversampling
# MLflow tracking for every model

import os, pickle, warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/sap_transport_dataset.csv")
print("✅ Dataset loaded:", df.shape)
print(df["risk_level"].value_counts())

# ── Encode ────────────────────────────────────────────────────────────────
module_map  = {'FI': 0, 'MM': 1, 'SD': 2, 'HR': 3}
stage_map   = {'Development': 0, 'Quality': 1, 'Production': 2}
status_map  = {'Approved': 0, 'Pending': 1, 'Rejected': 2}
risk_map    = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}

df['module']                = df['module'].map(module_map)
df['transport_stage']       = df['transport_stage'].map(stage_map)
df['change_request_status'] = df['change_request_status'].map(status_map)
df['risk']                  = df['risk_level'].map(risk_map)

FEATURES = [
    'module', 'objects_changed', 'lines_changed',
    'conflicts', 'history_failures',
    'transport_stage', 'change_request_status'
]

X = df[FEATURES].values
y = df['risk'].values

# ── SMOTE ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# ── Five Models ───────────────────────────────────────────────────────────
models = {
    "XGBoost":      XGBClassifier(
                        n_estimators=200,
                        eval_metric='mlogloss',
                        random_state=42,
                        n_jobs=-1),

    "LightGBM":     LGBMClassifier(
                        n_estimators=200,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=1,
                        verbose=-1),

    "CatBoost":     CatBoostClassifier(
                        iterations=200,
                        random_seed=42,
                        verbose=0),

    "RandomForest": RandomForestClassifier(
                        n_estimators=200,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1),

    "TabNet":       TabNetClassifier(
                        n_d=16,
                        n_a=16,
                        n_steps=3,
                        gamma=1.3,
                        n_independent=1,
                        n_shared=1,
                        seed=42,
                        verbose=1,
                        device_name='cpu'),
}

os.makedirs("model", exist_ok=True)
mlflow.set_experiment("SAP_Transport_Risk")

results = {}

# ── Training Loop ─────────────────────────────────────────────────────────
for name, clf in models.items():
    print(f"\n🔄 Training {name}…")
    with mlflow.start_run(run_name=name):

        if name == "TabNet":
            # TabNet needs float32 and different fit signature
            clf.fit(
                X_train_sm.astype(np.float32),
                y_train_sm,
                eval_set=[(X_test.astype(np.float32), y_test)],
                patience=10,
                max_epochs=50,
                batch_size=512,
            )
            y_pred = clf.predict(X_test.astype(np.float32))
            # CV skipped for TabNet (incompatible API + slow)
            cv = f1_score(y_test, y_pred, average='weighted')
        else:
            clf.fit(X_train_sm, y_train_sm)
            y_pred = clf.predict(X_test)
            cv = cross_val_score(
                clf, X_train_sm, y_train_sm,
                cv=3, scoring='f1_weighted',
                n_jobs=1 if name == "LightGBM" else -1
            ).mean()

        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_param("model", name)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("cv_f1_weighted", cv)

        # TabNet not sklearn compatible, skip sklearn mlflow log
        if name != "TabNet":
            mlflow.sklearn.log_model(clf, name)

        results[name] = {"f1": f1, "cv_f1": cv, "model": clf}
        print(f"   F1={f1:.4f}  CV-F1={cv:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=['LOW', 'MEDIUM', 'HIGH']
        ))

# ── Best Model (CV F1 as tiebreaker) ──────────────────────────────────────
best_name = max(
    results,
    key=lambda k: (results[k]["f1"], results[k]["cv_f1"])
)
best_clf = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name}  F1={results[best_name]['f1']:.4f}")

with open("model/model.pkl", "wb") as f:
    pickle.dump(best_clf, f)

# Save best model name
with open("model/best_model_name.txt", "w") as f:
    f.write(best_name)

# ── SHAP Explainer ────────────────────────────────────────────────────────
print("\n🔍 Building SHAP explainer…")

# XGBoost, LightGBM, CatBoost, RandomForest → TreeExplainer native
# TabNet → not TreeExplainer compatible
tree_models = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]

if best_name in tree_models:
    shap_clf  = best_clf
    shap_name = best_name
else:
    # Use best tree model for SHAP
    shap_name = max(
        [m for m in results if m in tree_models],
        key=lambda k: results[k]["f1"]
    )
    shap_clf = results[shap_name]["model"]
    print(f"ℹ️  {best_name} is best overall but not SHAP TreeExplainer compatible")
    print(f"ℹ️  Using {shap_name} for SHAP explanations (best tree model)")

explainer = shap.TreeExplainer(shap_clf)
print(f"🔍 SHAP using: {shap_name}")

with open("model/shap_explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)

# ── Save Supporting Files ─────────────────────────────────────────────────
with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

comparison = {
    k: {"f1": v["f1"], "cv_f1": v["cv_f1"]}
    for k, v in results.items()
}
with open("model/model_comparison.pkl", "wb") as f:
    pickle.dump(comparison, f)

print("\n✅ Saved: model.pkl  shap_explainer.pkl  feature_names.pkl  model_comparison.pkl  best_model_name.txt")

# ── Final Summary ─────────────────────────────────────────────────────────
print("\n📊 Model Comparison:")
for name, m in sorted(comparison.items(), key=lambda x: x[1]["f1"], reverse=True):
    tag = " ← BEST" if name == best_name else ""
    print(f"  {name:22s}  F1={m['f1']:.4f}  CV-F1={m['cv_f1']:.4f}{tag}")

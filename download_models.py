import os
import shutil
from huggingface_hub import hf_hub_download

# Remove cached files to force fresh download
cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

files = ['model/model.pkl', 'model/shap_explainer.pkl', 'model/model_comparison.pkl', 'model/feature_names.pkl', 'data/sap_transport_dataset.csv']
for f in files:
    print(f"Downloading {f}...")
    path = hf_hub_download(repo_id='nirusha-mr/sap-transport-risk-intelligence-system', filename=f, repo_type='space', local_dir='/app', force_download=True)
    print(f"Downloaded to: {path}")

print("Files in /app/model:", os.listdir('/app/model'))

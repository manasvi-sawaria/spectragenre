"""
Run this script once to save the scaler and label encoder as pickle files.
These are needed by the Streamlit app to preprocess audio features at inference time.

Usage:
    python save_artifacts.py
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Load and combine datasets (same logic as the notebook) ──
data_3s = pd.read_csv('features_3_sec.csv')
data_3s = data_3s.drop(['filename', 'length'], axis=1)

data_30s_path = 'features_30_sec.csv'
if os.path.exists(data_30s_path):
    data_30s = pd.read_csv(data_30s_path)
    data_30s = data_30s.drop(['filename', 'length'], axis=1)
    data = pd.concat([data_3s, data_30s], ignore_index=True)
else:
    data = data_3s

X = data.drop(['label'], axis=1)
y = data['label']

# ── Fit LabelEncoder ──
le = LabelEncoder()
le.fit(y)

# ── Fit StandardScaler ──
scaler = StandardScaler()
scaler.fit(X)

# ── Save to disk ──
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("[OK] Saved label_encoder.pkl")
print("[OK] Saved scaler.pkl")
print(f"   Genre classes: {list(le.classes_)}")

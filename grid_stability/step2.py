import pickle, os
import pandas as pd
import numpy as np

# Load model and scaler
rf = pickle.load(open('models/saved/random_forest.pkl', 'rb'))
scaler = pickle.load(open('models/saved/scaler.pkl', 'rb'))

# Load UCI data
df = pd.read_csv('data/grid_stability.csv')
print('=== UCI DATASET ===')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
label_col = df.columns[-1]
print('Class distribution:')
print(df[label_col].value_counts())

# Check what features the model expects
print()
print('=== MODEL EXPECTS ===')
if hasattr(rf, 'feature_names_in_'):
    print('Feature names:', rf.feature_names_in_.tolist())
else:
    print('No feature names stored — n_features:', rf.n_features_in_)

# Check default slider values in app.py
print()
print('=== DEFAULT SLIDER VALUES ===')
with open('dashboard/app.py', 'r') as f:
    content = f.read()
import re
defaults = re.findall(r'value\s*=\s*([0-9.]+)', content)
print('Default values found in app.py:', defaults[:20])

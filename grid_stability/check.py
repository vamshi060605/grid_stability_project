import pickle, os
import pandas as pd
import numpy as np

try:
    rf = pickle.load(open('models/saved/random_forest.pkl', 'rb'))
    scaler = pickle.load(open('models/saved/scaler.pkl', 'rb'))
except Exception as e:
    print('Error loading models:', e)

try:
    df = pd.read_csv('data/grid_stability.csv')
    print('=== UCI DATASET ===')
    print('Columns:', df.columns.tolist())
    print('Shape:', df.shape)
    label_col = df.columns[-1]
    print('Class distribution:')
    print(df[label_col].value_counts())
except Exception as e:
    print('Error loading dataset:', e)

try:
    print('\n=== MODEL EXPECTS ===')
    if hasattr(rf, 'feature_names_in_'):
        print('Feature names:', rf.feature_names_in_.tolist())
    else:
        print('No feature names stored — n_features:', rf.n_features_in_)
except Exception as e:
    print('Error checking model features:', e)

try:
    print('\n=== DEFAULT SLIDER VALUES ===')
    with open('dashboard/app.py', 'r') as f:
        content = f.read()
    import re
    defaults = re.findall(r'value\s*=\s*([0-9.]+)', content)
    print('Default values found in app.py:', defaults[:20])
except Exception as e:
    print('Error checking app.py:', e)

try:
    print('\n=== TEST PREDICTION (STEP 6) ===')
    test = scaler.transform([scaler.mean_])
    pred = rf.predict(test)
    prob = rf.predict_proba(test)
    print('Prediction:', pred)
    print('Probabilities:', prob)
except Exception as e:
    print('Error running test prediction:', e)

try:
    print('\n=== FILE CHECK (STEP 3) ===')
    files = [
        'evaluation/outputs/confusion_matrices.png',
        'evaluation/outputs/roc_curves.png',
        'evaluation/outputs/learning_curves.png',
        'evaluation/outputs/feature_importance.png',
        'evaluation/outputs/robustness.png',
        'reports/project_results.pdf',
        'docs/viva_qa.md',
        'docs/architecture.md',
        'docs/demo_script.md',
        'docs/pre_demo_checklist.md',
        'demo/static_demo.html'
    ]
    for f in files:
        if os.path.exists(f):
            print(f"{f}: EXISTS ({os.path.getsize(f)} bytes)")
        else:
            print(f"{f}: MISSING")
except Exception as e:
    print('Error checking files:', e)

import pickle
import numpy as np

rf = pickle.load(open('models/saved/random_forest.pkl', 'rb'))
scaler = pickle.load(open('models/saved/scaler.pkl', 'rb'))

# Test with default slider values (whatever they currently are)
# Reconstruct what _build_feature_vector() does at defaults
# and print the prediction + confidence
print('Test prediction at default values:')
# Use the unstable base profile
test = scaler.transform([scaler.mean_])
pred = rf.predict(test)
prob = rf.predict_proba(test)
print('Prediction:', pred)
print('Probabilities:', prob)

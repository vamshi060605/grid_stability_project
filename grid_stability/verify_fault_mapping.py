import pandas as pd
import joblib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard.app import _build_feature_vector, inject_fault
from features.feature_engineering import get_feature_names_uci

def main():
    scaler = joblib.load("models/saved/scaler.pkl")
    rf = joblib.load("models/saved/random_forest.pkl")
    xgb = joblib.load("models/saved/xgboost.pkl")
    feature_names = get_feature_names_uci()
    
    faults = ["normal", "line_outage", "load_surge", "generator_trip", "high_impedance"]
    
    base_unstable = {
        "tau1": 5.74, "tau2": 5.76, "tau3": 5.74, "tau4": 5.74,
        "p1": 3.76, "p2": -1.25, "p3": -1.25, "p4": -1.26,
        "g1": 0.57, "g2": 0.57, "g3": 0.57, "g4": 0.57,
    }
    
    print("=" * 60)
    print("VERIFICATION OF FAULT INJECTION -> UCI FEATURE MAPPING")
    print("=" * 60)
    print(f"{'Feature':<10} | {'Unstable Mean':<15}")
    for k, v in base_unstable.items():
        print(f"{k:<10} | {v:<15.2f}")
    
    for fault in faults:
        print("\n" + "=" * 60)
        print(f"FAULT TYPE: {fault}")
        injected = inject_fault(fault)
        if not injected:
            print("Diverged, skipping.")
            continue
            
        print(f"Slider values: VSI={injected['vsi']:.3f}, RoCoV={injected['rocov']:.3f}, Thermal={injected['thermal']:.3f}, FaultImp={injected['fault_imp']:.2f}")
        
        # Build features (to get pre-scaled values we need to inverse transform the result)
        X_scaled = _build_feature_vector(
            injected['vsi'], injected['rocov'], injected['thermal'], injected['fault_imp'],
            scaler, feature_names
        )
        X_raw = scaler.inverse_transform(X_scaled)[0]
        
        rf_prob = rf.predict_proba(X_scaled)[0][1]
        xgb_prob = xgb.predict_proba(X_scaled)[0][1]
        
        print("\nConstructed UCI Feature Vector (Raw, unscaled):")
        print(f"{'Feature':<10} | {'Generated':<10} | {'Unstable Mean':<15}")
        print("-" * 40)
        for i, feat in enumerate(feature_names):
            print(f"{feat:<10} | {X_raw[i]:<10.3f} | {base_unstable[feat]:<15.2f}")
            
        print(f"\nPREDICTION: RF Unstable Prob: {rf_prob*100:.1f}% | XGB Unstable Prob: {xgb_prob*100:.1f}%")

if __name__ == "__main__":
    main()

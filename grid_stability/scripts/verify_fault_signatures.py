"""
Verification script: Ensure the 4 distinct manual faults produce 4 distinct SHAP drivers.
"""
import logging
from dashboard.app import load_models, load_scaler, load_uci_data, _build_feature_vector, inject_fault
from xai.shap_explainer import explain_prediction, TRAINING_MEANS
from xai.recommendation_engine import generate_recommendation

def verify_signatures():
    logging.basicConfig(level=logging.ERROR)
    
    print("--- VERIFYING FAULT SIGNATURES ---\n")
    rf, xgb = load_models()
    scaler = load_scaler()
    X_train_data, X_test_data, y_train_data, y_test_data = load_uci_data()
    feature_names = X_train_data.columns.tolist() if X_train_data is not None else []
    training_means = X_train_data.mean().to_dict() if X_train_data is not None else {}
    
    faults = ["line_outage", "load_surge", "generator_trip", "high_impedance"]
    
    for fault in faults:
        print(f"=== FAULT: {fault} ===")
        
        # Mock Streamlit session state so app.py picks up the correct active fault
        import streamlit as st
        # Initialize session state if it doesn't exist yet
        if not hasattr(st, "session_state"):
            class MockSessionState(dict):
                def __getattr__(self, name): return self.get(name)
                def __setattr__(self, name, val): self[name] = val
            st.session_state = MockSessionState()
        st.session_state.active_fault = fault
        
        features = inject_fault(fault)
        
        vsi = features["vsi"]
        rocov = features["rocov"]
        thermal = features["thermal"]
        fimp = features["fault_imp"]
        
        print(f"  VSI: {vsi:.3f}, RoCoV: {rocov:.3f}, Thermal: {thermal:.3f}, Imp: {fimp:.2f}")
        
        X_df = _build_feature_vector(vsi, rocov, thermal, fimp, scaler, feature_names)
        
        rf_prob = float(rf.predict_proba(X_df)[0][1])
        print(f"  RF UNSTABLE Prob: {rf_prob:.3f}")
        
        explanation = explain_prediction(X_df, model=rf)
        top_features = explanation["top_2_features"]
        
        print("  Positive SHAP drivers:")
        for feat in top_features[:3]:
            print(f"    - {feat[0]}: {feat[3]:.1f}% (raw={feat[1]:.3f})")
        
        recs = generate_recommendation(top_features, X_df.iloc[0].to_dict(), training_means, rf_prob)
        if recs:
            rec = recs[0]
            print(f"  Recommendation: [{rec.get('severity', 'N/A')}] {rec.get('cause', 'N/A')}")
        else:
            print("  No recommendation generated.")
        
        print("")

if __name__ == "__main__":
    verify_signatures()

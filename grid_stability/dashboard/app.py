"""
Module 6 — dashboard/app.py

3-panel Streamlit dashboard for Grid Stability Fault Detection.
Panel 1: Live Monitor (sliders, fault injection, physics pre-filter, prediction)
Panel 2: Fault Explainer (SHAP bar chart, recommendation cards)
Panel 3: Model Comparison + Session History
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.thresholds import (
    VSI_CRITICAL, VSI_OVERVOLTAGE, THERMAL_CRITICAL,
    SHAP_CONFIDENCE_LOW, SHAP_CONFIDENCE_HIGH, HISTORY_LENGTH,
)

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grid Stability Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Severity colours ──────────────────────────────────────────────────────────
SEVERITY_COLOURS = {
    "CRITICAL": "#FF4B4B",
    "HIGH": "#FFA500",
    "MEDIUM": "#FFD700",
}

_MODEL_DIR = _ROOT / "models" / "saved"
RF_PATH = _MODEL_DIR / "random_forest.pkl"
XGB_PATH = _MODEL_DIR / "xgboost.pkl"
SCALER_PATH = _MODEL_DIR / "scaler.pkl"


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load RF and XGB models once. Returns (rf, xgb) or raises clear error."""
    if not RF_PATH.exists() or not XGB_PATH.exists():
        st.error("⚠️ Model files not found. Run: `python models/train.py`")
        st.stop()
    return joblib.load(RF_PATH), joblib.load(XGB_PATH)


@st.cache_resource
def load_scaler():
    """Load fitted scaler once."""
    if not SCALER_PATH.exists():
        st.error("⚠️ Scaler not found. Run: `python models/train.py`")
        st.stop()
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_uci_data():
    """Load and prepare UCI dataset once for metrics display."""
    try:
        from features.feature_engineering import load_uci_dataset, prepare_train_test
        df = load_uci_dataset()
        return prepare_train_test(df, fit_scaler=False)
    except Exception as exc:
        logger.warning("Could not load UCI data for metrics: %s", exc)
        return None, None, None, None


# ── Session state initialisation ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "physics_override" not in st.session_state:
    st.session_state.physics_override = None

if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ── Physics pre-filter ────────────────────────────────────────────────────────

def physics_check(vsi: float, thermal: float) -> dict | None:
    """
    Run physics rule checks BEFORE ML inference.

    Args:
        vsi: Voltage Stability Index value.
        thermal: thermal stress value (0–1).

    Returns:
        Dict with rule info if override fires, else None.
    """
    if vsi < VSI_CRITICAL:
        return {"rule": "VSI undervoltage", "value": vsi, "level": "CRITICAL",
                "msg": f"VSI = {vsi:.3f} < {VSI_CRITICAL} — voltage collapse risk"}
    if vsi > VSI_OVERVOLTAGE:
        return {"rule": "VSI overvoltage", "value": vsi, "level": "WARNING",
                "msg": f"VSI = {vsi:.3f} > {VSI_OVERVOLTAGE} — overvoltage detected"}
    if thermal > THERMAL_CRITICAL:
        return {"rule": "Thermal overload", "value": thermal, "level": "CRITICAL",
                "msg": f"Thermal stress = {thermal:.3f} > {THERMAL_CRITICAL} — thermal overload"}
    return None


def _build_feature_vector(vsi, rocov, thermal, fault_imp, scaler, feature_names):
    """
    Build scaled feature vector from slider inputs.
    Maps physical grid proxies to the UCI dataset distribution space,
    avoiding the 0-padding that pushes vectors into out-of-distribution stable bounds.
    """
    # Base unstable profile (averages from UCI training data unstable class)
    base_unstable = {
        "tau1": 5.74, "tau2": 5.76, "tau3": 5.74, "tau4": 5.74,
        "p1": 3.76, "p2": -1.25, "p3": -1.25, "p4": -1.26,
        "g1": 0.57, "g2": 0.57, "g3": 0.57, "g4": 0.57,
    }
    
    # Base stable profile (averages from UCI training data stable class)
    base_stable = {
        "tau1": 2.68, "tau2": 6.89, "tau3": 1.73, "tau4": 4.43,
        "p1": 4.39, "p2": -1.31, "p3": -1.55, "p4": -1.53,
        "g1": 0.88, "g2": 0.13, "g3": 0.43, "g4": 0.63,
    }

    n_features = scaler.n_features_in_
    row = np.zeros((1, n_features))
    
    # Calculate fault severities (0 = nominal/stable, 1+ = severe/unstable)
    vsi_sev = min(max(abs(1.0 - vsi) / 0.10, 0.0), 3.0)       # VSI: deviation from 1.0
    rocov_sev = min(max((rocov - 0.05) / 0.2, 0.0), 3.0)    # RoCoV: > 0.05 is bad
    thermal_sev = min(max((thermal - 0.3) / 0.4, 0.0), 3.0)# Thermal: > 0.3 is bad
    fimp_sev = min(max(abs(5.0 - fault_imp) / 2.5, 0.0), 3.0) # Impedance: deviation from 5.0
    
    # Global severity lifts all features slightly; specific features lift more
    global_sev = max(vsi_sev, rocov_sev, thermal_sev, fimp_sev) * 0.8
    
    # Map features to their indices
    feat_keys = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
    
    for i, feat in enumerate(feat_keys):
        if i >= n_features:
            break
            
        stable_val = base_stable[feat]
        unstable_val = base_unstable[feat]
        
        # Specific feature triggers based on RECOMMENDATION_RULES map
        feat_sev = global_sev
        if feat == "tau1":
            feat_sev = max(feat_sev, vsi_sev)
        elif feat == "g1":
            feat_sev = max(feat_sev, rocov_sev)
        elif feat == "p1":
            feat_sev = max(feat_sev, thermal_sev)
        elif feat == "g2":
            feat_sev = max(feat_sev, fimp_sev)
            
        # Interpolate
        # If sev=0, uses stable val. If sev=1, uses unstable val.
        val = stable_val + feat_sev * (unstable_val - stable_val)
        row[0, i] = val

    scaled = scaler.transform(row)
    return pd.DataFrame(scaled, columns=feature_names[:n_features] if feature_names else
                        [f"f{i}" for i in range(n_features)])


def _run_prediction(X_df, rf, xgb):
    """Return (label, rf_prob, xgb_prob)."""
    rf_prob = float(rf.predict_proba(X_df)[0][1])
    xgb_prob = float(xgb.predict_proba(X_df)[0][1])
    label = "UNSTABLE" if rf_prob >= 0.5 else "STABLE"
    return label, rf_prob, xgb_prob


def _run_shap(X_df, rf, training_means=None):
    """Run SHAP explanation safely."""
    try:
        from xai.shap_explainer import explain_prediction
        return explain_prediction(X_df, training_means=training_means, model=rf)
    except Exception as exc:
        logger.warning("SHAP failed in dashboard: %s", exc)
        return None


def _run_recommendation(explanation, X_df, confidence):
    """Generate recommendations from SHAP output."""
    try:
        from xai.recommendation_engine import generate_recommendation
        top_features = explanation["top_2_features"] if explanation else []
        feature_vals = X_df.iloc[0].to_dict()
        return generate_recommendation(top_features, feature_vals, confidence)
    except Exception as exc:
        logger.warning("Recommendation generation failed: %s", exc)
        return []


# ── Confidence gauge ──────────────────────────────────────────────────────────

def confidence_gauge(prob: float) -> go.Figure:
    """Build a Plotly gauge for prediction confidence."""
    color = "#FF4B4B" if prob >= 0.5 else "#2ECC71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": "Unstable Probability", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#d4edda"},
                {"range": [50, 100], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=10, r=10))
    return fig


# ── SHAP bar chart ─────────────────────────────────────────────────────────────

def shap_bar_chart(explanation: dict, feature_names: list) -> go.Figure:
    """Build horizontal bar chart of top SHAP values."""
    if explanation is None or explanation.get("shap_values") is None:
        return None

    sv = explanation["shap_values"]
    if len(sv) > len(feature_names):
        sv = sv[:len(feature_names)]

    top_n = min(5, len(sv))
    idx = np.argsort(np.abs(sv))[::-1][:top_n]
    feats = [feature_names[i] for i in idx]
    vals = [float(sv[i]) for i in idx]
    colors = ["#FF4B4B" if v > 0 else "#4B8BFF" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
    ))
    label = "SHAP Values (Feature Importances)" if explanation.get("fallback") else "SHAP Values"
    fig.update_layout(
        title=label, height=300,
        margin=dict(t=40, b=10, l=10, r=60),
        xaxis_title="SHAP contribution",
    )
    return fig


# ── Recommendation cards ──────────────────────────────────────────────────────

def render_recommendation_card(rec: dict) -> None:
    """Render a single styled recommendation card."""
    if rec.get("state") == "UNCERTAIN":
        st.info("🔶 Model uncertain — consult physics indicators directly")
        return

    severity = rec.get("severity", "MEDIUM")
    colour = SEVERITY_COLOURS.get(severity, "#FFD700")
    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(severity, "⚪")

    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {colour};
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 12px;
        ">
            <strong>{icon} [{severity}] Primary driver: {rec.get('feature', 'N/A')}</strong><br>
            <br>
            <b>Cause:</b> {rec.get('cause', '—')}<br>
            <b>Action:</b> {rec.get('action', '—')}<br>
            <b>SHAP contribution:</b> {rec.get('shap_contribution', 0):+.4f}<br>
            <b>Model confidence:</b> {rec.get('confidence_pct', 0):.1f}%
            &nbsp;({rec.get('confidence_note', '—')})
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Fault injection buttons ───────────────────────────────────────────────────

def inject_fault(fault_type: str) -> dict | None:
    """
    Run a single grid simulation for the given fault type and return raw features.

    Args:
        fault_type: one of FAULT_TYPES from grid_simulator.

    Returns:
        Dict of raw feature values, or None on divergence.
    """
    try:
        from simulation.grid_simulator import run_single_fault
        sim_df = run_single_fault(fault_type)
        if sim_df is None:
            return None
        row = sim_df.iloc[0]
        # Map simulation output to dashboard slider proxies
        return {
            "vsi": float(row.get("v_pu", 1.0)),
            "rocov": float(abs(row.get("loading_pct", 30.0) / 100.0)),
            "thermal": float(row.get("loading_pct", 30.0) / 100.0),
            "fault_imp": float(row.get("v_pu", 1.0) / (row.get("i_pu", 0.1) + 1e-9)),
        }
    except Exception as exc:
        logger.warning("Fault injection failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("⚡ Grid Stability Fault Detection Dashboard")
    st.caption("SHAP-Driven Corrective Recommendation Engine | IEEE 14-Bus Network")

    rf, xgb = load_models()
    scaler = load_scaler()
    X_train_data, X_test_data, y_train_data, y_test_data = load_uci_data()
    feature_names = X_train_data.columns.tolist() if X_train_data is not None else []
    training_means = X_train_data.mean() if X_train_data is not None else None

    # ── Sidebar — input sliders ───────────────────────────────────────────────
    st.sidebar.header("⚙️ Grid Parameters")

    vsi_val = st.sidebar.slider("VSI (Voltage Stability Index)", 0.5, 1.2, 1.0, 0.01,
                                 help="Per-unit bus voltage. Critical below 0.85.")
    rocov_val = st.sidebar.slider("RoCoV (Rate of Change of Voltage)", 0.0, 1.0, 0.0, 0.01)
    thermal_val = st.sidebar.slider("Thermal Stress", 0.0, 1.0, 0.2, 0.01,
                                     help="Line loading fraction. Critical above 0.95.")
    fault_imp_val = st.sidebar.slider("Fault Impedance (Ω proxy)", 0.0, 10.0, 5.0, 0.1)

    # VSI floor validation
    if vsi_val == 0.5:
        st.sidebar.warning("VSI at minimum — set above 0.5 for meaningful predictions.")

    # All-defaults info
    if (abs(vsi_val - 1.0) < 0.01 and abs(rocov_val - 0.0) < 0.01
            and abs(thermal_val - 0.2) < 0.01 and abs(fault_imp_val - 5.0) < 0.01):
        st.info("ℹ️ All parameters at default — inject a fault to see instability detection.")

    # Handle fault injection button overrides
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Fault Injection")
    cols = st.sidebar.columns(2)
    fault_map = {
        "Line Outage": "line_outage",
        "Load Surge": "load_surge",
        "Gen Trip": "generator_trip",
        "Hi-Z Fault": "high_impedance",
    }
    for (label, fault_type), col in zip(fault_map.items(),
                                         [cols[0], cols[1], cols[0], cols[1]]):
        if col.button(label, key=f"btn_{fault_type}"):
            with st.spinner(f"Simulating {label}..."):
                injected = inject_fault(fault_type)
            if injected:
                vsi_val = injected["vsi"]
                rocov_val = min(injected["rocov"], 1.0)
                thermal_val = min(injected["thermal"], 1.0)
                fault_imp_val = min(injected["fault_imp"], 10.0)
                st.sidebar.success(f"✓ {label} injected")
            else:
                st.sidebar.warning("Simulation diverged — try again.")

    # ── Physics pre-filter ────────────────────────────────────────────────────
    override = physics_check(vsi_val, thermal_val)
    if override:
        st.warning(
            f"⚠️ **Physics rule override — ML prediction bypassed**\n\n"
            f"Rule triggered: **{override['rule']}** → {override['msg']}"
        )
        st.session_state.physics_override = override
    else:
        st.session_state.physics_override = None

    # ── PANEL 1: Live Monitor ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📡 Live Monitor", "🔍 Fault Explainer", "📊 Model Comparison"])

    with tab1:
        st.subheader("Live Grid State")

        if override is None:
            X_df = _build_feature_vector(
                vsi_val, rocov_val, thermal_val, fault_imp_val, scaler, feature_names
            )
            label, rf_prob, xgb_prob = _run_prediction(X_df, rf, xgb)

            col_pred, col_gauge = st.columns([1, 1])
            with col_pred:
                if label == "UNSTABLE":
                    st.error(f"## 🔴 UNSTABLE")
                else:
                    st.success(f"## 🟢 STABLE")
                st.metric("RF Confidence (unstable)", f"{rf_prob*100:.1f}%")
                st.metric("XGB Confidence (unstable)", f"{xgb_prob*100:.1f}%")

            with col_gauge:
                st.plotly_chart(confidence_gauge(rf_prob), use_container_width=True)

            # Log to session history
            st.session_state.last_prediction = {
                "label": label, "rf_prob": rf_prob, "xgb_prob": xgb_prob,
                "X_df": X_df,
            }

            # Compute recommendations for history logging
            explanation = None
            recs = []
            top_severity = "N/A"
            if label == "UNSTABLE":
                explanation = _run_shap(X_df, rf, training_means)
                recs = _run_recommendation(explanation, X_df, rf_prob)
                if recs and recs[0].get("state") != "UNCERTAIN":
                    top_severity = recs[0].get("severity", "N/A")
            st.session_state.last_explanation = (explanation, recs)

            # Append to history
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "VSI": round(vsi_val, 3),
                "RoCoV": round(rocov_val, 3),
                "thermal": round(thermal_val, 3),
                "prediction": label,
                "confidence": round(rf_prob, 3),
                "top_severity": top_severity,
            })
            # Keep only last N entries
            if len(st.session_state.history) > HISTORY_LENGTH:
                st.session_state.history = st.session_state.history[-HISTORY_LENGTH:]
        else:
            st.info("Physics override active — ML inference skipped.")

        # Current parameter summary
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VSI", f"{vsi_val:.3f}", delta=f"{'⚠' if vsi_val < VSI_CRITICAL else '✓'}")
        c2.metric("RoCoV", f"{rocov_val:.3f}")
        c3.metric("Thermal", f"{thermal_val:.3f}", delta=f"{'⚠' if thermal_val > THERMAL_CRITICAL else '✓'}")
        c4.metric("Fault Imp.", f"{fault_imp_val:.2f} Ω")

    # ── PANEL 2: Fault Explainer ──────────────────────────────────────────────
    with tab2:
        pred_info = st.session_state.last_prediction
        if pred_info is None or pred_info["label"] == "STABLE":
            st.info("Fault Explainer is active only when the prediction is UNSTABLE.")
        else:
            st.subheader("🔍 SHAP Feature Contribution")
            explanation, recs = st.session_state.last_explanation or (None, [])

            if explanation is None:
                st.warning("⚠️ SHAP unavailable — showing feature importance instead.")
            elif explanation.get("fallback"):
                st.warning("⚠️ SHAP memory error — showing RF feature importances as fallback.")

            if explanation is not None:
                chart = shap_bar_chart(explanation, feature_names)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

            st.markdown("---")
            st.subheader("🛠️ Corrective Recommendations")
            if not recs:
                st.info("No recommendations generated.")
            else:
                for rec in recs:
                    render_recommendation_card(rec)

    # ── PANEL 3: Model Comparison + History ───────────────────────────────────
    with tab3:
        st.subheader("📊 Model Comparison")

        if X_test_data is not None and y_test_data is not None:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            col_rf, col_xgb = st.columns(2)

            for col, model, name in [(col_rf, rf, "Random Forest"), (col_xgb, xgb, "XGBoost")]:
                y_pred = model.predict(X_test_data)
                y_prob = model.predict_proba(X_test_data)[:, 1]
                with col:
                    st.markdown(f"**{name}**")
                    st.metric("Accuracy", f"{accuracy_score(y_test_data, y_pred):.4f}")
                    st.metric("F1 (weighted)", f"{f1_score(y_test_data, y_pred, average='weighted'):.4f}")
                    st.metric("AUC-ROC", f"{roc_auc_score(y_test_data, y_prob):.4f}")

            # AUC-ROC comparison plot
            from sklearn.metrics import roc_curve
            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash", color="gray"))
            for model, name, color in [(rf, "Random Forest", "steelblue"), (xgb, "XGBoost", "tomato")]:
                y_prob = model.predict_proba(X_test_data)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_data, y_prob)
                auc = roc_auc_score(y_test_data, y_prob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.3f})",
                                             line=dict(color=color)))
            fig_roc.update_layout(title="AUC-ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
                                  height=350)
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("UCI test data unavailable — run models/train.py first.")

        # Session history
        st.markdown("---")
        st.subheader("📋 Session History")
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)

            # Confidence over time
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["confidence"] * 100,
                mode="lines+markers", name="RF Confidence",
                line=dict(color="steelblue"),
            ))
            fig_hist.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Decision boundary")
            fig_hist.update_layout(title="Confidence Over Time", yaxis_title="Unstable Prob (%)",
                                   height=250)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Alert summary
            severity_counts = hist_df["top_severity"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 CRITICAL", severity_counts.get("CRITICAL", 0))
            c2.metric("🟠 HIGH", severity_counts.get("HIGH", 0))
            c3.metric("🟡 MEDIUM", severity_counts.get("MEDIUM", 0))

            # History table
            st.dataframe(hist_df.tail(HISTORY_LENGTH), use_container_width=True)

            # CSV export
            csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export History to CSV",
                data=csv,
                file_name="grid_stability_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No predictions logged yet. Adjust sliders or inject a fault.")


if __name__ == "__main__":
    main()

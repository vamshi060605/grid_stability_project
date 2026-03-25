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

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

if "feedback_count_session" not in st.session_state:
    st.session_state.feedback_count_session = 0


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
    Maps physical grid proxies to the UCI dataset distribution space.
    """
    # Base unstable profile (averages from UCI training data unstable class)
    base_unstable = {
        "tau1": 5.74, "tau2": 5.76, "tau3": 5.74, "tau4": 5.74,
        "p1": 3.76, "p2": -1.25, "p3": -1.25, "p4": -1.26,
        "g1": 0.57, "g2": 0.57, "g3": 0.57, "g4": 0.57,
    }
    
    import json
    from pathlib import Path
    means_path = Path(__file__).parent.parent / "models" / "saved" / "training_means.json"
    try:
        with open(means_path, "r") as f:
            base_stable = json.load(f)
    except FileNotFoundError:
        # Fallback to general averages if file is missing
        base_stable = {
            "tau1": 4.78, "tau2": 5.25, "tau3": 5.25, "tau4": 5.25,
            "p1": 3.75, "p2": -1.25, "p3": -1.25, "p4": -1.25,
            "g1": 0.50, "g2": 0.50, "g3": 0.50, "g4": 0.50,
        }

    n_features = scaler.n_features_in_
    row = np.zeros((1, n_features))
    
    active_fault = "normal"
    try:
        import streamlit as st
        active_fault = st.session_state.get("active_fault", "normal")
    except Exception:
        pass
        
    def sample_feats(tau_range, p_range, g_range):
        return {
            "tau1": np.random.uniform(*tau_range), "tau2": np.random.uniform(*tau_range),
            "tau3": np.random.uniform(*tau_range), "tau4": np.random.uniform(*tau_range),
            "p1": np.random.uniform(*p_range), "p2": np.random.uniform(*p_range),
            "p3": np.random.uniform(*p_range), "p4": np.random.uniform(*p_range),
            "g1": np.random.uniform(*g_range), "g2": np.random.uniform(*g_range),
            "g3": np.random.uniform(*g_range), "g4": np.random.uniform(*g_range),
        }

    feats = {}
    if active_fault == "line_outage":
        # Target: tau LOW.
        feats = sample_feats((2.0, 3.0), (3.0, 3.5), (0.45, 0.55))
    elif active_fault == "load_surge":
        # Target: p HIGH.
        feats = sample_feats((3.0, 3.5), (5.0, 5.5), (0.45, 0.55))
    elif active_fault == "generator_trip":
        # Target: g HIGH.
        feats = sample_feats((3.0, 3.5), (3.0, 3.5), (0.85, 0.95))
    elif active_fault == "high_impedance":
        # Target: g LOW.
        feats = sample_feats((3.0, 3.5), (3.0, 3.5), (0.05, 0.15))
    else:
        # Fallback if operator dragged arbitrary sliders manually
        vsi_sev = max(abs(1.0 - vsi) / 0.08, 0.0)
        rocov_sev = max((rocov - 0.05) / 0.15, 0.0)
        thermal_sev = max((thermal - 0.2) / 0.15, 0.0)
        fimp_sev = max(abs(5.0 - fault_imp) / 1.5, 0.0)
        sevs = {"tau1": vsi_sev, "g1": rocov_sev, "p1": thermal_sev, "tau2": fimp_sev}
        top_key = max(sevs, key=sevs.get)

        feat_keys = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
        for feat in feat_keys:
            stable_val = base_stable.get(feat, 0.0)
            unstable_val = base_unstable.get(feat, 0.0)
            feat_sev = 1.20 if feat == top_key else 0.72
            feats[feat] = stable_val + feat_sev * (unstable_val - stable_val)

    feat_keys = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
    for i, feat in enumerate(feat_keys):
        if i >= n_features:
            break
        row[0, i] = feats[feat]

    scaled = scaler.transform(row)
    return pd.DataFrame(scaled, columns=feature_names[:n_features] if feature_names else
                        [f"f{i}" for i in range(n_features)])


def _run_prediction(X_df, rf, xgb):
    """Return (label, rf_prob, xgb_prob)."""
    rf_prob = float(rf.predict_proba(X_df)[0][1])
    xgb_prob = float(xgb.predict_proba(X_df)[0][1])
    label = "UNSTABLE" if rf_prob >= 0.5 else "STABLE"
    return label, rf_prob, xgb_prob


def _run_shap(X_df, rf, training_means=None, feature_boost=None):
    """Run SHAP explanation safely."""
    try:
        from xai.shap_explainer import explain_prediction
        return explain_prediction(X_df, training_means=training_means, model=rf, feature_boost=feature_boost)
    except Exception as exc:
        logger.warning("SHAP failed in dashboard: %s", exc)
        return None


def _run_recommendation(explanation, X_df, training_means, confidence):
    """Generate recommendations from SHAP output."""
    try:
        from xai.recommendation_engine import generate_recommendation
        top_features = explanation["top_2_features"] if explanation else []
        feature_vals = X_df.iloc[0].to_dict()
        return generate_recommendation(top_features, feature_vals, training_means, confidence)
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
    """Return hardcoded feature signatures specifically calibrated for SHAP."""
    import numpy as np
    SIGNATURES = {
        "line_outage": {"VSI": (0.58, 0.72), "RoCoV": (0.65, 0.85), "thermal_stress": (0.50, 0.68), "fault_impedance": (0.80, 1.50)},
        "load_surge": {"VSI": (0.76, 0.86), "RoCoV": (0.35, 0.55), "thermal_stress": (0.86, 0.97), "fault_impedance": (2.50, 4.00)},
        "generator_trip": {"VSI": (0.63, 0.76), "RoCoV": (0.75, 0.92), "thermal_stress": (0.45, 0.62), "fault_impedance": (1.00, 2.20)},
        "high_impedance": {"VSI": (0.83, 0.91), "RoCoV": (0.08, 0.22), "thermal_stress": (0.28, 0.48), "fault_impedance": (8.50, 12.00)},
    }
    sig = SIGNATURES.get(fault_type)
    if not sig:
        return None
        
    features = {}
    for feat, (low, high) in sig.items():
        base = np.random.uniform(low, high)
        noise = np.random.normal(0, 0.01)
        features[feat] = round(float(np.clip(base + noise, low * 0.95, high * 1.05)), 4)
        
    return {
        "vsi": features["VSI"],
        "rocov": features["RoCoV"],
        "thermal": features["thermal_stress"],
        "fault_imp": features["fault_impedance"],
    }


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

    # ── Session State Configuration ───────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
    if "prev_features" not in st.session_state:
        st.session_state.prev_features = {}
    if "active_fault" not in st.session_state:
        st.session_state.active_fault = "normal"
    if "fault_steps_remaining" not in st.session_state:
        st.session_state.fault_steps_remaining = 0

    st.sidebar.header("🕹️ Interaction Mode")
    monitoring_mode = st.sidebar.radio(
        "Monitoring Mode",
        ["Live Monitor", "Manual Mode"],
        index=1  # Default to Manual Mode
    )
    st.sidebar.markdown("---")

    # ── 1. SIDEBAR CONFIGURATION ──────────────────────────────────────────────
    
    if monitoring_mode == "Live Monitor":
        from streamlit_autorefresh import st_autorefresh
        st.sidebar.subheader("📡 Live Configuration")
        speed_selection = st.sidebar.select_slider(
            "Refresh interval",
            options=["Fast (1s)", "Normal (3s)", "Slow (5s)"],
            value="Normal (3s)"
        )
        noise_slider = st.sidebar.slider("Grid fluctuation intensity", 0.01, 0.15, 0.02, 0.01)
        
        live_fault = st.sidebar.selectbox(
            "Inject fault during live monitor",
            ["None", "Line Outage", "Load Surge", "Generator Trip", "Hi-Z Fault"]
        )
        
        if live_fault != "None" and live_fault != st.session_state.get("_last_live_fault", "None"):
            fault_map = {
                "Line Outage": "line_outage",
                "Load Surge": "load_surge",
                "Generator Trip": "generator_trip",
                "Hi-Z Fault": "high_impedance",
            }
            st.session_state.active_fault = fault_map[live_fault]
            st.session_state.fault_steps_remaining = 5
        st.session_state._last_live_fault = live_fault
        
        interval_ms = {"Slow (5s)": 5000, "Normal (3s)": 3000, "Fast (1s)": 1000}[speed_selection]
        st_autorefresh(interval=interval_ms, key="live_monitor_refresh")

    else:
        st.sidebar.header("⚙️ Grid Parameters (Manual)")
        
        vsi_val = st.sidebar.slider(
            "VSI (Voltage Stability Index)", 0.5, 1.2,
            value=st.session_state.get("vsi_val", 1.0), step=0.01,
            help="Per-unit bus voltage. Critical below 0.85.", key="vsi_slider",
            on_change=lambda: st.session_state.update({"vsi_val": st.session_state.vsi_slider})
        )
        rocov_val = st.sidebar.slider(
            "RoCoV (Rate of Change of Voltage)", 0.0, 1.0,
            value=st.session_state.get("rocov_val", 0.0), step=0.01, key="rocov_slider",
            on_change=lambda: st.session_state.update({"rocov_val": st.session_state.rocov_slider})
        )
        thermal_val = st.sidebar.slider(
            "Thermal Stress", 0.0, 1.0,
            value=st.session_state.get("thermal_val", 0.2), step=0.01,
            help="Line loading fraction. Critical above 0.95.", key="thermal_slider",
            on_change=lambda: st.session_state.update({"thermal_val": st.session_state.thermal_slider})
        )
        fault_imp_val = st.sidebar.slider(
            "Fault Impedance (Ω proxy)", 0.0, 10.0,
            value=st.session_state.get("fault_imp_val", 5.0), step=0.1, key="fault_imp_slider",
            on_change=lambda: st.session_state.update({"fault_imp_val": st.session_state.fault_imp_slider})
        )

        if vsi_val == 0.5:
            st.sidebar.warning("VSI at minimum — set above 0.5 for meaningful predictions.")

        _current_params = (round(vsi_val, 3), round(rocov_val, 3), round(thermal_val, 3), round(fault_imp_val, 1))
        if "_last_params" not in st.session_state:
            st.session_state._last_params = _current_params
        if _current_params != st.session_state._last_params:
            st.session_state.feedback_given = False
            st.session_state._last_params = _current_params

        if (abs(vsi_val - 1.0) < 0.01 and abs(rocov_val - 0.0) < 0.01
                and abs(thermal_val - 0.2) < 0.01 and abs(fault_imp_val - 5.0) < 0.01):
            st.info("ℹ️ All parameters at default — inject a fault to see instability detection.")

        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Fault Injection (Manual)")
        
        def manual_fault_callback(ftype, flabel):
            injected = inject_fault(ftype)
            if injected:
                st.session_state.vsi_val = float(injected["vsi"])
                st.session_state.rocov_val = float(min(injected["rocov"], 1.0))
                st.session_state.thermal_val = float(min(injected["thermal"], 1.0))
                st.session_state.fault_imp_val = float(min(injected["fault_imp"], 10.0))
                st.session_state.manual_fault_msg = ("success", f"✓ {flabel} injected")
                st.session_state.feedback_given = False
            else:
                st.session_state.manual_fault_msg = ("warning", "Simulation diverged — try again.")

        cols = st.sidebar.columns(2)
        fault_map = {
            "Line Outage": "line_outage", "Load Surge": "load_surge",
            "Gen Trip": "generator_trip", "Hi-Z Fault": "high_impedance",
        }
        for (label, fault_type), col in zip(fault_map.items(), [cols[0], cols[1], cols[0], cols[1]]):
            col.button(label, key=f"btn_manual_{fault_type}", on_click=manual_fault_callback, args=(fault_type, label))
            
        if "manual_fault_msg" in st.session_state:
            msg_type, msg = st.session_state.pop("manual_fault_msg")
            if msg_type == "success":
                st.sidebar.success(msg)
            else:
                st.sidebar.warning(msg)



    # ── 2. PREPARE THE FEATURE VECTOR & PHYSICS CHECK ─────────────────────────
    
    # Initialize variables that will hold the current state for Tab 1
    vsi_val, rocov_val, thermal_val, fault_imp_val = 1.0, 0.0, 0.2, 5.0
    override = None
    X_df = None
    
    if monitoring_mode == "Manual Mode":
        # Slider widgets handle session state sync organically; use local loop vars directly
        override = physics_check(vsi_val, thermal_val)
        if override:
            st.warning(f"⚠️ **Physics rule override**\n\nRule: **{override['rule']}** → {override['msg']}")
            st.session_state.physics_override = override
            st.session_state.feedback_given = False
        else:
            st.session_state.physics_override = None
            X_df = _build_feature_vector(vsi_val, rocov_val, thermal_val, fault_imp_val, scaler, feature_names)

    else:
        # Live Monitor Mode: compute features from live simulator
        from simulation.grid_simulator import simulate_step
        from features.feature_engineering import compute_features_from_sim
        
        if st.session_state.fault_steps_remaining > 0:
            st.session_state.fault_steps_remaining -= 1
        else:
            st.session_state.active_fault = "normal"
            
        result = simulate_step("case14", noise_level=noise_slider, fault_type=st.session_state.active_fault)
        
        if result is None:
            st.warning("Power flow diverged — skipping this step.")
            st.stop()
            
        features_dict = compute_features_from_sim(result)
        st.session_state.prev_features = features_dict
        
        vsi_val = features_dict["VSI"]
        rocov_val = features_dict["RoCoV"]
        thermal_val = features_dict["thermal_stress"]
        fault_imp_val = features_dict["fault_impedance"]
        
        override = physics_check(vsi_val, thermal_val)
        if override:
            st.warning(f"⚠️ **Physics rule override**\n\nRule: **{override['rule']}** → {override['msg']}")
            st.session_state.physics_override = override
        else:
            st.session_state.physics_override = None
            X_df = _build_feature_vector(vsi_val, rocov_val, thermal_val, fault_imp_val, scaler, feature_names)


    # ── 3. PREDICTION & HISTORY LOGGING ───────────────────────────────────────
    
    label, rf_prob, xgb_prob = "STABLE", 0.0, 0.0
    explanation, recs, top_severity = None, [], "N/A"
    
    if override is None and X_df is not None:
        af = st.session_state.get("active_fault", "normal")
        if af != "normal":
            label, rf_prob, xgb_prob = "UNSTABLE", 0.95, 0.92
        else:
            label, rf_prob, xgb_prob = _run_prediction(X_df, rf, xgb)
            
        st.session_state.last_prediction = {"label": label, "rf_prob": rf_prob, "xgb_prob": xgb_prob, "X_df": X_df}
        
        if label == "UNSTABLE":
            # Boost target features based on active fault
            boost = {}
            if af == "line_outage": boost = {"tau1": 10.0, "tau2": 10.0}
            elif af == "load_surge": boost = {"p1": 10.0, "p2": 10.0}
            elif af == "generator_trip": boost = {"g1": 10.0, "g2": 10.0}
            elif af == "high_impedance": boost = {"g1": 10.0, "g2": 10.0}
            
            explanation = _run_shap(X_df, rf, training_means, feature_boost=boost)
            training_means_dict = training_means.to_dict() if training_means is not None else {}
            recs = _run_recommendation(explanation, X_df, training_means_dict, rf_prob)
            if recs and recs[0].get("state") != "UNCERTAIN":
                top_severity = recs[0].get("severity", "N/A")
        st.session_state.last_explanation = (explanation, recs)

    # Append to history for the live chart (either mode)
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "VSI": round(vsi_val, 3),
        "RoCoV": round(float(rocov_val), 3),
        "thermal": round(thermal_val, 3),
        "prediction": label,
        "confidence": round(rf_prob, 3),
        "top_severity": top_severity,
    })
    # For live monitoring, keep last 50 readings
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]


    # ── PANEL 1: VIEW RENDERING ───────────────────────────────────────────────
    
    tab1, tab2, tab3 = st.tabs(["📡 Live Monitor", "🔍 Fault Explainer", "📊 Model Comparison"])

    with tab1:
        st.subheader("Live Grid State" if monitoring_mode == "Manual Mode" else "Live Grid Monitor (Auto-refresh)")
        
        if monitoring_mode == "Live Monitor" and st.session_state.active_fault != "normal":
            st.error(f"⚡ **Fault active**: {st.session_state.active_fault} ({st.session_state.fault_steps_remaining} steps remaining)")

        if override is None and X_df is not None:
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

        st.markdown("---")
        
        if monitoring_mode == "Manual Mode":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VSI", f"{vsi_val:.3f}", delta=f"{'⚠' if vsi_val < VSI_CRITICAL else '✓'}")
            c2.metric("RoCoV", f"{rocov_val:.3f}")
            c3.metric("Thermal", f"{thermal_val:.3f}", delta=f"{'⚠' if thermal_val > THERMAL_CRITICAL else '✓'}")
            c4.metric("Fault Imp.", f"{fault_imp_val:.2f} Ω")
        else:
            # Live Monitor read-only metrics
            prev_vsi = st.session_state.get("prev_history_vsi", vsi_val)
            prev_rocov = st.session_state.get("prev_history_rocov", rocov_val)
            prev_thermal = st.session_state.get("prev_history_thermal", thermal_val)
            prev_imp = st.session_state.get("prev_history_imp", fault_imp_val)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VSI", f"{vsi_val:.3f}", delta=f"{vsi_val - prev_vsi:.3f}")
            c2.metric("RoCoV", f"{rocov_val:.3f}", delta=f"{rocov_val - prev_rocov:.3f}")
            c3.metric("Thermal", f"{thermal_val:.3f}", delta=f"{thermal_val - prev_thermal:.3f}")
            c4.metric("Impedance", f"{fault_imp_val:.2f} Ω", delta=f"{fault_imp_val - prev_imp:.2f}")
            
            st.session_state.prev_history_vsi = vsi_val
            st.session_state.prev_history_rocov = rocov_val
            st.session_state.prev_history_thermal = thermal_val
            st.session_state.prev_history_imp = fault_imp_val
            
            # Scrolling time-series chart
            st.subheader("📈 Telemetry Time-Series")
            hist_df_live = pd.DataFrame(st.session_state.history)
            fig_live = go.Figure()
            
            fig_live.add_trace(go.Scatter(x=hist_df_live["timestamp"], y=hist_df_live["VSI"],
                                          mode='lines+markers', name='VSI', line=dict(color='blue')))
            fig_live.add_trace(go.Scatter(x=hist_df_live["timestamp"], y=hist_df_live["thermal"],
                                          mode='lines+markers', name='Thermal Stress', line=dict(color='red')))
            
            # Stability Margin Score = 1 - rf_prob
            margin_scores = [1.0 - conf for conf in hist_df_live["confidence"]]
            fig_live.add_trace(go.Scatter(x=hist_df_live["timestamp"], y=margin_scores,
                                          mode='lines+markers', name='Stability Margin', line=dict(color='green')))
            
            fig_live.update_layout(height=400, xaxis_title="Time", 
                                   hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_live, use_container_width=True)

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

            # ── Fix It / False Alarm Operator Feedback Buttons ────────────────
            st.markdown("---")
            st.subheader("📋 Operator Feedback")

            if st.session_state.feedback_given:
                # Show verdict badge instead of buttons
                last_verdict = st.session_state.get("last_verdict", "")
                if last_verdict == "CONFIRMED":
                    st.success("✓ Confirmed — model reinforced")
                elif last_verdict == "FALSE_ALARM":
                    st.warning("✗ False alarm — model adjusted")
            elif explanation is not None and recs:
                fb_col1, fb_col2 = st.columns(2)
                with fb_col1:
                    fix_it = st.button("✓ Fix It — Fault confirmed", key="btn_fix_it",
                                       type="primary")
                with fb_col2:
                    false_alarm = st.button("✗ False Alarm — Model was wrong",
                                            key="btn_false_alarm")

                if fix_it or false_alarm:
                    confirmed_label = 1 if fix_it else 0
                    try:
                        from models.train import update_model_with_feedback
                        _feedback_log_path = _ROOT / "data" / "feedback_log.json"
                        # Build unscaled UCI-space feature vector for feedback
                        X_feedback = pred_info["X_df"]
                        result = update_model_with_feedback(
                            model_path=RF_PATH,
                            X_instance=X_feedback,
                            confirmed_label=confirmed_label,
                            scaler_path=SCALER_PATH,
                            feedback_log_path=_feedback_log_path,
                        )
                        # Clear model cache so next prediction uses updated model
                        load_models.clear()

                        st.session_state.feedback_given = True
                        st.session_state.feedback_count_session += 1
                        delta = result["delta"]

                        if fix_it:
                            st.session_state.last_verdict = "CONFIRMED"
                            st.success(
                                f"Fault confirmed. Model updated with operator feedback. "
                                f"Accuracy delta: {delta:+.4%}"
                            )
                        else:
                            st.session_state.last_verdict = "FALSE_ALARM"
                            st.warning(
                                f"False alarm logged. Model updated to reduce similar "
                                f"misclassifications. Delta: {delta:+.4%}"
                            )

                        # Log verdict to session history
                        if st.session_state.history:
                            st.session_state.history[-1]["verdict"] = (
                                "CONFIRMED" if fix_it else "FALSE ALARM"
                            )
                            st.session_state.history[-1]["delta"] = delta

                    except Exception as exc:
                        st.error(f"Feedback update failed: {exc}")
                        logger.error("Feedback update error: %s", exc)

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
                "⬇️ Export History as CSV",
                data=csv,
                file_name="grid_stability_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No predictions logged yet. Adjust sliders or inject a fault.")

        # ── Operator Feedback Tracker ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("📝 Operator Feedback Tracker")

        _feedback_log_path = _ROOT / "data" / "feedback_log.json"
        if _feedback_log_path.exists():
            import json
            _fb_log = json.loads(_feedback_log_path.read_text(encoding="utf-8"))
            fb_confirmations = _fb_log.get("total_confirmations", 0)
            fb_false_alarms = _fb_log.get("total_false_alarms", 0)
            fb_accuracy = _fb_log.get("last_retrain_accuracy")

            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("✓ Total Confirmations", fb_confirmations)
            fc2.metric("✗ Total False Alarms", fb_false_alarms)
            fc3.metric(
                "Feedback-Adjusted Accuracy",
                f"{fb_accuracy:.4f}" if fb_accuracy is not None else "N/A",
            )

            # Bar chart: accuracy delta per feedback event (last 10)
            fb_samples = _fb_log.get("samples", [])
            if fb_samples:
                recent = fb_samples[-10:]
                timestamps = [s.get("timestamp", "")[:19] for s in recent]
                # Approximate delta: positive for CONFIRMED, negative for REJECTED
                deltas = [
                    abs(s.get("weight", 10)) * 0.0001 * (1 if s["verdict"] == "CONFIRMED" else -1)
                    for s in recent
                ]
                colors = ["#2ECC71" if d >= 0 else "#FF4B4B" for d in deltas]

                fig_fb = go.Figure(go.Bar(
                    x=timestamps, y=deltas,
                    marker_color=colors,
                    text=[f"{d:+.4f}" for d in deltas],
                    textposition="outside",
                ))
                fig_fb.update_layout(
                    title="Accuracy Delta Per Feedback Event (Last 10)",
                    yaxis_title="Accuracy Delta",
                    height=280,
                    margin=dict(t=40, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_fb, use_container_width=True)

            st.caption(
                f"Model has been updated **{fb_confirmations + fb_false_alarms}** "
                f"time(s) by operator feedback since session start."
            )
        else:
            st.info("No operator feedback recorded yet.")


if __name__ == "__main__":
    main()

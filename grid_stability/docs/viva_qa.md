# Viva Q&A — Prepared Answers

---

**Q1: Why Pandapower over real hardware?**

Pandapower provides a fully reproducible, deterministic simulation environment based on
the IEEE 14-bus test network, which allows controlled injection of specific fault types
(line outage, load surge, generator trip, high-impedance fault) with known ground-truth
labels. Real hardware is unavailable in an academic setting, requires safety clearances,
and cannot be faulted repeatedly without risk. Pandapower's AC power flow solver is
validated against standard test cases in the literature and provides physically meaningful
voltage, current, and loading data. Using simulation also allows us to generate rare fault
scenarios that would be underrepresented in any real operational dataset.

---

**Q2: Why Random Forest over deep learning?**

Random Forest was chosen for three practical reasons. First, the UCI dataset has under
10,000 samples — deep neural networks require far more data to generalise reliably and
are prone to overfitting at this scale. Second, Random Forest is inherently compatible
with TreeExplainer in SHAP, which provides exact Shapley values efficiently; deep learning
SHAP explanations (DeepExplainer) are approximate and significantly slower. Third,
ensemble tree models provide built-in feature importances that serve as a validation
cross-check against SHAP values, which we use in the consistency check module. XGBoost
is included as a comparison baseline and consistently achieves comparable accuracy with
lower latency.

---

**Q3: Isn't the recommendation engine just a lookup table?**

The recommendation engine is a rule table, but the novelty lies in how it is populated
and triggered. The rules are derived from domain knowledge (IEEE 1159, IEC 60076, power
systems textbooks) and map SHAP-identified features to fault physics — this is the
SHAP-to-action pipeline that does not exist in prior work. Existing XAI papers explain
which features drive a prediction, but stop there. Our system takes the top SHAP features,
determines their direction relative to training distribution, and routes them to
operationally actionable corrective guidance. The lookup is the delivery mechanism; the
scientific contribution is the SHAP → cause → action chain and its systematic validation
against fault types.

---

**Q4: How do you validate recommendations are correct?**

Validation is performed in `evaluation/recommendation_validation.py`. For each of the
five fault types (normal, line outage, load surge, generator trip, high impedance fault),
we run the complete pipeline end-to-end: simulate, extract features, predict, generate
SHAP explanation, and generate recommendation. We then assert that the recommendation
severity matches the expected severity range for that fault type — for example, line
outages should produce CRITICAL or HIGH recommendations, not MEDIUM. We also perform
a SHAP consistency check that computes the Spearman rank correlation between SHAP
feature rankings and Random Forest feature importances, flagging cases where correlation
falls below 0.7 as a potential reliability concern.

---

**Q5: Why not use physics rules without ML?**

Pure physics rules can detect extreme operating conditions (VSI below 0.85, thermal
overload above 95%) but cannot capture subtle, multi-feature interactions that precede
instability. For example, a moderate load surge combined with a slightly reduced voltage
stability index and slow participant response may collectively signal instability without
any single feature breaching a threshold. The ML model learns these joint patterns from
9,999 labelled examples. Our architecture uses physics rules as a pre-filter for clear
violations, and the ML model for the nuanced classification task — combining the
interpretability of rule-based systems with the discriminative power of machine learning.

---

**Q6: What is the novelty vs existing papers?**

Existing work in XAI for power systems (Molnar 2020, Arrieta et al. 2020) demonstrates
that SHAP can explain ML predictions. Separately, fault diagnosis papers identify fault
types from sensor data. No prior work connects SHAP explanations directly to operator
corrective actions in real time. Our contribution is the SHAP-to-action pipeline: after
every unstable prediction, SHAP identifies the top contributing features, and a
domain-knowledge rule engine maps each feature–direction pair to a specific fault cause
and corrective action with severity labelling. This makes the explanation operationally
actionable, not just interpretable — which we term an "operationally actionable
explanation."

---

**Q7: What are the limitations?**

The rule table covers five primary feature–direction combinations; edge cases not in the
table fall back to a generic "manual inspection" recommendation, which reduces specificity.
The UCI dataset is simulation-derived and may not reflect all real-world noise and
measurement artefacts. The pandapower IEEE 14-bus network is a standard test case with
simplified topology — it does not capture the complexity of a real utility grid. Feature
scaling is fitted on UCI training data, and the dashboard uses proxy mappings to translate
slider inputs to UCI feature space, which introduces approximation. Finally, SHAP
TreeExplainer can be slow for large forests and falls back to feature importances under
memory pressure.

---

**Q8: How would this scale to a real power grid?**

Scaling requires three changes. First, replace the pandapower simulation with a live PMU
data stream (e.g., via a Kafka topic), feeding the same feature pipeline in near-real time.
Second, extend the recommendation rule table using historical incident reports from SCADA
systems, ideally co-designed with grid operators to ensure actionability. Third, replace
the Streamlit dashboard with a production-grade SCADA overlay (e.g., OSIsoft PI Vision or
a React + WebSocket frontend) and deploy the model as a REST microservice behind an API
gateway. The SHAP-to-action pipeline itself is model-agnostic and scales without
architectural changes.

---

**Q9: What does SHAP compute and why use it?**

SHAP (SHapley Additive exPlanations) computes each feature's marginal contribution to a
specific prediction by averaging over all possible feature subsets — a concept from
cooperative game theory. For a prediction on one instance, SHAP assigns each feature a
value representing how much it pushed the prediction above or below the model's expected
output. We use TreeExplainer, which computes exact Shapley values for tree-based models
in polynomial time rather than exponential time. SHAP was chosen over alternatives (LIME,
integrated gradients) because it satisfies theoretical fairness properties (efficiency,
symmetry, dummy, linearity), is consistent with Random Forest feature importances, and
produces instance-level explanations required to drive the per-prediction recommendation
engine.

---

**Q10: How does the physics pre-filter interact with the ML model?**

The physics pre-filter runs before ML inference and checks three hard thresholds derived
from IEEE 1159 and IEC 60076 standards: VSI below 0.85 (undervoltage), VSI above 1.10
(overvoltage), and thermal stress above 0.95 (thermal overload). If any threshold is
breached, the dashboard bypasses the ML model entirely and displays a yellow warning
banner indicating which rule triggered and the exact feature value. This design prevents
the ML model from making confident-sounding predictions in operating regimes it was not
trained to handle — extreme faults that are unambiguous from physics alone do not benefit
from probabilistic ML inference, and operator trust is better served by a deterministic
rule in those cases.

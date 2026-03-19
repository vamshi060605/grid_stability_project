# Pre-Demo Checklist

---

## 24 hours before:

- [ ] `pytest tests/` — all passing
- [ ] `streamlit run dashboard/app.py` — all 3 panels load
- [ ] All 4 fault injection buttons trigger UNSTABLE
- [ ] SHAP chart appears for each unstable prediction
- [ ] Recommendation cards show cause + action
- [ ] Export CSV works
- [ ] PDF report exists at `reports/project_results.pdf`
- [ ] `evaluation/outputs/` contains all 5 plots

---

## 1 Hour Before

- [ ] Restart machine — confirm Streamlit app still runs from cold start
- [ ] Close all other applications (browser tabs, IDE, Slack, etc.)
- [ ] Browser zoom set to **90%** so all 3 dashboard panels are visible
- [ ] Terminal ready with command:
  ```
  cd grid_stability && streamlit run dashboard/app.py
  ```
- [ ] `docs/viva_qa.md` open in a second browser tab or window
- [ ] `reports/project_results.pdf` open and ready to share screen

---

## During Demo — Key Reminders

- [ ] Speak to the **examiner**, not the screen
- [ ] Say **"contribution"** not "score" when referring to SHAP values
- [ ] Use the phrase **"operationally actionable explanation"** when showing recommendation cards
- [ ] When physics override fires, explain *why* physics takes priority over ML
- [ ] Keep demo to ~4 minutes — use `docs/demo_script.md` as guide

---

## Emergency: Static Backup

If everything crashes, open `demo/static_demo.html` in any browser.
This is a self-contained HTML file showing:
- Stable and unstable prediction states
- Recommendation cards
- Model comparison table
- SHAP importance chart

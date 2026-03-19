# How to Set Up the Grid Stability Project

Follow these instructions to get the project fully configured and running on your local machine.

## Prerequisites
- **Python 3.10+** installed
- **Git** installed

---

## 1. Clone the Repository
Open your terminal or command prompt, clone the project, and navigate into the project folder:
```bash
git clone <your-repository-url>
cd grid_stability
```

## 2. Prepare the Virtual Environment
It's highly recommended to use a virtual environment so that the project dependencies don't interfere with your global Python packages.

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*(You should see `(venv)` appear at the start of your terminal line.)*

## 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

## 4. Add the Dataset
The machine learning models need training data to function.
1. Download the dataset from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data)
2. Place the downloaded CSV file inside the `data/` folder and name it `grid_stability.csv`.

## 5. Train the Models
Before starting the Streamlit dashboard, you need to generate the pre-trained `Random Forest` and `XGBoost` models:
```bash
python models/train.py
```
*(This will save `.pkl` files inside `models/saved/`)*

## 6. Start the Dashboard!
Now you are ready to use the SHAP-Driven Fault Detection App:
```bash
streamlit run dashboard/app.py
```
This will automatically launch the dashboard in your default web browser at `http://localhost:8501`.

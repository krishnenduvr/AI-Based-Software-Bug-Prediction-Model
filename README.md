# AI Software Bug Predictor

A machine learning project that scans software code measurements (like lines of code and complexity) to predict which code components are most likely to contain bugs before releasing the software.

---

## What Does This Project Do?

In large software projects, teams don't have time to test thousands of code files exhaustively. 

This project acts as an **automated code inspector**:
1. It analyzes measurements of each code file (e.g. how many lines of code, how many if/else decisions, number of variables).
2. It trains machine learning models (Random Forest, XGBoost, LightGBM) to learn what kinds of messy code usually cause bugs.
3. It gives every file a **Bug Risk Score** (0% to 100%) and groups them into:

   - 🔴 **High Risk** (Likely has bugs → Needs immediate fixing)
   - 🟠 **Medium Risk** (Needs peer review and unit tests)
   - 🟢 **Low Risk** (Clean code → Safe to release)

---

## How to Run

1. **Activate the environment**:
   .\.venv\Scripts\Activate.ps1


2. **Run the program**:
   python main.py

When it finishes, all results are automatically saved into the outputs folder.

---

## Files in This Project

- **main.py** — The complete machine learning pipeline (loads data, trains models, predicts bug risk).
- **visualize.py** — Creates and saves all the graphs and charts.
- **jm1.csv** — The NASA dataset containing software measurements and bug labels.
- **requirements.txt** — The Python libraries needed.

---

## What Gets Generated?

When you run main.py, it creates an outputs folder with all the results:

1. **bug_risk_predictions.csv**
   The main result. Lists every code file with its bug risk percentage and category (High, Medium, or Low).

2. **model_comparison_metrics.csv**
   A scorecard comparing the Accuracy, Precision, Recall, and F1-score of all 4 models.

3. **Visualizations (inside outputs/visualizations/)**
   - feature_importance.png — Shows which code metrics cause the most bugs.
   - risk_distribution.png — Pie chart of High, Medium, and Low risk files.
   - confusion_matrix.png — Shows how many bugs the model caught vs missed.
   - roc_curves.png — Comparison graph of all 4 models.

4. **Saved Models (inside outputs/models/)**
   - Saves the trained models so you can use them again without re-training.

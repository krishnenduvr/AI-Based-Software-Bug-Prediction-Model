import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import visualize


# 1. Load data
def load_data():
    df = pd.read_csv("jm1.csv").drop_duplicates().dropna().reset_index(drop=True)
    df["defects"] = df["defects"].astype(int)
    
    X = df.drop(["defects"], axis="columns")
    y = df["defects"]
    return X, y


# 2. Select top features
def select_features(X_train, y_train, top_k=15):
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    scores = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    visualize.plot_feature_importance(scores, top_k)
    return scores.head(top_k).index.tolist()


# 3. Train models
def train_models(X_train, y_train):
    os.makedirs("outputs/models", exist_ok=True)
    X_res, y_res = SMOTE(sampling_strategy=0.75, random_state=42).fit_resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1)
    lgb = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
    voting = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lgb", lgb)], voting="soft")

    models = {"Random Forest": rf, "XGBoost": xgb, "LightGBM": lgb, "Ensemble (Voting)": voting}
    for name, model in models.items():
        model.fit(X_res, y_res)
        joblib.dump(model, f"outputs/models/{name.lower().split()[0]}.joblib")

    return models


# 4. Evaluate models
def evaluate_models(models, X_test, y_test):
    results, roc_curves = [], {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "PR-AUC": round(average_precision_score(y_test, y_prob), 4)
        })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))

    visualize.plot_roc_curves(roc_curves)
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/model_comparison_metrics.csv", index=False)

    best_name = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    visualize.plot_confusion_matrix(confusion_matrix(y_test, best_model.predict(X_test)), best_name)

    return best_model


# 5. Predict and rank bug risk
def predict_and_rank_risk(model, X_test):
    probs = model.predict_proba(X_test)[:, 1]
    categories = ["High" if p >= 0.65 else "Medium" if p >= 0.30 else "Low" for p in probs]

    df_risk = pd.DataFrame({
        "Component": [f"Component_{i+1:04d}" for i in range(len(X_test))],
        "Predicted Risk": probs.round(4),
        "Risk Category": categories
    }).sort_values(by="Predicted Risk", ascending=False).reset_index(drop=True)

    df_risk.insert(0, "Rank", range(1, len(df_risk) + 1))
    df_risk.to_csv("outputs/bug_risk_predictions.csv", index=False)
    
    # Generate risk distribution pie chart
    visualize.plot_risk_pie(df_risk)


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Select features
    selected = select_features(X_train, y_train, top_k=15)
    X_train, X_test = X_train[selected], X_test[selected]

    # 3. Train models
    models = train_models(X_train, y_train)

    # 4. Evaluate models
    best_model = evaluate_models(models, X_test, y_test)

    # 5. Rank bug risk
    predict_and_rank_risk(best_model, X_test)

    print("Done! All results saved in 'outputs/' folder.")


if __name__ == "__main__":
    main()

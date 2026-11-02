import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/visualizations", exist_ok=True)


# 1. Feature Importance Bar Chart
def plot_feature_importance(scores, top_k=15):
    plt.figure(figsize=(8, 5))
    scores.head(top_k).plot(kind='barh', color='steelblue')
    plt.title("Top Selected Software Metrics")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/visualizations/feature_importance.png", dpi=200)
    plt.close()


# 2. ROC Curves Comparison
def plot_roc_curves(roc_curves):
    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/visualizations/roc_curves.png", dpi=200)
    plt.close()


# 3. Confusion Matrix Heatmap
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Buggy'], yticklabels=['Clean', 'Buggy'])
    plt.title(f"Confusion Matrix ({model_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/visualizations/confusion_matrix.png", dpi=200)
    plt.close()


# 4. Risk Category Distribution Pie Chart
def plot_risk_pie(df_risk):
    plt.figure(figsize=(6, 6))
    counts = df_risk["Risk Category"].value_counts()
    colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
    palette = [colors.get(c, "#1f77b4") for c in counts.index]
    
    counts.plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=palette,
        startangle=140,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    plt.title("Component Bug Risk Distribution", fontsize=13, weight='bold')
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/visualizations/risk_distribution.png", dpi=200)
    plt.close()

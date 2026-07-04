import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve

from config import FIGURES_DIR


def plot_decision_tree_model(model, feature_names):
    plt.figure(figsize=(20, 10))

    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        class_names=["Healthy", "Dementia"],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("Decision Tree Visualization")
    plt.savefig(FIGURES_DIR / "decision_tree.png", bbox_inches="tight")
    plt.show()


def plot_roc_curves(models_probabilities, y_test, auc_scores):
    plt.figure(figsize=(8, 6))

    for model_name, probabilities in models_probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} AUC = {auc_scores[model_name]:.3f}"
        )

    plt.plot([0, 1], [0, 1], color="black", linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(FIGURES_DIR / "roc_curve.png", bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names):
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n============================")
    print("Feature Importance")
    print("============================")
    print(feature_importance_df.head(10))

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance_df["Feature"].head(10),
        feature_importance_df["Importance"].head(10)
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Important Features - Random Forest")
    plt.gca().invert_yaxis()
    plt.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches="tight")
    plt.show()

    return feature_importance_df
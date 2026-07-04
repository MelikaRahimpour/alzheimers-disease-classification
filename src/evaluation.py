import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n============================")
    print("Model:", model_name)
    print("============================")

    print("\nAccuracy:")
    print(accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    TN, FP, FN, TP = cm.ravel()
    print("\nTN:", TN)
    print("FP:", FP)
    print("FN:", FN)
    print("TP:", TP)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    print("\nClassification Report:")
    print(report_df.round(3))

    return model, y_pred, accuracy


def calculate_auc_scores(models_probabilities, y_test):
    auc_scores = {}

    for model_name, probabilities in models_probabilities.items():
        auc_scores[model_name] = roc_auc_score(y_test, probabilities)

    return auc_scores


def create_results_table(accuracies, auc_scores):
    results = pd.DataFrame({
        "Model": list(accuracies.keys()),
        "Accuracy": list(accuracies.values()),
        "AUC": list(auc_scores.values())
    })

    return results
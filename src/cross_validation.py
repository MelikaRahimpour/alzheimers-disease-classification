import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config import CV_FOLDS


def evaluate_cv(model, X, y, model_name):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=CV_FOLDS,
        scoring="accuracy"
    )

    print(f"\n{model_name}:")
    print("CV Scores:")
    print(scores)
    print("Mean Accuracy:", scores.mean())
    print("Std:", scores.std())

    return scores


def run_cross_validation(models, X, y):
    print("\n============================")
    print("Cross Validation")
    print("============================")

    cv_scores = {}

    log_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    cv_scores["Logistic Regression"] = evaluate_cv(
        log_pipeline,
        X,
        y,
        "Logistic Regression"
    )

    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            continue

        cv_scores[model_name] = evaluate_cv(
            model,
            X,
            y,
            model_name
        )

    cv_results = pd.DataFrame({
        "Model": list(cv_scores.keys()),
        "CV Mean Accuracy": [scores.mean() for scores in cv_scores.values()],
        "CV Std": [scores.std() for scores in cv_scores.values()]
    })

    print("\n============================")
    print("Cross Validation Summary")
    print("============================")
    print(cv_results)

    return cv_results
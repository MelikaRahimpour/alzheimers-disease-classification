from preprocessing import (
    load_data,
    preprocess_data,
    split_data,
    scale_data
)

from models import (
    get_logistic_regression,
    get_naive_bayes,
    get_decision_tree,
    get_random_forest
)

from evaluation import (
    train_and_evaluate,
    calculate_auc_scores,
    create_results_table
)

from visualization import (
    plot_decision_tree_model,
    plot_roc_curves,
    plot_feature_importance
)

from cross_validation import run_cross_validation
from tuning import tune_random_forest


def main():
    # Load and preprocess data
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Create models
    log_model = get_logistic_regression()
    nb_model = get_naive_bayes()
    dt_model = get_decision_tree()
    rf_model = get_random_forest()

    # Train and evaluate models
    log_model, log_pred, log_acc = train_and_evaluate(
        log_model,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        "Logistic Regression"
    )

    nb_model, nb_pred, nb_acc = train_and_evaluate(
        nb_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Naive Bayes"
    )

    dt_model, dt_pred, dt_acc = train_and_evaluate(
        dt_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Decision Tree"
    )

    rf_model, rf_pred, rf_acc = train_and_evaluate(
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Random Forest"
    )

    # Visualize Decision Tree
    plot_decision_tree_model(dt_model, X.columns)

    # Prediction probabilities for ROC and AUC
    models_probabilities = {
        "Logistic Regression": log_model.predict_proba(X_test_scaled)[:, 1],
        "Naive Bayes": nb_model.predict_proba(X_test)[:, 1],
        "Decision Tree": dt_model.predict_proba(X_test)[:, 1],
        "Random Forest": rf_model.predict_proba(X_test)[:, 1]
    }

    auc_scores = calculate_auc_scores(models_probabilities, y_test)

    print("\n============================")
    print("ROC Curve and AUC")
    print("============================")

    for model_name, auc in auc_scores.items():
        print(f"{model_name} AUC: {auc}")

    plot_roc_curves(models_probabilities, y_test, auc_scores)

    # Model comparison
    accuracies = {
        "Logistic Regression": log_acc,
        "Naive Bayes": nb_acc,
        "Decision Tree": dt_acc,
        "Random Forest": rf_acc
    }

    results = create_results_table(accuracies, auc_scores)

    print("\n============================")
    print("Model Comparison")
    print("============================")
    print(results)

    # Cross Validation
    models = {
        "Logistic Regression": log_model,
        "Naive Bayes": nb_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }

    run_cross_validation(models, X, y)

    # Grid Search CV for Random Forest
    tune_random_forest(X_train, y_train)

    # Feature Importance
    plot_feature_importance(rf_model, X.columns)


if __name__ == "__main__":
    main()
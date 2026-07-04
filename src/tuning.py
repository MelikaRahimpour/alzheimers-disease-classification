from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from config import RANDOM_STATE, CV_FOLDS


def tune_random_forest(X_train, y_train):
    print("\n============================")
    print("Grid Search CV")
    print("============================")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:")
    print(grid.best_params_)

    print("\nBest Cross Validation Score:")
    print(grid.best_score_)

    return grid
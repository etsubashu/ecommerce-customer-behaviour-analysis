from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_model(X_train, y_train):

    params = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        params,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_
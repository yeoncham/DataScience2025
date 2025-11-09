from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_hyperparameters(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )
    grid.fit(x_train, y_train)
    return grid.best_estimator_, grid.best_params_

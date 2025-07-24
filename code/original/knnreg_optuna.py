from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor

import numpy as np
import optuna
import sys
import joblib

X_train = np.load(sys.argv[1])
y_train = np.load(sys.argv[2])
X_test = np.load(sys.argv[3])
y_test = np.load(sys.argv[4])
output_dir = sys.argv[5]

def objective(trial):

    n_neighbours = trial.suggest_int("number_neighbours", 2, 85)

    weights = trial.suggest_categorical("weights", ['uniform', 'distance'])

    p_num = trial.suggest_int("p", 1, 2)

    knn_reg = KNeighborsRegressor(n_neighbors=n_neighbours, weights=weights, p=p_num)
    trained_model = knn_reg.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)
    test_loss = mean_absolute_percentage_error(y_pred, y_test)

    return test_loss  # Optuna minimizes this

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

best_model = KNeighborsRegressor(**study.best_params)
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)
best_test_loss = mean_absolute_percentage_error(best_pred, y_test)
joblib.dump(best_model, f"{output_dir}/best_model.pkl")

# Save study for later visualization
joblib.dump(study, f"{output_dir}/optuna_study.pkl")

import json

dataset_name = sys.argv[6]
imputation_method = sys.argv[7]

summary = {
    "dataset": dataset_name,
    "imputation": imputation_method,
    "model": 'kNN Regressor',
    "best_params": study.best_params,
    "validation_accuracy": best_test_loss,
}

with open(f"{output_dir}/results.json", "w") as f:
    json.dump(summary, f, indent=2)
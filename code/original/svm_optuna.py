from sklearn.metrics import mean_absolute_percentage_error
from sklearn import svm

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

    degree = trial.suggest_int("_polydegree", 2, 10)

    kernel = trial.suggest_categorical("kernel", ['rbf', 'poly'])

    epsilon = trial.suggest_float("epsilon", 0.05, 0.5)

    regulariser = trial.suggest_float("regulariser", 0.5, 1)
    #print(degree, 'degree', kernel, 'kernel', epsilon, 'epsilon', regulariser, 'regulariser')

    svm_model = svm.SVR(degree=degree, kernel=kernel, epsilon=epsilon, C=regulariser)
    trained_model = svm_model.fit(X_train, y_train)
    y_pred = trained_model.predict(X_test)
    test_loss = mean_absolute_percentage_error(y_pred, y_test)

    return test_loss  # Optuna minimizes this

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

best_model = svm.SVR(**study.best_params)
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
    "model": 'support vector machine',
    "best_params": study.best_params,
    "validation_accuracy": best_test_loss,
}

with open(f"{output_dir}/results.json", "w") as f:
    json.dump(summary, f, indent=2)
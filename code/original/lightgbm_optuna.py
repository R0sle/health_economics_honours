from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import numpy as np
import optuna
import sys
import joblib
import pandas as pd

X_tv = pd.read_csv(sys.argv[1])
X_tv = X_tv.drop('Unnamed: 0', axis=1)
y_tv = pd.read_csv(sys.argv[2])
y_tv = y_tv.drop('Unnamed: 0', axis=1)
cv = pd.read_pickle(sys.argv[3])
output_dir = sys.argv[4]
proportion = sys.argv[5]

if proportion == '95':
    crossval = cv.iloc[0]
elif proportion == '100':
    crossval = cv.iloc[1]

def objective(trial, train_data, train_labels, val_data, val_labels):

    n_trees = trial.suggest_int("number_trees", 10, 200)

    max_depth = trial.suggest_int("max_tree_depth", 5, 40)

    boosting_type = trial.suggest_categorical("boosting_type", ['gbdt', 'dart'])

    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
    l1_norm = trial.suggest_float("l1_norm", 0, 0.001)
    l2_norm = trial.suggest_float("l2_norm", 0, 0.001)

    lgbm = lgb.LGBMRegressor(n_estimators=n_trees, random_state=42, boosting_type=boosting_type, max_depth=max_depth, learning_rate=learning_rate, reg_alpha=l1_norm, reg_lambda=l2_norm, verbosity=-1)
    trained_model = lgbm.fit(train_data, train_labels)
    y_pred = trained_model.predict(val_data)
    test_loss = mean_squared_error(y_pred, val_labels)

    return test_loss  # Optuna minimizes this

for fold in range(0, 5):
    #crossval dictionary = split_number : [train_index, val_index, groups, ratio]
    train_index = crossval.iloc[fold][0]
    val_index = crossval.iloc[fold][1]
    train_data = X_tv.iloc[train_index]
    train_labels = y_tv.iloc[train_index]
    val_data = X_tv.iloc[val_index]
    val_labels = y_tv.iloc[val_index]

    #Create a study object and optimize the objective function.
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, train_labels, val_data, val_labels), n_trials=300)

    best_model = lgb.LGBMRegressor(**study.best_params)
    best_model.fit(train_data, train_labels)
    #val_data_cuda = xgb.DMatrix(val_data, device='cuda')
    best_pred = best_model.predict(val_data)
    best_test_loss = mean_squared_error(best_pred, val_labels)
    #best_test_loss = mean_absolute_percentage_error(best_pred, val_labels)
    joblib.dump(best_model, f"{output_dir}/best_model_{fold}.pkl")

    # Save study for later visualization
    joblib.dump(study, f"{output_dir}/optuna_study_{fold}.pkl")

    import json

    dataset_name = sys.argv[6]
    imputation_method = sys.argv[7]

    summary = {
        "dataset": dataset_name,
        "fold" : fold,
        "imputation": imputation_method,
        "model": 'lightgbm',
        "best_params": study.best_params,
        "validation_accuracy": best_test_loss,
    }

    with open(f"{output_dir}/results_{fold}.json", "w") as f:
        json.dump(summary, f, indent=2)





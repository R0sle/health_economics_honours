from sklearn.ensemble import RandomForestRegressor
import optuna.visualization as vis

import numpy as np
import optuna
import sys
import joblib
import pandas as pd
import json

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

X_train = pd.read_csv('split_year_data/train/X_train_0_1.csv')
train_x = X_train.drop('Unnamed: 0', axis=1)

y_train = pd.read_csv('split_year_data/train/y_train_0_1.csv')
train_y = y_train.drop('Unnamed: 0', axis=1)

X_val = pd.read_csv('split_year_data/val/X_val_0.csv')
validation = X_val.drop('Unnamed: 0', axis=1)

y_val = pd.read_csv('split_year_data/val/y_val_0.csv')
validation_y = y_val.drop('Unnamed: 0', axis=1)

output_dir = 'split_year_models/rf'

def objective(trial, x_train, y_train, x_val, y_val):

    n_trees = trial.suggest_int("n_estimators", 10, 300)

    max_depth = trial.suggest_int("max_depth", 3, 25)

    min_sample_split = trial.suggest_int("min_samples_split", 2, 10)

    bootstrapping = trial.suggest_categorical("bootstrap", [True, False])
    if bootstrapping == False:
        max_samples = None
    else:
        max_samples = trial.suggest_float("max_samples", 0.01, 1.0)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1, min_samples_split=min_sample_split, bootstrap=bootstrapping, max_samples=max_samples, n_estimators=n_trees, max_depth=max_depth)
    trained_model = rf.fit(x_train, y_train)
    y_pred = trained_model.predict(x_val)
    val_loss = mean_squared_error(y_pred, y_val)

    return val_loss  # Optuna minimizes this

val_input_data = validation.copy()
val_label = validation_y.copy()

train_input_data = train_x.copy()
train_label = train_y.copy()

val_relevant_input = val_input_data[train_input_data.columns]

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_input_data, train_label, val_relevant_input, val_label), n_trials=1000)

#save best model 
joblib.dump(study.best_params, f"{output_dir}/best_params_0_1.pkl")

# Save study for later visualization
joblib.dump(study, f"{output_dir}/optuna_study_0_1.pkl")

summary = {
    "dataset": '0_1',
    "fold" : '0',
    "threshold": '1',
    "model": 'rf',
    "best_params": study.best_params,
    "best_optuna_loss": study.best_value
}

with open(f"{output_dir}/results_0_1.json", "w") as f:
    json.dump(summary, f, indent=2)
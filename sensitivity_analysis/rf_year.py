from sklearn.ensemble import RandomForestRegressor
import optuna.visualization as vis

import numpy as np
import optuna
import sys
import joblib
import pandas as pd
import json
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

features = sys.argv[1]
sensitivity = sys.argv[2]
thresh = sys.argv[3]
fold = sys.argv[4]
train_data_file = 'year_data/' + sensitivity + '_year.pkl'
with open(train_data_file, 'rb') as f:
    train_data = joblib.load(f)

if thresh == '85':
    thresh_idx = 0
elif thresh == '90':
    thresh_idx = 2
elif thresh == '95':
    thresh_idx = 4
elif thresh == '1':
    thresh_idx = 6

train_x = train_data[int(fold)][features][thresh_idx]
train_x = train_x.drop(columns=['index'], axis=1)
train_y = train_data[int(fold)][features][thresh_idx + 1]

val_data_file = 'year_data/val_' + sensitivity + '_year.pkl'
with open(val_data_file, 'rb') as vf:
    val_data = joblib.load(vf)

val_x = val_data[int(fold)][0]
val_y = val_data[int(fold)][1]

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

val_input_data = val_x.copy()
val_label = val_y.copy()

train_input_data = train_x.copy()
train_label = train_y.copy()

val_relevant_input = val_input_data[train_input_data.columns]

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_input_data, train_label, val_relevant_input, val_label), n_trials=1000)

#save best model 
joblib.dump(study.best_params, f"{sensitivity}_year/rf/best_params_{fold}_{thresh}_{features}.pkl")

# Save study for later visualization
joblib.dump(study, f"{sensitivity}_year/rf/optuna_study_{fold}_{thresh}_{features}.pkl")

summary = {
    "features": features,
    "fold" : fold,
    "threshold": thresh,
    "sensitivity": sensitivity,
    "model": 'rf',
    "best_params": study.best_params,
    "best_optuna_loss": study.best_value
}

with open(f"{sensitivity}_year/rf/results_{fold}_{thresh}_{features}.json", "w") as f:
    json.dump(summary, f, indent=2)
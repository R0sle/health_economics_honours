import lightgbm as lgb
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
train_data_file = 'sensitivity_analysis/year_data/' + sensitivity + '_year.pkl'
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

val_data_file = 'sensitivity_analysis/year_data/val_' + sensitivity + '_year.pkl'
with open(val_data_file, 'rb') as vf:
    val_data = joblib.load(vf)

val_x = val_data[int(fold)][0]
val_y = val_data[int(fold)][1]

def objective(trial, x_train, y_train, x_val, y_val):

    n_trees = trial.suggest_int("number_trees", 10, 300)

    max_depth = trial.suggest_int("max_tree_depth", 3, 25)

    boosting_type = trial.suggest_categorical("boosting_type", ['gbdt', 'dart'])
    
    bagging_fraction = trial.suggest_float("bagging_fraction", 0.1, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 0, 10)

    learning_rate = trial.suggest_float("learning_rate", 0, 1)
    l1_norm = trial.suggest_float("l1_norm", 0, 0.001)
    l2_norm = trial.suggest_float("l2_norm", 0, 0.001)

    lgbm_model = lgb.LGBMRegressor(random_state=42, verbosity = -1, n_estimators=n_trees, boosting=boosting_type, max_depth=max_depth, learning_rate=learning_rate, reg_alpha=l1_norm, reg_lambda=l2_norm, bagging_fraction=bagging_fraction, bagging_freq=bagging_freq)
    trained_model = lgbm_model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_val)
    val_loss = mean_squared_error(y_pred, y_val)

    return val_loss  # Optuna minimizes this

val_input_data = val_x.copy()
val_input_data.columns = val_input_data.columns.str.replace(r'[\"\[\]\{\}\\:,]', '', regex=True)
val_label = val_y.copy()
val_label.column = 'Maternal mortality ratio (national estimate per 100000 live births)'

train_input_data = train_x.copy()
train_input_data.columns = train_input_data.columns.str.replace(r'[\"\[\]\{\}\\:,]', '', regex=True)

train_label = train_y.copy()
train_label.column = 'Maternal mortality ratio (national estimate per 100000 live births)'

val_relevant_input = val_input_data[train_input_data.columns]

#Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_input_data, train_label, val_relevant_input, val_label), n_trials=1000)

#save best model 
joblib.dump(study.best_params, f"sensitivity_analysis/{sensitivity}_year/lgb/best_params_{fold}_{thresh}_{features}.pkl")

# Save study for later visualization
joblib.dump(study, f"sensitivity_analysis/{sensitivity}_year/lgb/optuna_study_{fold}_{thresh}_{features}.pkl")

summary = {
    "features": features,
    "fold" : fold,
    "threshold": thresh,
    "sensitivity": sensitivity,
    "model": 'lightgbm',
    "best_params": study.best_params,
    "best_optuna_loss": study.best_value
}

with open(f"sensitivity_analysis/{sensitivity}_year/lgb/results_{fold}_{thresh}_{features}.json", "w") as f:
    json.dump(summary, f, indent=2)
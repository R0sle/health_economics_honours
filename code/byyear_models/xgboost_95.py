import xgboost as xgb 
import optuna.visualization as vis

import numpy as np
import optuna
import sys
import joblib
import pandas as pd
import json
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

X_train_0 = pd.read_csv(sys.argv[1])
X_train_0 = X_train_0.drop('Unnamed: 0', axis=1)

X_train_1 = pd.read_csv(sys.argv[2])
X_train_1 = X_train_1.drop('Unnamed: 0', axis=1)

X_train_2 = pd.read_csv(sys.argv[3])
X_train_2 = X_train_2.drop('Unnamed: 0', axis=1)

X_train_3 = pd.read_csv(sys.argv[4])
X_train_3 = X_train_3.drop('Unnamed: 0', axis=1)

X_train_4 = pd.read_csv(sys.argv[5])
X_train_4 = X_train_4.drop('Unnamed: 0', axis=1)

y_train_0 = pd.read_csv(sys.argv[6])
y_train_0 = y_train_0.drop('Unnamed: 0', axis=1)

y_train_1 = pd.read_csv(sys.argv[7])
y_train_1 = y_train_1.drop('Unnamed: 0', axis=1)

y_train_2 = pd.read_csv(sys.argv[8])
y_train_2 = y_train_2.drop('Unnamed: 0', axis=1)

y_train_3 = pd.read_csv(sys.argv[9])
y_train_3 = y_train_3.drop('Unnamed: 0', axis=1)

y_train_4 = pd.read_csv(sys.argv[10])
y_train_4 = y_train_4.drop('Unnamed: 0', axis=1)

X_val_0 = pd.read_csv(sys.argv[11])
X_val_0 = X_val_0.drop('Unnamed: 0', axis=1)

X_val_1 = pd.read_csv(sys.argv[12])
X_val_1 = X_val_1.drop('Unnamed: 0', axis=1)

X_val_2 = pd.read_csv(sys.argv[13])
X_val_2 = X_val_2.drop('Unnamed: 0', axis=1)

X_val_3 = pd.read_csv(sys.argv[14])
X_val_3 = X_val_3.drop('Unnamed: 0', axis=1)

X_val_4 = pd.read_csv(sys.argv[15])
X_val_4 = X_val_4.drop('Unnamed: 0', axis=1)

y_val_0 = pd.read_csv(sys.argv[16])
y_val_0 = y_val_0.drop('Unnamed: 0', axis=1)

y_val_1 = pd.read_csv(sys.argv[17])
y_val_1 = y_val_1.drop('Unnamed: 0', axis=1)

y_val_2 = pd.read_csv(sys.argv[18])
y_val_2 = y_val_2.drop('Unnamed: 0', axis=1)

y_val_3 = pd.read_csv(sys.argv[19])
y_val_3 = y_val_3.drop('Unnamed: 0', axis=1)

y_val_4 = pd.read_csv(sys.argv[20])
y_val_4 = y_val_4.drop('Unnamed: 0', axis=1)

X_test = pd.read_csv(sys.argv[21])
X_test = X_test.drop('Unnamed: 0', axis=1)

y_test = pd.read_csv(sys.argv[22])
y_test = y_test.drop('Unnamed: 0', axis=1)

output_dir = sys.argv[23]
countries_dict_file = sys.argv[24]

with open(countries_dict_file, 'rb') as f:
    countries_dict = pickle.load(f)

def objective(trial, x_train, y_train, x_val, y_val):

    n_trees = trial.suggest_int("number_trees", 10, 300)

    max_depth = trial.suggest_int("max_tree_depth", 3, 25)

    boosting_type = trial.suggest_categorical("boosting_type", ['gbtree', 'dart'])
    subsample = trial.suggest_float("subsample", 0.1, 1)

    learning_rate = trial.suggest_float("learning_rate", 0, 1)
    l1_norm = trial.suggest_float("l1_norm", 0, 0.001)
    l2_norm = trial.suggest_float("l2_norm", 0, 0.001)

    xgb_model = xgb.XGBRegressor(enable_categorical=True, device='cuda', missing=np.nan, random_state=42, n_estimators=n_trees, booster=boosting_type, max_depth=max_depth, learning_rate=learning_rate, reg_alpha=l1_norm, reg_lambda=l2_norm, sampling_method="uniform", subsample=subsample)
    trained_model = xgb_model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_val)
    val_loss = mean_squared_error(y_pred, y_val)

    return val_loss  # Optuna minimizes this

X_test['setting'] = X_test['setting'].astype("category")
validation = [X_val_0, X_val_1, X_val_2, X_val_3, X_val_4]
validation_y = [y_val_0, y_val_1, y_val_2, y_val_3, y_val_4]

train_x = [X_train_0, X_train_1, X_train_2, X_train_3, X_train_4]
train_y = [y_train_0, y_train_1, y_train_2, y_train_3, y_train_4]

for fold in range(0, 5):
    val_input_data = validation[fold].copy()
    val_input_data['setting'] = val_input_data['setting'].astype("category")

    for thresh_idx, thresh in enumerate(['85', '95', '1']):
        if thresh_idx != 1:
            continue

        train_input_data = train_x[fold].copy()
        train_input_data['setting'] = train_input_data['setting'].astype("category")

        columns_needed = val_input_data.columns.intersection(train_input_data.columns)
        val_relevant_input = val_input_data[columns_needed]

        #Create a study object and optimize the objective function.
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, train_input_data, train_y[fold], val_relevant_input, validation_y[fold]), n_trials=300)
        best_model = xgb.XGBRegressor(**study.best_params, enable_categorical=True)
        best_model.fit(train_input_data, train_y[fold])
        
        #save best model 
        best_model.save_model(output_dir + '/best_model_' + str(fold) + '_' + thresh +  '.json')
        joblib.dump(study.best_params, f"{output_dir}/best_params_{fold}_{thresh}.pkl")

        # Save study for later visualization
        joblib.dump(study, f"{output_dir}/optuna_study_{fold}_{thresh}.pkl")

        summary = {
            "dataset": str(fold) + '_' + thresh,
            "fold" : fold,
            "threshold": thresh,
            "model": 'xgboost',
            "best_params": study.best_params,
            "best_optuna_loss": study.best_value
        }

        with open(f"{output_dir}/results_{fold}_{thresh}.json", "w") as f:
            json.dump(summary, f, indent=2)
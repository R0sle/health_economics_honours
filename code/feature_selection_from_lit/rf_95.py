from sklearn.ensemble import RandomForestRegressor
import optuna.visualization as vis
import pickle

import numpy as np
import optuna
import sys
import joblib
import pandas as pd
import json

from sklearn.metrics import mean_squared_error

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

output_dir = sys.argv[21]

countries_dict_file = sys.argv[22]

with open(countries_dict_file, 'rb') as f:
    countries_dict = pickle.load(f)

def objective(trial, x_train, y_train, x_val, y_val):

    n_trees = trial.suggest_int("n_estimators", 10, 300)

    max_depth = trial.suggest_int("max_depth", 3, 25)

    min_sample_split = trial.suggest_int("min_samples_split", 2, 10)

    bootstrapping = trial.suggest_categorical("bootstrap", [True, False])
    if bootstrapping == False:
        max_samples = None
    else:
        max_samples = trial.suggest_float("max_samples", 0.01, 1.0)

    rf = RandomForestRegressor(random_state=42, min_samples_split=min_sample_split, bootstrap=bootstrapping, max_samples=max_samples, n_estimators=n_trees, max_depth=max_depth)
    trained_model = rf.fit(x_train, y_train)
    y_pred = trained_model.predict(x_val)
    val_loss = mean_squared_error(y_pred, y_val)

    return val_loss  # Optuna minimizes this

validation = [X_val_0, X_val_1, X_val_2, X_val_3, X_val_4]
validation_y = [y_val_0, y_val_1, y_val_2, y_val_3, y_val_4]

train_x = [X_train_0, X_train_1, X_train_2, X_train_3, X_train_4]
train_y = [y_train_0, y_train_1, y_train_2, y_train_3, y_train_4]

for fold in range(0, 5):
        val_input_data = validation[fold].copy()
        val_input_data['setting'] = val_input_data['setting'].map(countries_dict)
        val_label = validation_y[fold].copy()

        for thresh_idx, thresh in enumerate(['85', '95', '1']):
            if thresh_idx != 1:
                continue
            
            train_input_data = train_x[fold].copy()    
            train_input_data['setting'] = train_input_data['setting'].map(countries_dict) 
            train_label = train_y[fold].copy()

            val_relevant_input = val_input_data[train_input_data.columns]

            #Create a study object and optimize the objective function.
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, train_input_data, train_label, val_relevant_input, val_label), n_trials=300)
            best_model = RandomForestRegressor(**study.best_params)
            best_model.fit(train_input_data, train_label)
            
            #save best model 
            joblib.dump(best_model, output_dir + '/best_model_' + str(fold) + '_' + thresh +  '.pkl')
            joblib.dump(study.best_params, f"{output_dir}/best_params_{fold}_{thresh}.pkl")

            # Save study for later visualization
            joblib.dump(study, f"{output_dir}/optuna_study_{fold}_{thresh}.pkl")

            summary = {
                "dataset": str(fold) + '_' + thresh,
                "fold" : fold,
                "threshold": thresh,
                "model": 'rf',
                "best_params": study.best_params,
                "best_optuna_loss": study.best_value
            }

            with open(f"{output_dir}/results_{fold}_{thresh}.json", "w") as f:
                json.dump(summary, f, indent=2)
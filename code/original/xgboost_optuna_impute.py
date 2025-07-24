from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.impute import KNNImputer

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
X_test = pd.read_csv(sys.argv[8])
X_test = X_test.drop('Unnamed: 0', axis=1)

if proportion == '95':
    crossval = cv.iloc[0]
elif proportion == '100':
    crossval = cv.iloc[1]

def objective(trial, fold_x_train, fold_y_train, fold_x_val, fold_y_val):

    n_trees = trial.suggest_int("number_trees", 10, 200)

    max_depth = trial.suggest_int("max_tree_depth", 5, 40)

    boosting_type = trial.suggest_categorical("boosting_type", ['gbtree', 'dart'])

    learning_rate = trial.suggest_float("learning_rate", 0, 1)
    l1_norm = trial.suggest_float("l1_norm", 0, 0.001)
    l2_norm = trial.suggest_float("l2_norm", 0, 0.001)

    xgb_model = xgb.XGBRegressor(missing=np.nan, device='cuda', random_state=42, n_estimators=n_trees, booster=boosting_type, max_depth=max_depth, learning_rate=learning_rate, reg_alpha=l1_norm, reg_lambda=l2_norm)
    trained_model = xgb_model.fit(fold_x_train, fold_y_train)
    val_data_cuda = xgb.DMatrix(fold_x_val, device='cuda')
    y_pred = trained_model.predict(val_data_cuda)
    test_loss = mean_squared_error(y_pred, fold_y_val)
    #test_loss = mean_absolute_percentage_error(y_pred, fold_y_val)

    return test_loss  # Optuna minimizes this


for fold in range(0, 5):
    #crossval dictionary = split_number : [train_index, val_index, groups, ratio]
    train_index = crossval.iloc[fold][0]
    val_index = crossval.iloc[fold][1]
    train_data = X_tv.iloc[train_index]
    train_alllabels = y_tv.iloc[train_index]
    val_data = X_tv.iloc[val_index]
    val_alllabels = y_tv.iloc[val_index]

    #to impute the X data 
    imputer_knn = KNNImputer(n_neighbors=5, keep_empty_features=True)
    train_data_imputed = imputer_knn.fit_transform(train_data)
    val_imputed = imputer_knn.transform(val_data)
    test_imputed = imputer_knn.transform(X_test)

    #converting the national and modelled training y-values to a single MMR estimate 
    mean_model_train = train_alllabels['Maternal mortality ratio (modeled estimate, per 100,000 live births)'].mean(skipna=True)
    std_model_train = train_alllabels['Maternal mortality ratio (modeled estimate, per 100,000 live births)'].std(skipna=True)
    standard_model_train = (train_alllabels['Maternal mortality ratio (modeled estimate, per 100,000 live births)'] - mean_model_train) / std_model_train
    mean_national_train = train_alllabels['Maternal mortality ratio (national estimate, per 100,000 live births)'].mean(skipna=True)
    std_national_train = train_alllabels['Maternal mortality ratio (national estimate, per 100,000 live births)'].std(skipna=True)
    standard_national_train = (train_alllabels['Maternal mortality ratio (national estimate, per 100,000 live births)'] - mean_national_train) / std_national_train
    #merging the standardised versions of the national and modelled estimates
    standardised_merged_train = pd.DataFrame(standard_model_train).join(pd.DataFrame(standard_national_train))
    standardised_merged_train = standardised_merged_train.reset_index(drop=True)
    standardised_merged_train = standardised_merged_train.drop('index', axis=1)
    init_train_labels = standardised_merged_train.mean(axis=1)
    init_train_labels = pd.DataFrame(init_train_labels)
    init_train_labels.columns = ['MMR Ratio']
    train_labels = (init_train_labels * std_model_train) + mean_model_train


    #converting the national and modelled validation y-values to a single MMR estimate 
    mean_model_val = val_alllabels['Maternal mortality ratio (modeled estimate, per 100,000 live births)'].mean(skipna=True)
    std_model_val = val_alllabels['Maternal mortality ratio (modeled estimate, per 100,000 live births)'].std(skipna=True)
    standard_model_val = (val_alllabels['Maternal mortality ratio (modeled estimate, per 100,000 live births)'] - mean_model_val) / std_model_val
    mean_national_val = val_alllabels['Maternal mortality ratio (national estimate, per 100,000 live births)'].mean(skipna=True)
    std_national_val = val_alllabels['Maternal mortality ratio (national estimate, per 100,000 live births)'].std(skipna=True)
    standard_national_val = (val_alllabels['Maternal mortality ratio (national estimate, per 100,000 live births)'] - mean_national_val) / std_national_val
    #merging the standardised versions of the national and modelled estimates
    standardised_merged_val = pd.DataFrame(standard_model_val).join(pd.DataFrame(standard_national_val))
    standardised_merged_val = standardised_merged_val.reset_index(drop=True)
    standardised_merged_val = standardised_merged_val.drop('index', axis=1)
    init_val_labels = standardised_merged_val.mean(axis=1)
    init_val_labels = pd.DataFrame(init_val_labels)
    init_val_labels.columns = ['MMR Ratio']
    val_labels = (init_val_labels * std_model_val) + mean_model_val


    #Create a study object and optimize the objective function.
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data_imputed, train_labels, val_imputed, val_labels), n_trials=300)
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(train_data_imputed, train_labels)

    val_data_cuda = xgb.DMatrix(val_imputed, device='cuda')
    best_pred = best_model.predict(val_data_cuda)
    best_val_loss = mean_squared_error(best_pred, val_labels)

    test_data_cuda = xgb.DMatrix(test_imputed, device='cuda')
    y_test_labels = pd.read_csv(sys.argv[9])
    y_test_labels = y_test_labels.drop('Unnamed: 0', axis=1)
    
    best_pred_test = best_model.predict(test_data_cuda)
    best_test_loss = mean_squared_error(best_pred_test, y_test_labels)
    
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
        "model": 'xgboost',
        "best_params": study.best_params,
        "test_mse": best_test_loss,
        "validation_mse": best_val_loss
    }

    with open(f"{output_dir}/results_{fold}.json", "w") as f:
        json.dump(summary, f, indent=2)
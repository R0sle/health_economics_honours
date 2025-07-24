#!/bin/bash
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=380GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P dx61 
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/te06 
#PBS -l wd

module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

for method in knn knn_std mf mf_std p1 p1_std p2 p2_std p3 p3_std std no; do
        python3 scripts_inc/xgboost_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_95.csv split_income_data/imputed/${method}/train/y_train_uncorr_95.csv split_income_data/imputed/${method}/val/X_val_uncorr_95.csv split_income_data/imputed/${method}/val/y_val_uncorr_95.csv results_inc/xgboost/${method}/uncorr_95 uncorr_95 ${method}
        python3 scripts_inc/xgboost_optuna.py split_income_data/imputed/${method}/train/X_train_corr_95.csv split_income_data/imputed/${method}/train/y_train_corr_95.csv split_income_data/imputed/${method}/val/X_val_corr_95.csv split_income_data/imputed/${method}/val/y_val_corr_95.csv results_inc/xgboost/${method}/corr_95 corr_95 ${method}    

        python3 scripts_inc/lightgbm_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_95.csv split_income_data/imputed/${method}/train/y_train_uncorr_95.csv split_income_data/imputed/${method}/val/X_val_uncorr_95.csv split_income_data/imputed/${method}/val/y_val_uncorr_95.csv results_inc/lightgbm/${method}/uncorr_95 uncorr_95 ${method}
        python3 scripts_inc/lightgbm_optuna.py split_income_data/imputed/${method}/train/X_train_corr_95.csv split_income_data/imputed/${method}/train/y_train_corr_95.csv split_income_data/imputed/${method}/val/X_val_corr_95.csv split_income_data/imputed/${method}/val/y_val_corr_95.csv results_inc/lightgbm/${method}/corr_95 corr_95 ${method}   

        python3 scripts_inc/adaboost_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_95.csv split_income_data/imputed/${method}/train/y_train_uncorr_95.csv split_income_data/imputed/${method}/val/X_val_uncorr_95.csv split_income_data/imputed/${method}/val/y_val_uncorr_95.csv results_inc/adaboost/${method}/uncorr_95 uncorr_95 ${method}
        python3 scripts_inc/adaboost_optuna.py split_income_data/imputed/${method}/train/X_train_corr_95.csv split_income_data/imputed/${method}/train/y_train_corr_95.csv split_income_data/imputed/${method}/val/X_val_corr_95.csv split_income_data/imputed/${method}/val/y_val_corr_95.csv results_inc/adaboost/${method}/corr_95 corr_95 ${method} 

        python3 scripts_inc/svm_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_95.csv split_income_data/imputed/${method}/train/y_train_uncorr_95.csv split_income_data/imputed/${method}/val/X_val_uncorr_95.csv split_income_data/imputed/${method}/val/y_val_uncorr_95.csv results_inc/svm/${method}/uncorr_95 uncorr_95 ${method}
        python3 scripts_inc/svm_optuna.py split_income_data/imputed/${method}/train/X_train_corr_95.csv split_income_data/imputed/${method}/train/y_train_corr_95.csv split_income_data/imputed/${method}/val/X_val_corr_95.csv split_income_data/imputed/${method}/val/y_val_corr_95.csv results_inc/svm/${method}/corr_95 corr_95 ${method}

        python3 scripts_inc/knnreg_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_95.csv split_income_data/imputed/${method}/train/y_train_uncorr_95.csv split_income_data/imputed/${method}/val/X_val_uncorr_95.csv split_income_data/imputed/${method}/val/y_val_uncorr_95.csv results_inc/knnreg/${method}/uncorr_95 uncorr_95 ${method}
        python3 scripts_inc/knnreg_optuna.py split_income_data/imputed/${method}/train/X_train_corr_95.csv split_income_data/imputed/${method}/train/y_train_corr_95.csv split_income_data/imputed/${method}/val/X_val_corr_95.csv split_income_data/imputed/${method}/val/y_val_corr_95.csv results_inc/knnreg/${method}/corr_95 corr_95 ${method}

        python3 scripts_inc/mlp_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_95.csv split_income_data/imputed/${method}/train/y_train_uncorr_95.csv split_income_data/imputed/${method}/val/X_val_uncorr_95.csv split_income_data/imputed/${method}/val/y_val_uncorr_95.csv results_inc/mlp/${method}/uncorr_95 uncorr_95 ${method}
        python3 scripts_inc/mlp_optuna.py split_income_data/imputed/${method}/train/X_train_corr_95.csv split_income_data/imputed/${method}/train/y_train_corr_95.csv split_income_data/imputed/${method}/val/X_val_corr_95.csv split_income_data/imputed/${method}/val/y_val_corr_95.csv results_inc/mlp/${method}/corr_95 corr_95 ${method}
    done

    for method in std no; do
        python3 scripts_inc/xgboost_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_100.csv split_income_data/imputed/${method}/train/y_train_uncorr_100.csv split_income_data/imputed/${method}/val/X_val_uncorr_100.csv split_income_data/imputed/${method}/val/y_val_uncorr_100.csv results_inc/xgboost/${method}/uncorr_100 uncorr_100 ${method}
        python3 scripts_inc/xgboost_optuna.py split_income_data/imputed/${method}/train/X_train_corr_100.csv split_income_data/imputed/${method}/train/y_train_corr_100.csv split_income_data/imputed/${method}/val/X_val_corr_100.csv split_income_data/imputed/${method}/val/y_val_corr_100.csv results_inc/xgboost/${method}/corr_100 corr_95 ${method}    

        python3 scripts_inc/lightgbm_optuna.py split_income_data/imputed/${method}/train/X_train_uncorr_100.csv split_income_data/imputed/${method}/train/y_train_uncorr_100.csv split_income_data/imputed/${method}/val/X_val_uncorr_100.csv split_income_data/imputed/${method}/val/y_val_uncorr_100.csv results_inc/lightgbm/${method}/uncorr_100 uncorr_100 ${method}
        python3 scripts_inc/lightgbm_optuna.py split_income_data/imputed/${method}/train/X_train_corr_100.csv split_income_data/imputed/${method}/train/y_train_corr_100.csv split_income_data/imputed/${method}/val/X_val_corr_100.csv split_income_data/imputed/${method}/val/y_val_corr_100.csv results_inc/lightgbm/${method}/corr_100 corr_100 ${method}   
    done

deactivate
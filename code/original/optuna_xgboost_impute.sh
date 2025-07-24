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
module load cuda
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

for method in knn; do

python3 scripts_inc/xgboost_optuna_cv.py split_income_data/train/X_train_uncorr_95.csv split_income_data/train/y_train_uncorr_95.csv split_income_data/train/uncorr_cv.csv results_inc/${method}/xgboost/uncorr_95 '95' uncorr_95 ${method} split_income_data/test/X_test_uncorr_95.csv split_income_data/test/y_test_uncorr_95_labels.csv
python3 scripts_inc/xgboost_optuna_cv.py split_income_data/train/X_train_corr_95.csv split_income_data/train/y_train_corr_95.csv split_income_data/train/corr_cv.csv results_inc/${method}/xgboost/corr_95 '95' corr_95 ${method} split_income_data/test/X_test_corr_95.csv split_income_data/test/y_test_corr_95_labels.csv
python3 scripts_inc/xgboost_optuna_cv.py split_income_data/train/X_train_uncorr_100.csv split_income_data/train/y_train_uncorr_100.csv split_income_data/train/uncorr_cv.csv results_inc/${method}/xgboost/uncorr_100 '100' uncorr_100 ${method} split_income_data/test/X_test_uncorr_100.csv split_income_data/test/y_test_uncorr_100_labels.csv
python3 scripts_inc/xgboost_optuna_cv.py split_income_data/train/X_train_corr_100.csv split_income_data/train/y_train_corr_100.csv split_income_data/train/corr_cv.csv results_inc/${method}/xgboost/corr_100 '100' corr_100 ${method} split_income_data/test/X_test_corr_100.csv split_income_data/test/y_test_corr_100_labels.csv

    done

deactivate
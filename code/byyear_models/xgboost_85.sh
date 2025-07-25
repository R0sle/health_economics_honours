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

python3 year_code/xgboost_85.py year_data/train/X_train_0_85.csv year_data/train/X_train_1_85.csv year_data/train/X_train_2_85.csv year_data/train/X_train_3_85.csv year_data/train/X_train_4_85.csv year_data/train/y_train_0_85.csv year_data/train/y_train_1_85.csv year_data/train/y_train_2_85.csv year_data/train/y_train_3_85.csv year_data/train/y_train_4_85.csv year_data/val/X_val_0.csv year_data/val/X_val_1.csv year_data/val/X_val_2.csv year_data/val/X_val_3.csv year_data/val/X_val_4.csv year_data/val/y_val_0.csv year_data/val/y_val_1.csv year_data/val/y_val_2.csv year_data/val/y_val_3.csv year_data/val/y_val_4.csv year_data/test/X_test.csv year_data/test/y_test.csv year_models/xgboost/ year_data/countries_dict.pkl

deactivate
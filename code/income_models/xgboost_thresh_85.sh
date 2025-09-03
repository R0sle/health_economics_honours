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

python3 inc_code/xgboost_script.py inc_data/train/X_train_0_85.csv inc_data/train/X_train_1_85.csv inc_data/train/X_train_2_85.csv inc_data/train/X_train_3_85.csv inc_data/train/X_train_4_85.csv inc_data/train/y_train_0_85.csv inc_data/train/y_train_1_85.csv inc_data/train/y_train_2_85.csv inc_data/train/y_train_3_85.csv inc_data/train/y_train_4_85.csv inc_data/val/X_val_0.csv inc_data/val/X_val_1.csv inc_data/val/X_val_2.csv inc_data/val/X_val_3.csv inc_data/val/X_val_4.csv inc_data/val/y_val_0.csv inc_data/val/y_val_1.csv inc_data/val/y_val_2.csv inc_data/val/y_val_3.csv inc_data/val/y_val_4.csv inc_models/xgboost/ 85

deactivate
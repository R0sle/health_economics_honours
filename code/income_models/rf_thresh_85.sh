#!/bin/bash
#PBS -l ncpus=20 
#PBS -l mem=80GB 
#PBS -l jobfs=10GB 
#PBS -q normal 
#PBS -P dx61 
#PBS -l walltime=24:00:00 
#PBS -l storage=scratch/te06 
#PBS -l wd 


module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

python3 inc_code/rf_thresh_85.py inc_data/train/X_train_0_85.csv inc_data/train/X_train_1_85.csv inc_data/train/X_train_2_85.csv inc_data/train/X_train_3_85.csv inc_data/train/X_train_4_85.csv inc_data/train/y_train_0_85.csv inc_data/train/y_train_1_85.csv inc_data/train/y_train_2_85.csv inc_data/train/y_train_3_85.csv inc_data/train/y_train_4_85.csv inc_data/val/X_val_0.csv inc_data/val/X_val_1.csv inc_data/val/X_val_2.csv inc_data/val/X_val_3.csv inc_data/val/X_val_4.csv inc_data/val/y_val_0.csv inc_data/val/y_val_1.csv inc_data/val/y_val_2.csv inc_data/val/y_val_3.csv inc_data/val/y_val_4.csv inc_data/test/X_test.csv inc_data/test/y_test.csv inc_models/rf/ inc_data/countries_dict.pkl

deactivate
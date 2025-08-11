#!/bin/bash
#PBS -l ncpus=20
#PBS -l mem=80GB 
#PBS -l jobfs=10GB 
#PBS -q normal 
#PBS -P dx61 
#PBS -l walltime=15:00:00 
#PBS -l storage=scratch/te06 
#PBS -l wd 


module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

python3 fs_fromlit_scripts/rf_85.py fs_fromlit_data/income/train/X_train_0_85.csv fs_fromlit_data/income/train/X_train_1_85.csv fs_fromlit_data/income/train/X_train_2_85.csv fs_fromlit_data/income/train/X_train_3_85.csv fs_fromlit_data/income/train/X_train_4_85.csv fs_fromlit_data/income/train/y_train_0_85.csv fs_fromlit_data/income/train/y_train_1_85.csv fs_fromlit_data/income/train/y_train_2_85.csv fs_fromlit_data/income/train/y_train_3_85.csv fs_fromlit_data/income/train/y_train_4_85.csv fs_fromlit_data/income/val/X_val_0.csv fs_fromlit_data/income/val/X_val_1.csv fs_fromlit_data/income/val/X_val_2.csv fs_fromlit_data/income/val/X_val_3.csv fs_fromlit_data/income/val/X_val_4.csv fs_fromlit_data/income/val/y_val_0.csv fs_fromlit_data/income/val/y_val_1.csv fs_fromlit_data/income/val/y_val_2.csv fs_fromlit_data/income/val/y_val_3.csv fs_fromlit_data/income/val/y_val_4.csv fs_fromlit_models/income/rf/ inc_data/countries_dict.pkl


deactivate
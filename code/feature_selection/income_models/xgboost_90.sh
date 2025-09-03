#!/bin/bash
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=380GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P dx61 
#PBS -l walltime=20:00:00
#PBS -l storage=scratch/te06 
#PBS -l wd


module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

for corr in 60 70 80; do
    python3 feature_scripts/xgboost_script.py fs_corr_data/income/train/$corr/X_train_0_90.csv fs_corr_data/income/train/$corr/X_train_1_90.csv fs_corr_data/income/train/$corr/X_train_2_90.csv fs_corr_data/income/train/$corr/X_train_3_90.csv fs_corr_data/income/train/$corr/X_train_4_90.csv fs_corr_data/income/train/$corr/y_train_0_90.csv fs_corr_data/income/train/$corr/y_train_1_90.csv fs_corr_data/income/train/$corr/y_train_2_90.csv fs_corr_data/income/train/$corr/y_train_3_90.csv fs_corr_data/income/train/$corr/y_train_4_90.csv fs_corr_data/income/val/$corr/X_val_0.csv fs_corr_data/income/val/$corr/X_val_1.csv fs_corr_data/income/val/$corr/X_val_2.csv fs_corr_data/income/val/$corr/X_val_3.csv fs_corr_data/income/val/$corr/X_val_4.csv fs_corr_data/income/val/$corr/y_val_0.csv fs_corr_data/income/val/$corr/y_val_1.csv fs_corr_data/income/val/$corr/y_val_2.csv fs_corr_data/income/val/$corr/y_val_3.csv fs_corr_data/income/val/$corr/y_val_4.csv feature_models/income/xgboost/ $corr 90
done

deactivate
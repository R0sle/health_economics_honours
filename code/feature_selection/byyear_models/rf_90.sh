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

for corr in 60 70 80; do
    python3 feature_scripts/rf.py fs_corr_data/year/train/$corr/X_train_0_90.csv fs_corr_data/year/train/$corr/X_train_1_90.csv fs_corr_data/year/train/$corr/X_train_2_90.csv fs_corr_data/year/train/$corr/X_train_3_90.csv fs_corr_data/year/train/$corr/X_train_4_90.csv fs_corr_data/year/train/$corr/y_train_0_90.csv fs_corr_data/year/train/$corr/y_train_1_90.csv fs_corr_data/year/train/$corr/y_train_2_90.csv fs_corr_data/year/train/$corr/y_train_3_90.csv fs_corr_data/year/train/$corr/y_train_4_90.csv fs_corr_data/year/val/$corr/X_val_0.csv fs_corr_data/year/val/$corr/X_val_1.csv fs_corr_data/year/val/$corr/X_val_2.csv fs_corr_data/year/val/$corr/X_val_3.csv fs_corr_data/year/val/$corr/X_val_4.csv fs_corr_data/year/val/$corr/y_val_0.csv fs_corr_data/year/val/$corr/y_val_1.csv fs_corr_data/year/val/$corr/y_val_2.csv fs_corr_data/year/val/$corr/y_val_3.csv fs_corr_data/year/val/$corr/y_val_4.csv feature_models/year/rf/ $corr 90
done

deactivate
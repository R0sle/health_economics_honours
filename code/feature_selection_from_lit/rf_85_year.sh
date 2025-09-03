#!/bin/bash
#PBS -l ncpus=40
#PBS -l mem=160GB 
#PBS -l jobfs=10GB 
#PBS -q normal 
#PBS -P dx61 
#PBS -l walltime=24:00:00 
#PBS -l storage=scratch/te06 
#PBS -l wd 


module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

python3 fs_fromlit_scripts/rf.py fs_fromlit_data/year/train/X_train_0_85.csv fs_fromlit_data/year/train/X_train_1_85.csv fs_fromlit_data/year/train/X_train_2_85.csv fs_fromlit_data/year/train/X_train_3_85.csv fs_fromlit_data/year/train/X_train_4_85.csv fs_fromlit_data/year/train/y_train_0_85.csv fs_fromlit_data/year/train/y_train_1_85.csv fs_fromlit_data/year/train/y_train_2_85.csv fs_fromlit_data/year/train/y_train_3_85.csv fs_fromlit_data/year/train/y_train_4_85.csv fs_fromlit_data/year/val/X_val_0.csv fs_fromlit_data/year/val/X_val_1.csv fs_fromlit_data/year/val/X_val_2.csv fs_fromlit_data/year/val/X_val_3.csv fs_fromlit_data/year/val/X_val_4.csv fs_fromlit_data/year/val/y_val_0.csv fs_fromlit_data/year/val/y_val_1.csv fs_fromlit_data/year/val/y_val_2.csv fs_fromlit_data/year/val/y_val_3.csv fs_fromlit_data/year/val/y_val_4.csv fs_fromlit_models/year/rf/ 85


deactivate
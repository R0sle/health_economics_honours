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

python3 year_code/rf.py year_data/train/X_train_3_90.csv year_data/train/y_train_3_90.csv year_data/val/X_val_3.csv year_data/val/y_val_3.csv year_models/rf/ 90 3

deactivate
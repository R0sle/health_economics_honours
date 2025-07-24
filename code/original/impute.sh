#!/bin/bash
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=96GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P dx61 
#PBS -l walltime=4:00:00
#PBS -l storage=scratch/te06 
#PBS -l wd

module load python3

for method in knn knn_std mf mf_std p1 p1_std p2 p2_std p3 p3_std std no; do
    for co in uncorr corr; do
        python3 scripts_inc/impute.py split_income_data/train/X_train_${co}_95.csv split_income_data/val/X_val_${co}_95.csv ${method} split_income_data/imputed/${method}/train/X_train_${co}_95.csv split_income_data/imputed/${method}/val/X_val_${co}_95.csv False split_income_data/test/X_test_${co}_95.csv split_income_data/imputed_test/${method}/X_test_${co}_95.csv 
        python3 scripts_inc/impute.py split_income_data/train/y_train_${co}_95.csv split_income_data/val/y_val_${co}_95.csv ${method} split_income_data/imputed/${method}/train/y_train_${co}_95.csv split_income_data/imputed/${method}/val/y_val_${co}_95.csv True split_income_data/test/y_test_${co}_95.csv split_income_data/imputed_test/${method}/y_test_${co}_95.csv 
    done
done

for method in std no; do
    for co in uncorr corr; do
        python3 scripts_inc/impute.py split_income_data/train/X_train_${co}_100.csv split_income_data/val/X_val_${co}_100.csv ${method} split_income_data/imputed/${method}/train/X_train_${co}_100.csv split_income_data/imputed/${method}/val/X_val_${co}_100.csv False split_income_data/test/X_test_${co}_100.csv split_income_data/imputed_test/${method}/X_test_${co}_100.csv 
        python3 scripts_inc/impute.py split_income_data/train/y_train_${co}_100.csv split_income_data/val/y_val_${co}_100.csv ${method} split_income_data/imputed/${method}/train/y_train_${co}_100.csv split_income_data/imputed/${method}/val/y_val_${co}_100.csv True split_income_data/test/y_test_${co}_100.csv split_income_data/imputed_test/${method}/y_test_${co}_100.csv 
    done
done


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

for method in knn knn_std p1 p1_std p2 p2_std p3 p3_std; do
        for data_type in X y; do
            for co in uncorr corr; do
                    python scripts_inc/impute_test.py split_income_data/test/${data_type}_test_${co}_95.parquet ${method} split_income_data/imputed/${method}/test/${data_type}_train_${co}_95.parquet split_income_data/imputed/${method}/val/${data_type}_val_${co}_95.parquet
            done
        done
done
#!/bin/bash
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=380GB
#PBS -l jobfs=200GB
#PBS -q gpuvolta
#PBS -P dx61 
#PBS -l walltime=30:00:00
#PBS -l storage=scratch/te06 
#PBS -l wd

module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

for features in lit
do
    for sensitivity in pre post
    do
        for thresh in 85 90 95 1
        do
            for fold in 0 1 2 3 4
            do
            python3 xgboost_year.py $features $sensitivity $thresh $fold

            done
        done
    done
done
deactivate

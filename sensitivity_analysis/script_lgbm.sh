#!/bin/bash
#PBS -l ncpus=4
#PBS -l mem=16GB 
#PBS -l jobfs=10GB 
#PBS -q normal 
#PBS -P dx61 
#PBS -l walltime=24:00:00 
#PBS -l storage=scratch/te06 
#PBS -l wd 

module load python3
source /scratch/dx61/rr4398/mmr/modenv/bin/activate

for features in full lit corr_60 corr_70 corr_80
do
    for sensitivity in low ul um high pre post
    do
        for thresh in 85 90 95 1
        do
            for fold in 0 1 2 3 4
            do
            python3 lgb_country.py $features $sensitivity $thresh $fold

            done
        done
    done
done

deactivate
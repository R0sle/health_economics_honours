#!/bin/bash


for features in corr_70
do
    for sensitivity in ul um high pre post
    do
        for thresh in 85 90 95 1
        do
            for fold in 0 1 2 3 4
            do
            python3 sensitivity_analysis/lgbm_year.py $features $sensitivity $thresh $fold

            done
        done
    done
done


#!/bin/bash


for features in corr_70
do
    for sensitivity in pre post
    do
        for thresh in 85 90 95 1
        do
            for fold in 0 1 2 3 4
            do
            python3 sensitivity_analysis/xgboost_modelcode.py $features $sensitivity $thresh $fold

            done
        done
    done
done

    
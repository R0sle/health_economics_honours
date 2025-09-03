#!/bin/bash


for features in corr_60
do
    for sensitivity in high
    do
        for thresh in 95
        do
            for fold in 4
            do
            python3 sensitivity_analysis/xgboost_modelcode.py $features $sensitivity $thresh $fold

            done
        done
    done
done

    
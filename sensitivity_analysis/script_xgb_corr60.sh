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
    for sensitivity in post
    do
        for thresh in 90 95 1
        do
            for fold in 0 1 2 3 4
            do
            python3 sensitivity_analysis/xgboost_modelcode.py $features $sensitivity $thresh $fold

            done
        done
    done
done

    
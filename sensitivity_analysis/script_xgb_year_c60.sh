#!/bin/bash

for features in corr_60
do
    for sensitivity in high
    do
        for thresh in 1
        do
            for fold in 1 2 3 4
            do
                python3 xgboost_year.py $features $sensitivity $thresh $fold

            done
        done
    done
done

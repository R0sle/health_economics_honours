#!/bin/bash


for features in corr_80
do
    for sensitivity in high
    do
        for thresh in 85
        do
            for fold in 0 1 2 3 4
            do
            python3 lgbm_year.py $features $sensitivity $thresh $fold

            done
        done
        for thresh in 90
        do
            for fold in 1 3 4
            do
            python3 lgbm_year.py $features $sensitivity $thresh $fold

            done
        done
    done
done


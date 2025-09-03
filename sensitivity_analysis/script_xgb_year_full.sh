#!/bin/bash

for features in full
do
    for sensitivity in um
    do
        for thresh in 95 1
        do
            for fold in 0 1 2 3 4
            do
                python3 xgboost_year.py $features $sensitivity $thresh $fold

            done
        done
    done
done

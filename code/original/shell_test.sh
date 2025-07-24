#!/bin/bash

for filepath in split_income_data/**/X*95.parquet; do
    echo "Found dataset: $filepath"
done
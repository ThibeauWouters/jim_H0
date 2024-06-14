#!/bin/bash

# List of numbers
numbers=(81 143 187 399 404 419 532 541 737 795 868 1017 1032 1064 1247 1316 1572 1654 1920 2097 2335 2786 2810 2949 3267 3554 3875 3904 3912 3938 4019 4139 4397 4543 4627 4748)

# Loop over the list and run the python command
for N in "${numbers[@]}"; do
    python injection_recovery.py \
        --outdir ./outdir/ \
        --n-loop-training 200 \
        --n-loop-production 20 \
        --N "$N" \
        --stopping-criterion-global-acc 0.40
done

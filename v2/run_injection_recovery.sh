#!/bin/bash

python injection_recovery.py \
    --outdir ./outdir/ \
    --N "GW170817" \
    --n-loop-training 100 \
    --n-loop-production 20 \
    --relative-binning-binsize 1000 \
    --stopping-criterion-global-acc 0.20
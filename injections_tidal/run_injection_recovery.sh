#!/bin/bash

python injection_recovery.py \
    --outdir ./outdir/ \
    --n-loop-training 200 \
    --n-loop-production 20 \
    --N "737" \
    --stopping-criterion-global-acc 0.40
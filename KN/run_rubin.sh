#!/bin/bash

# This script runs NMMA on the LCs. Just in a small .sh script to make it a bit more organized.

# Define which injection to do here
INJECTION_NUM=0

lightcurve-analysis \
    --model Bu2022Ye \
    --interpolation-type tensorflow \
    --outdir ./outdir/BNS/$INJECTION_NUM \
    --label injection_Bu2022Ye_$INJECTION_NUM \
    --prior ./Bu2022Ye.prior \
    --tmin 0 \
    --tmax 20 \
    --dt 0.5 \
    --error-budget 1 \
    --nlive 2048 \
    --Ebv-max 0 \
    --injection ./outdir/injection_Bu2022Ye.json \
    --injection-num $INJECTION_NUM \
    --injection-detection-limit 23.9,25.0,24.7,24.0,23.3,22.1 \
    --injection-outfile ./outdir/BNS/$INJECTION_NUM/lc.csv \
    --generation-seed 42 \
    --remove-nondetections \
    --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__y,ps1__z \
    --rubin-ToO-type BNS \
    --rubin-ToO \
    --local-only \
    --svd-path ./svdmodels \
    # --plot \
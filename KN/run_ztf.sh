#!/bin/bash

# This script runs NMMA on the LCs. Just in a small .sh script to make it a bit more organized.

### For a single injection: define which injection to do here
# INJECTION_NUM=0

for INJECTION_NUM in {0..31}; do
    echo "========== Running injection $INJECTION_NUM =========="
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
        --injection-detection-limit 21.7,21.4,20.9 \
        --injection-outfile ./outdir/BNS/$INJECTION_NUM/lc.csv \
        --generation-seed 42 \
        --remove-nondetections \
        --filters ztfg,ztfr,ztfi \
        --ztf-ToO 300  \
        --ztf-uncertainties \
        --ztf-sampling \
        --ztf-ToO 300 \
        --local-only \
        --svd-path ./svdmodels \
        --plot
done
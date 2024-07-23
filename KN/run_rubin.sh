#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for INJECTION_NUM in {0..20}; do
    OUTDIR=./outdir_Rubin/BNS/$INJECTION_NUM
    
    echo "========== Running injection $INJECTION_NUM =========="
    mpiexec -np 16 lightcurve-analysis \
        --model Bu2022Ye \
        --interpolation-type tensorflow \
        --outdir $OUTDIR \
        --label injection_Bu2022Ye_$INJECTION_NUM \
        --prior ./Bu2022Ye.prior \
        --tmin 0 \
        --tmax 20 \
        --dt 0.5 \
        --error-budget 1 \
        --nlive 2048 \
        --Ebv-max 0 \
        --generation-seed 43 \
        --injection ./outdir_Rubin/injection_Bu2022Ye.json \
        --injection-num $INJECTION_NUM \
        --injection-outfile ./outdir_Rubin/BNS/$INJECTION_NUM/lc.csv \
        --filters sdssu,ps1__g,ps1__r,ps1__i,ps1__y,ps1__z \
        --rubin-ToO-type BNS \
        --rubin-ToO \
        --svd-path ./svdmodels \
        --plot \
        --injection-detection-limit 23.9,25.0,24.7,24.0,23.3,22.1 \
        --local-only \
        # --ensure-detections
        # --remove-nondetections \
        # --injection-detection-limit None \ # for testing the setup, but this is not realistic
done
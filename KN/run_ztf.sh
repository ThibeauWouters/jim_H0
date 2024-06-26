#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for INJECTION_NUM in {2..29}; do
    echo "========== Running injection $INJECTION_NUM =========="
    mpiexec -np 16 lightcurve-analysis \
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
        --generation-seed 42 \
        --injection ./outdir/injection_Bu2022Ye.json \
        --injection-num $INJECTION_NUM \
        --injection-outfile ./outdir/BNS/$INJECTION_NUM/lc.csv \
        --filters ztfg,ztfr,ztfi \
        --ztf-sampling \
        --ztf-ToO 300 \
        --local-only \
        --svd-path ./svdmodels \
        --plot \
        # --ztf-uncertainties \
        # --remove-nondetections \
        # --injection-detection-limit None \ # TODO: for testing, remove later!
done
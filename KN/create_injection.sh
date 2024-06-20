#!/bin/bash

# This script performs the NMMA create injection. Just in a small .sh script to make it a bit more organized.

nmma-create-injection \
    --prior-file ./Bu2022Ye.prior \
    --injection-file ./injections.dat \
    --eos-file ../injections_tidal/36022_macroscopic.dat \
    --binary-type BNS \
    --extension json \
    -f ./outdir/injection_Bu2022Ye \
    --generation-seed 42 \
    --aligned-spin \
    --eject
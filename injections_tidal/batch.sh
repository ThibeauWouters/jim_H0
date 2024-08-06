#!/bin/bash

# Perform the batch of GW run, i.e., loop over outdir and do the injection for each N

outdir="./outdir_uniform"

# Loop over each subdirectory in the output directory
for dir in "$outdir"/injection_*; do
    # Extract the identifier (the number after "injection_")
    identifier=$(basename "$dir" | sed 's/injection_//')
    
    # Print the identifier to the screen
    echo "Processing injection with identifier: $identifier"
    
    # Call the original script with the extracted identifier
    python injection_recovery.py \
        --outdir "$outdir" \
        --n-loop-training 200 \
        --n-loop-production 20 \
        --N "$identifier" \
        --stopping-criterion-global-acc 0.40
done
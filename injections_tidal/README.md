The new results are in `outdir`, which had the powerlaw prior. A new set of runs with uniform prior will be saved to `outdir_uniform`.

The likeliest EOS from the Hauke et al review paper is taken from the Potsdam machines at `/home/aya/work/hkoehn/EOS_analysis/likeliest_eos/36022_macroscopic.dat` 

Note:
- The `wrong_outdir` contains the original GW runs: but these gave very bad KN runs due to the larger distances involved. This is because these GW runs are using a cut on spins and masses, therefore, the events are more spread out towards larger dL values. The corresponding data is in `failed_downsampled_data` 

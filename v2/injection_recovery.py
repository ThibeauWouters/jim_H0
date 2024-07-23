"""
Idea: try different learning rate schemes to try and fix the injections
"""
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
import numpy as np
# Regular imports 
import argparse
import copy
import numpy as np
from astropy.time import Time
import time
import shutil
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.prior import Uniform, Composite, PowerLaw, Normal
from jimgw.transforms import Transform, default_functions
import utils # our plotting and postprocessing utilities script

from ripple import Mc_eta_to_ms

import optax

# Names of the parameters and their ranges for sampling parameters for the injection
NAMING = ['M_c', 'q', 's1_z', 's2_z', 'lambda_1', 'lambda_2', 'H_0', 'z', 't_c', 'phase_c', 'cos_iota', 'psi', 'ra', 'sin_dec']

# TODO: change the prior for injections
# PRIOR = {
#         "M_c": [1.00662100315094, 2.44429349899292], # but we are going to override this with a more focused prior!
#         "q": [0.125, 1.0], 
#         "s1_z": [-0.05, 0.05], 
#         "s2_z": [-0.05, 0.05], 
#         "lambda_1": [0.0, 5000.0], 
#         "lambda_2": [0.0, 5000.0], 
#         # "d_L": [30.0, 2000.0], 
#         "H_0": [60.0, 80.0], 
#         "z": [0.009783 - 0.000023, 0.009783 + 0.000023], 
#         "t_c": [-0.1, 0.1], 
#         "phase_c": [0.0, 2 * jnp.pi], 
#         "cos_iota": [-1.0, 1.0], 
#         "psi": [0.0, jnp.pi], 
#         "ra": [0.0, 2 * jnp.pi], 
#         "sin_dec": [-1, 1]
# }

PRIOR = {
        "M_c": [1.187, 2.07], # but we are going to override this with a more focused prior!
        "q": [0.125, 1.0], 
        "s1_z": [-0.05, 0.05], 
        "s2_z": [-0.05, 0.05], 
        "lambda_1": [0.0, 5000.0], 
        "lambda_2": [0.0, 5000.0], 
        "H_0": [40.0, 80.0], 
        # "z": [0.009783 - 0.000023, 0.009783 + 0.000023], 
        "z": [0.009783 - 0.0001, 0.009783 + 0.0001], # but will be skipped in favour fo 
        "t_c": [-0.1, 0.1], 
        "phase_c": [0.0, 2 * jnp.pi], 
        "cos_iota": [-1.0, 1.0], 
        "psi": [0.0, jnp.pi], 
        "ra": [0.0, 2 * jnp.pi], 
        "sin_dec": [-1, 1]
}

    
####################
### Script setup ###
####################

def body(args):
    """
    Run an injection and recovery. To get an explanation of the hyperparameters, go to:
        - jim hyperparameters: https://github.com/ThibeauWouters/jim/blob/8cb4ef09fefe9b353bfb89273a4bc0ee52060d72/src/jimgw/jim.py#L26
        - flowMC hyperparameters: https://github.com/ThibeauWouters/flowMC/blob/ad1a32dcb6984b2e178d7204a53d5da54b578073/src/flowMC/sampler/Sampler.py#L40
    """
    
    start_time = time.time()
    # TODO move and get these as arguments
    # Deal with the hyperparameters
    naming = NAMING
    HYPERPARAMETERS = {
    "flowmc": 
        {
            "n_loop_training": 400,
            "n_loop_production": 50,
            "n_local_steps": 5,
            "n_global_steps": 400,
            "n_epochs": 50,
            "n_chains": 1000, 
            "learning_rate": 0.001, # using a scheduler below
            "max_samples": 50000, 
            "momentum": 0.9, 
            "batch_size": 50000, 
            "use_global": True, 
            "logging": True, 
            "keep_quantile": 0.0, 
            "local_autotune": None, 
            "train_thinning": 10, 
            "output_thinning": 30, 
            "n_sample_max": 10000, 
            "precompile": False, 
            "verbose": False, 
            "outdir": args.outdir,
            "stopping_criterion_global_acc": args.stopping_criterion_global_acc,
            "which_local_sampler": "MALA"
        }, 
    "jim": 
        {
            "seed": 0, 
            "n_chains": 1000, 
            "num_layers": 10, 
            "hidden_size": [128, 128], 
            "num_bins": 8, 
        }
    }
    
    flowmc_hyperparameters = HYPERPARAMETERS["flowmc"]
    jim_hyperparameters = HYPERPARAMETERS["jim"]
    hyperparameters = {**flowmc_hyperparameters, **jim_hyperparameters}
    
    # TODO can I just replace this with update dict?
    for key, value in args.__dict__.items():
        if key in hyperparameters:
            hyperparameters[key] = value
            
    ### POLYNOMIAL SCHEDULER
    if args.use_scheduler:
        print("Using polynomial learning rate scheduler")
        total_epochs = hyperparameters["n_epochs"] * hyperparameters["n_loop_training"]
        start = int(total_epochs / 10)
        start_lr = 1e-3
        end_lr = 1e-5
        power = 4.0
        schedule_fn = optax.polynomial_schedule(start_lr, end_lr, power, total_epochs-start, transition_begin=start)
        hyperparameters["learning_rate"] = schedule_fn

    print(f"Saving output to {args.outdir}")
    
    ripple_waveform_fn = RippleIMRPhenomD_NRTidalv2
    reference_waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0, no_taper = True)

    if args.outdir[-1] != "/":
        args.outdir += "/"

    outdir = f"{args.outdir}injection_{args.N}/"
    
    # Get the prior bounds, both as 1D and 2D arrays
    prior_ranges = jnp.array([PRIOR[name] for name in naming])
    prior_low, prior_high = prior_ranges[:, 0], prior_ranges[:, 1]
    bounds = np.array(list(PRIOR.values()))
    bounds = None # TODO: disabled bounds for now, remove if we want to analyze real events!!!!
    
    # Reading parameters
    config_path = f"{outdir}config.json"
    print(f"Loading existing config, path: {config_path}")
    config = json.load(open(config_path))
        
    key = jax.random.PRNGKey(config["seed"])
    
    # TODO: compute the lambdas
    mc = config["M_c"]
    q = config["q"]
    eta = q / (1 + q) ** 2
    
    m1, m2 = Mc_eta_to_ms(jnp.array([mc, eta]))
    
    config["lambda_1"] = utils.get_lambda(m1)
    config["lambda_2"] = utils.get_lambda(m2)
    
    print(f"Lambda 1 is computed to be {config['lambda_1']}")
    print(f"Lambda 2 is computed to be {config['lambda_2']}")
    
    # Get the mass from the lambda
    
    
    # Save the given script hyperparams
    with open(f"{outdir}script_args.json", 'w') as json_file:
        json.dump(args.__dict__, json_file)
    
    # Start injections
    print("Injecting signals . . .")
    waveform = ripple_waveform_fn(f_ref=config["fref"])
    
    # Create frequency grid
    freqs = jnp.arange(
        config["fmin"],
        config["f_sampling"] / 2,  # maximum frequency being halved of sampling frequency
        1. / config["duration"]
        )
    # convert injected mass ratio to eta, and apply arccos and arcsin
    q = config["q"]
    eta = q / (1 + q) ** 2
    iota = float(jnp.arccos(config["cos_iota"]))
    dec = float(jnp.arcsin(config["sin_dec"]))
    # Setup the timing setting for the injection
    epoch = config["duration"] - config["post_trigger_duration"]
    gmst = Time(config["trigger_time"], format='gps').sidereal_time('apparent', 'greenwich').rad
    # Array of injection parameters
    # true_param = {
    #     'M_c':       config["M_c"],       # chirp mass
    #     'eta':       eta,                 # symmetric mass ratio 0 < eta <= 0.25
    #     's1_z':      config["s1_z"],      # aligned spin of priminary component s1_z.
    #     's2_z':      config["s2_z"],      # aligned spin of secondary component s2_z.
    #     'lambda_1':  config["lambda_1"],  # tidal deformability of priminary component lambda_1.
    #     'lambda_2':  config["lambda_2"],  # tidal deformability of secondary component lambda_2.
    #     # 'd_L':       config["d_L"],       # luminosity distance
    #     'H_0':       config["H_0"],       # Hubble constant
    #     'z':         config["z"],         # redshift
    #     't_c':       config["t_c"],       # timeshift w.r.t. trigger time
    #     'phase_c':   config["phase_c"],   # merging phase
    #     'iota':      iota,                # inclination angle
    #     'psi':       config["psi"],       # polarization angle
    #     'ra':        config["ra"],        # right ascension
    #     'dec':       dec                  # declination
    #     }
    
    # TODO: perhaps best to move around?
    # Create the transforms:
    transforms_list = [default_functions["q_to_eta"],
                       default_functions["cos_iota_to_iota"],
                       default_functions["sin_dec_to_dec"],
                       default_functions["H0_z_to_dL"]]
    
    transform = Transform(transforms_list, ["M_c", "s1_z", "s2_z", "lambda_1", "lambda_2", "t_c", "phase_c", "psi", "ra"])
    
    # Get true param and transformed as well.
    true_param_original = {key: config[key] for key in naming}
    true_param = transform.transform(true_param_original)
    
    # Get the true parameter values for the plots
    # truths = copy.deepcopy(true_param_original)
    truths = None # TODO: annoying to get actually...
    
    detector_param = {
        'ra':     config["ra"],
        'dec':    dec,
        'gmst':   gmst,
        'psi':    config["psi"],
        'epoch':  epoch,
        't_c':    config["t_c"],
        }
    print(f"The injected parameters are {true_param}")
    
    # Generating the geocenter waveform
    h_sky = waveform(freqs, true_param)
    
    # TODO: add the interferometers again!
    # # Setup interferometers
    # ifos_string_list = config["ifos"] # list of strings taken from config
    # ifos = [] # list of detector objects
    # for ifo_string in ifos_string_list:
    #     print("Adding interferometer ", ifo_string)
    #     eval(f'ifos.append({ifo_string})')
        
    psd_filename_list = ["./psds/AplusDesign_psd.txt", 
                         "./psds/AplusDesign_psd.txt",
                         "./psds/psd_virgo.txt"]
    
    ifos = [H1, L1, V1]
    
    network_snr = 0
    for ifo, psd_filename in zip(ifos, psd_filename_list):
        key, subkey = jax.random.split(key)
        ifo.inject_signal(
            subkey,
            freqs,
            h_sky,
            detector_param,
            psd_file=psd_filename,
        )
        network_snr += utils.compute_snr(ifo, h_sky, detector_param) ** 2
    print("Signal injected")
    
    # Show network SNR
    network_snr = np.sqrt(network_snr)
    print("Network SNR:", network_snr)
    
    print(f"Saving network SNR")
    with open(outdir + 'network_snr.txt', 'w') as file:
        file.write(str(network_snr))

    print("Start prior setup")
    
    # Priors without transformation 
    if args.chirp_mass_prior == "tight":
        print("INFO: Using a tight chirp mass prior")
        true_mc = true_param["M_c"]
        Mc_prior = Uniform(true_mc - 0.1, true_mc + 0.1, naming=['M_c'])
    else:
        print("INFO: Using regular (broad) chirp mass prior")
        Mc_prior       = Uniform(prior_low[0], prior_high[0], naming=['M_c'])
    q_prior        = Uniform(prior_low[1], prior_high[1], naming=['q'])
    s1z_prior      = Uniform(prior_low[2], prior_high[2], naming=['s1_z'])
    s2z_prior      = Uniform(prior_low[3], prior_high[3], naming=['s2_z'])
    lambda_1_prior = Uniform(prior_low[4], prior_high[4], naming=['lambda_1'])
    lambda_2_prior = Uniform(prior_low[5], prior_high[5], naming=['lambda_2'])
    H0_prior       = Uniform(prior_low[6], prior_high[6], naming=['H_0'])
    z_prior        = Normal(0.009783, 0.000023, naming=['z'])
    tc_prior       = Uniform(prior_low[8], prior_high[8], naming=['t_c'])
    phic_prior     = Uniform(prior_low[9], prior_high[9], naming=['phase_c'])
    cos_iota_prior = Uniform(prior_low[10], prior_high[10], naming=["cos_iota"])
    psi_prior      = Uniform(prior_low[11], prior_high[11], naming=["psi"])
    ra_prior       = Uniform(prior_low[12], prior_high[12], naming=["ra"])
    sin_dec_prior  = Uniform(prior_low[13], prior_high[13], naming=["sin_dec"])
    
    # Compose the prior
    prior_list = [
            Mc_prior,
            q_prior,
            s1z_prior,
            s2z_prior,
            lambda_1_prior,
            lambda_2_prior,
            H0_prior,
            z_prior,
            tc_prior,
            phic_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
    ]
    
    complete_prior = Composite(prior_list)
    bounds = None # TODO: disabled bounds for now, remove if we want to analyze real events!!!!
    # bounds = jnp.array([[p.xmin, p.xmax] for p in complete_prior.priors])
    print("Finished prior setup")
    
    print("Initializing likelihood")
    if args.relative_binning_ref_params_equal_true_params:
        ref_params = true_param
        print("Using the true parameters as reference parameters for the relative binning")
    else:
        ref_params = None
        print("Will search for reference waveform for relative binning")
    
    likelihood = HeterodynedTransientLikelihoodFD(
        ifos,
        prior=complete_prior,
        bounds=bounds,
        n_bins = args.relative_binning_binsize,
        waveform=waveform,
        trigger_time=config["trigger_time"],
        duration=config["duration"],
        post_trigger_duration=config["post_trigger_duration"],
        ref_params=ref_params,
        reference_waveform=reference_waveform,
        )
    
    # Save the ref params
    utils.save_relative_binning_ref_params(likelihood, outdir)

    # Generate arguments for the local samplercd
    mass_matrix = jnp.eye(len(prior_list))
    for idx, prior in enumerate(prior_list):
        try:
            mass_matrix = mass_matrix.at[idx, idx].set(prior.xmax - prior.xmin) # fetch the prior range
        except Exception as e:
            print("There was an exception:", e)
            mass_matrix = mass_matrix.at[idx, idx].set(0.001) # fetch the prior range
    local_sampler_arg = {'step_size': mass_matrix * args.eps_mass_matrix} # set the overall step size
    hyperparameters["local_sampler_arg"] = local_sampler_arg
    
    # Create jim object
    jim = Jim(
        likelihood,
        complete_prior,
        transform,
        **hyperparameters
    )
    
    if args.smart_initial_guess:
        n_chains = hyperparameters["n_chains"]
        n_dim = len(prior_list)
        initial_guess = utils.generate_smart_initial_guess(gmst, ifos, true_param, n_chains, n_dim, prior_low, prior_high)
        # Plot it
        utils.plot_chains(initial_guess, "initial_guess", outdir, truths = truths)
    else:
        initial_guess = jnp.array([])
    
    ### Finally, do the sampling
    jim.sample(jax.random.PRNGKey(24), initial_guess = initial_guess)
        
    # === Show results, save output ===

    # Print a summary to screen:
    jim.print_summary()

    # Save and plot the results of the run
    #  - training phase
    
    name = outdir + f'results_training.npz'
    print(f"Saving samples to {name}")
    state = jim.Sampler.get_sampler_state(training = True)
    chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
    local_accs = jnp.mean(local_accs, axis=0)
    global_accs = jnp.mean(global_accs, axis=0)
    if args.save_training_chains:
        np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals, chains=chains)
    else:
        np.savez(name, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs, loss_vals=loss_vals)
    
    utils.plot_accs(local_accs, "Local accs (training)", "local_accs_training", outdir)
    utils.plot_accs(global_accs, "Global accs (training)", "global_accs_training", outdir)
    utils.plot_loss_vals(loss_vals, "Loss", "loss_vals", outdir)
    utils.plot_log_prob(log_prob, "Log probability (training)", "log_prob_training", outdir)
    
    #  - production phase
    name = outdir + f'results_production.npz'
    state = jim.Sampler.get_sampler_state(training = False)
    chains, log_prob, local_accs, global_accs = state["chains"], state["log_prob"], state["local_accs"], state["global_accs"]
    local_accs = jnp.mean(local_accs, axis=0)
    global_accs = jnp.mean(global_accs, axis=0)
    np.savez(name, chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

    utils.plot_accs(local_accs, "Local accs (production)", "local_accs_production", outdir)
    utils.plot_accs(global_accs, "Global accs (production)", "global_accs_production", outdir)
    utils.plot_log_prob(log_prob, "Log probability (production)", "log_prob_production", outdir)

    # Plot the chains as corner plots
    utils.plot_chains(chains, "chains_production", outdir, truths = truths)
    
    # Save the NF and show a plot of samples from the flow
    print("Saving the NF")
    jim.Sampler.save_flow(outdir + "nf_model")
    name = outdir + 'results_NF.npz'
    chains = jim.Sampler.sample_flow(10_000)
    np.savez(name, chains = chains)
    
    # Finally, copy over this script to the outdir for reproducibility
    shutil.copy2(__file__, outdir + "copy_injection_recovery.py")
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken: {runtime} seconds ({(runtime)/60} minutes)")
    
    print(f"Saving runtime")
    with open(outdir + 'runtime.txt', 'w') as file:
        file.write(str(runtime))
    
    print("Finished injection recovery successfully!")

############
### MAIN ###
############

def main(given_args = None):
    
    parser = utils.get_parser()
    args = parser.parse_args()
    
    print(given_args)
    
    # Update with given args
    if given_args is not None:
        args.__dict__.update(given_args)
        
    print("------------------------------------")
    print("Arguments script:")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("------------------------------------")
        
    print("Starting main code")
    
    # Execute the script
    body(args)
    
if __name__ == "__main__":
    main()
"""
Small utility file to handle loading and processing of data. This file is checked in the main notebook. 
"""

import os
import json
import numpy as np
np.random.seed(0)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gwpy.table import EventTable
from astropy.table import Table, join
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value

from ripple import ms_to_Mc_eta

################
### PREAMBLE ###
################

# These are the column names of the subpopulations .dat files
COLUMN_NAMES = ["simulation_id", 
                "longitude", 
                "latitude", 
                "inclination", 
                "distance", 
                "mass1", 
                "mass2", 
                "spin1z", 
                "spin2z", 
                "polarization", 
                "coa_phase", 
                "geocent_end_time", 
                "geocent_end_time_ns"]

INJECTION_OUTDIR = "/home/thibeau.wouters/projects/jim_H0/injections/"

def filter_dict_by_indices(original_dict: dict, indices: list[int]) -> dict:
    """
    Filter the values of a dictionary by the provided indices. The keys remain the same. If a single index is passed, the values will be floats, whereas if multiple indices are passed, the values will be Numpy arrays.

    Args:
        original_dict (dict): Original dictionary
        indices (list[int]): List of indices to filter by

    Returns:
        dict: Filtered dictionary
    """
    filtered_dict = {}
    
    # Iterate through each key-value pair in the original dictionary
    for key, values in original_dict.items():
        # Filter the values by the provided indices, either as float or as array
        if len(indices) == 1:
            filtered_values = values[indices[0]]
        else:
            filtered_values = np.array([values[i] for i in indices])
        
        # Save them
        filtered_dict[key] = filtered_values
    
    return filtered_dict

####################
### READING DATA ###
####################

def get_events_info(table: EventTable, simulation_ids: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Get the SNR and ifos of the given simulation IDs from the given table. Table has to be the events.xml file.

    Args:
        table (EventTable): Table with the SNR values
        simulation_ids (np.ndarray): Array with the simulation IDs

    Returns:
        tuple[np.ndarray, list]: Numpy array with the SNR values and list with strings for ifos in the event.
    """
    
    # Result will be an array of the SNRs
    snr_array = np.zeros(len(simulation_ids))
    ifos_array = []
    
    # Get the values that we wish to fetch
    snr_xml = np.array(table["snr"])
    ifos_xml = np.array(table["ifos"])
    
    # Get the simulation IDs that are in this table
    simulation_id_xml = np.array(table["coinc_event_id"])
    
    # Iterate over the given simulation ids and fetch their SNR
    for i, sim_id in enumerate(simulation_ids):
        idx = np.where(simulation_id_xml == int(sim_id))[0][0]
        
        snr_array[i] = snr_xml[idx]
        ifos_array.append(ifos_xml[idx])
    
    return snr_array, ifos_array

def read_injections_file(filename: str) -> dict:
    """
    Read in the injections file and return a dictionary with the data. We read in the original columns and their values, but also add the redshift and source masses. Note: this format only works for the splitted version per subpopulation that Weizmann provided.

    Args:
        filename (str): Name of the injections.xml file

    Returns:
        dict: Dictionary with the data
    """
    
    # Read it, and convert to numpy array
    data = np.genfromtxt(filename, names = True)
    data = np.array([list(row) for row in data])
    data = data.T # shape is: (n_dim, n_events)
    
    # Put it in a dictionary
    data_dict = {name: data[i] for i, name in enumerate(COLUMN_NAMES)}
    data_dict["simulation_id"] = data_dict["simulation_id"].astype(int)
    
    # Also add the redshift
    z = z_at_value(cosmo.luminosity_distance, data_dict["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    data_dict["redshift"] = z
    
    # Also add source masses for convenience
    zp1 = z + 1

    source_mass1 = data_dict["mass1"] / zp1
    source_mass2 = data_dict["mass2"] / zp1
    
    data_dict["source_mass1"] = source_mass1
    data_dict["source_mass2"] = source_mass2
    
    return data_dict

#################
### INJECTION ###
#################

def generate_config(params_dict: dict, 
                    outdir: str = "./outdir/",
                    ) -> dict:
    """
    Generate a configuration file for the given parameters. This file can be used to inject a signal into the data.

    Args:
        params_dict (dict): Dictionary with the parameters of the injection
        outdir (str, optional): Where to save the results. Defaults to "./outdir/".

    Returns:
        dict: The generated configuration file as a dictionary
    """
    
    # Get the simulation id and use it to get the full directory path
    N = params_dict["simulation_id"]
    output_path = f"{outdir}injection_{str(N)}/"
    
    # Get the parameters of this injection
    m1, m2 = params_dict["source_mass1"], params_dict["source_mass2"]
    assert m1 > m2, "Mass 1 should be larger than mass 2!"
    
    mc, _ = ms_to_Mc_eta(jnp.array([m1, m2]))
    q = m2 / m1 # smaller than 1
    
    s1_z = params_dict["spin1z"]
    s2_z = params_dict["spin2z"]
    d_L = params_dict["distance"]
    t_c = 0.0 # TODO: make sure this is handled properly?
    trigger_time = params_dict["geocent_end_time"]
    phase_c = params_dict["coa_phase"]
    cos_iota = np.cos(params_dict["inclination"])
    psi = params_dict["polarization"]
    ra = params_dict["longitude"] # TODO: check if this is OK?
    sin_dec = np.sin(params_dict["latitude"]) # TODO: check if this is OK?
        
    # Create new injection file
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Made injection directory: ", output_path)
    filename = output_path + f"config.json"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Made injection directory: ", output_path)
    else:
        print("Injection directory exists: ", output_path)

    # Note: ifos might be a string, can easily get it as a list instead
    ifos = params_dict["ifos"]
    if isinstance(params_dict["ifos"], str):
        ifos = params_dict["ifos"].split(",")

    # Create the injection dictionary
    seed = np.random.randint(low=0, high=10000)
    injection_dict = {
        # Random stuff
        'seed': seed,
        'ifos': ifos,
        'outdir' : output_path,
        'snr' : params_dict["snr"], # note: as computed by Weizmann, can be checked with my own code for sanity checking
        
        # Frequency and duration
        'f_sampling': 2 * 2048,
        'fmin': 20,
        'fref': 20,
        'trigger_time': trigger_time,
        'duration': 128, # TODO: how to get the right duration
        'post_trigger_duration': 2,
        
        # Parameters of the injection
        'M_c': float(mc),
        'q': q,
        's1_z': s1_z,
        's2_z': s2_z,
        'd_L': d_L,
        't_c': t_c,
        'phase_c': phase_c,
        'cos_iota': cos_iota,
        'psi': psi,
        'ra': ra,
        'sin_dec': sin_dec
    }
    
    # Save the injection file to the output directory as JSON
    with open(filename, 'w') as f:
        json.dump(injection_dict, f)
    
    return injection_dict
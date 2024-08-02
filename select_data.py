"""Script to load the data and select the events out of all the data that we wish to analyze"""

# mandatory stuff to get things working properly
import psutil
p = psutil.Process()
p.cpu_affinity([0])
import os
import json
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import utils
from gwpy.table import EventTable

# import jax.numpy as jnp
# from ripple import ms_to_Mc_eta, Mc_eta_to_ms

from bilby.gw.conversion import component_masses_to_chirp_mass, chirp_mass_and_mass_ratio_to_component_masses

rcparams = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(rcparams)


############
### DATA ###
############

# Define the locations etc here:
my_outdir = "./injections_tidal/outdir/" # where the rundirs will come
data_dir = "./data_split/"
bns_o5 = data_dir + "bns_O5HL_injections.dat"
bns_o4 = data_dir + "bns_O4HL_injections.dat"

xml_filename_o5 = "/home/thibeau.wouters/gw-datasets/H0_inference_O5/events_O5.xml" # on CIT!
xml_filename_o4 = "/home/thibeau.wouters/gw-datasets/H0_inference_O4/events_O4.xml" # on CIT!

# Choose which to load:
which_run = "O5"

print(f"Analyzing data from {which_run}")

if which_run == "O5":
    xml_filename = xml_filename_o5
    bns = bns_o5
else:
    xml_filename = xml_filename_o4
    bns = bns_o4

# Load in the injections file first
data_dict = utils.read_injections_file(bns)
table = EventTable.read(xml_filename, tablename = "coinc_inspiral")

# # Get the keys of this table
# keys = table.keys()
# print("keys")
# print(keys)

# # Get some specific variables that I want to use
# simulation_id_xml = np.array(table["coinc_event_id"])
# snr_xml = np.array(table["snr"])
# end_time = np.array(table["end_time"])

snr_array, ifos_array = utils.get_events_info(table, data_dict["simulation_id"])

# print("Examples of SNR and ifos:")
# print(snr_array[:5])
# print(ifos_array[:5])

data_dict["snr"] = snr_array
data_dict["ifos"] = ifos_array
assert len(data_dict["ifos"]) == len(data_dict["simulation_id"]), "ifos and simulation_id have different lengths"

#################
### SELECTION ###
#################

# Here, we select which events we want to analyze for the H0 inference

max_nb = 30

sort_idx = np.argsort(data_dict["snr"])[::-1]
sampled_indices = sort_idx[:max_nb]

keys = ["snr", "mass1", "mass2", "spin1z", "spin2z", "distance"]

for key in keys:
    print(f"{key} min and max")
    values = data_dict[key][sampled_indices]
    print(np.min(values), np.max(values))
    
filtered_dict = utils.filter_dict_by_indices(data_dict, sampled_indices)

# In separate_dicts we have ALL the injections but as a list of individual dicts
separate_dicts = utils.split_dict(data_dict)

# In selected_dicts we have ONLY the chosen/selected injections as a list of individual dicts
selected_dicts = utils.split_dict(filtered_dict)

for config_dict in selected_dicts:
    _ = utils.generate_config(config_dict, "./injections_tidal/outdir/")
    

################
### CHECKING ###
################

# Let's check whether the injected data looks good

# MTOV stuff
MTOV_middle = 2.26
MTOV_min = MTOV_middle - 0.22
MTOV_max = MTOV_middle + 0.45

dirs = os.listdir("./injections_tidal/outdir/")

mass1 = []
mass2 = []
spin1z = []
spin2z = []
dL = []
snr = []

# Check what is in the config
for d in dirs:
    path = os.path.join(os.path.join("./injections_tidal/outdir/", d), "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    
    # Masses
    mc, q = config["M_c"], config["q"]
    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc, q)
    mass1.append(m1)
    mass2.append(m2)
    
    # Spins
    spin1z.append(config["s1_z"])
    spin2z.append(config["s2_z"])
    
    # Distance and SNR
    dL.append(config["d_L"])
    snr.append(config["snr"])
    
    for values, name in zip([mass1, mass2, spin1z, spin2z, dL, snr], ["m1", "m2", "spin1z", "spin2z", "dL", "snr"]):
        plt.hist(values, bins = 20, histtype = "step", density=True, lw = 2, color = "blue")
        plt.xlabel(name)
        plt.ylabel("Density")
        if name in ["m1", "m2"]:
            ymin = 0
            ymax = 5
            plt.fill_between([MTOV_min, MTOV_max], ymin, ymax, color = "gray", alpha = 0.25, label = r"$M_{\rm{TOV}}$ bound")
            plt.axvline(MTOV_middle, ymin=ymin, ymax=ymax, color = "gray", linestyle = "--")
        plt.savefig("./figures/selected_events_" + name + ".png")
        plt.close()
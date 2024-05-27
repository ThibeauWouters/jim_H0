# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : December 2023
@description    : Tools for Reading and Parsing Observing Scenarios Simulation Data
                (Farah/GWTC-3 distribution).
---------------------------------------------------------------------------------------------------
"""

import os
from tqdm.auto import tqdm
from pathlib import Path

from astropy.table import Table, join
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value

from ligo.skymap.io import read_sky_map
from gwpy.table import Table as gwpy_Table

# -------------------------------------------------------------------------------
# Set 'nmma' to False if these parameters are not intended for 'nmma' inferences.
# For 'nmma' inferences, redshift (z) should be less than 2, constraining
# the Compact Binary Coalescences (CBCs) within a radius of 15740 Mpc.
# -------------------------------------------------------------------------------
nmma = True

# ------------------------------------------------------------------------------
# Set 'GPS_TIME' to True to include GPS time in parameters.
# Note: GPS times are already present in the '.fits' skymap events,
#       adding them might prolong reading times. The formula for gps_time is:
#       gps_time ≈ geocent_end_time + (geocent_end_time_ns)^(10^(-9)),
#       where geocent_end_time is in seconds (s) and geocent_end_time_ns
#       is in nanoseconds, easily obtainable from the xml files.
# ------------------------------------------------------------------------------
GPS_TIME = False


def populations(table, pop, ns_max_mass=3.0):

    """Splits Compact Binary Coalescence (CBC) events based on source frame mass.

    :param table: Table containing the CBC data
    :type table: astropy.table.Table

    :param ns_max_mass: Maximum neutron star mass threshold
    :type ns_max_mass: float

    :param pop: Type of population (BNS, NSBH, and BBH.)
    :type pop: str

    :return: Subset of data based on population criteria
    :rtype: astropy.table.Table
    """
    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1

    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1

    if pop == "BNS":
        data = table[(source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)]
    elif pop == "NSBH":
        data = table[(source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)]
    else:
        data = table[(source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)]

    return data


def extract_GPSTime(data, xml_data, GPS_TIME=False, path=None):

    """Extracts GPS time, polarization and coa_phase informations provided by ligo XML data.

    :param data: Subset of data based on population criteria get in populations().
    :type data: astropy.table.Table

    :param xml_data: Events from LIGO-LW XML data.
    :type xml_data:  astropy.table.table.Table

    :param GPS_TIME: Flag indicating whether GPS time from skymap ".fits" should be extracted.
    :type GPS_TIME: bool, optional

    :param path: Path to the  run name (eg: "./data/runs/O4/farah) data  location.
                 But don't need if "GPS_TIME=False".
    :type path: str

    :return: Tables containing,  geocent_end_time, polarization and related details.
    :rtype: astropy.table.Table
    """
    gps_time = []
    geocent_end_time = []
    geocent_end_time_ns = []
    polarization = []
    coa_phase = []

    simulation_ID = data["simulation_id"]
    for ID in simulation_ID:
        geocent_end_time.append(xml_data[ID]["geocent_end_time"])
        geocent_end_time_ns.append(xml_data[ID]["geocent_end_time_ns"])
        polarization.append(xml_data[ID]["polarization"])
        coa_phase.append(xml_data[ID]["coa_phase"])

        if GPS_TIME:
            gps_time.append(read_sky_map(f"{path}/allsky/{ID}.fits")[1]["gps_time"])

    if GPS_TIME:
        time_dict = {
            "simulation_id": simulation_ID,
            "polarization": polarization,
            "coa_phase": coa_phase,
            "geocent_end_time": geocent_end_time,
            "geocent_end_time_ns": geocent_end_time_ns,
            "gps_time": gps_time,
        }
    else:
        time_dict = {
            "simulation_id": simulation_ID,
            "polarization": polarization,
            "coa_phase": coa_phase,
            "geocent_end_time": geocent_end_time,
            "geocent_end_time_ns": geocent_end_time_ns,
        }

    tables = Table(time_dict)

    return tables


# Directory containing simulation data runs
datapath = "runs"

# Create output directories to save split BNS, NSBH, and BBH results.
outdir = "./output/nmma_GWparams"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3.0

pops = ["BNS", "NSBH", "BBH"]
run_names = run_dirs = ["O4HL", "O5HL"]

with tqdm(total=len(run_names) * len(pops)) as progress:

    for run_name, run_dir in zip(run_names, run_dirs):
        path = Path(f"{datapath}/{run_dir}/farah")

        # Read injections.dat files
        injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")

        print("\n===============================================================\n")
        for pop in pops:
            if pop in ["BNS", "NSBH"]:

                Data = populations(table=injections, pop=pop, ns_max_mass=ns_max_mass)

                # Select BNS and NSBH data with z <= 1.98 for nmma inference
                if nmma:
                    print(
                        f"The number of subpopulation in {run_name} in within the radius of 15740 Mpc : "
                    )
                    Data = Data[Data["distance"] <= 15740]

                # Read LIGO-LW XML events to extract extension parameters, such as polarization and coa_phase,
                # for the namma process
                xmlData = gwpy_Table.read(
                    str(path / "events.xml.gz"),
                    format="ligolw",
                    tablename="sim_inspiral",
                )

                gpsTables = extract_GPSTime(
                    data=Data, xml_data=xmlData, GPS_TIME=False, path=None
                )

                ### combine gpsTable  to Data table
                tables = join(Data, gpsTables)

                # Save the subpopulations
                tables.write(
                    Path(f"{outdir}/{pop.lower()}_{run_name}_injections.dat"),
                    format="ascii.tab",
                    overwrite=True,
                )

                print(f"{pop} {len(tables)} ; ")
                print(
                    "\n==============================================================="
                )

                progress.update()

    del (
        injections,
        Data,
        xmlData,
        gpsTables,
        tables,
        path,
    )

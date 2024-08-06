"""Custom corner plot script since NMMA throws an error when trying to plot the corner plot."""

import numpy as np
import json
import matplotlib.pyplot as plt
import corner

KEYS = ['luminosity_distance', 'log10_mej_dyn', 'vej_dyn', 'Yedyn', 'log10_mej_wind', 'vej_wind']

def load_samples(id: int,
                 which_run: str = "Rubin"):
    """
    Load the samples from the NMMA JSON result

    Args:
        id (int): Identifier of the run
        which_run (str): Which run to load the samples from, either "Rubin" or "ZTF". Default is "Rubin".
    """
    
    supported_runs = ["Rubin", "ZTF"]
    if which_run not in supported_runs:
        raise ValueError(f"which_run argument must be one of {supported_runs}")
    
    # Load the JSON
    with open(f"outdir_{which_run}/{id}/injection_Bu2022Ye_{id}_result.json") as f:
        data = json.load(f)
        
    # Fetch posterior samples
    posterior = data["posterior"]['content']
    
    posterior_samples = np.array([posterior[key] for key in KEYS]).T
    
    return posterior_samples
    
def make_corner_plot(samples: np.array,
                     id: int,
                     which_run: str = "Rubin",
                     labels: list[str] = KEYS):
    
    corner.corner(samples, labels=labels)
    plt.savefig(f"outdir_{which_run}/{id}/cornerplot.png")

def main():
    # Load samples
    N = 20
    for id in range(0, N+1):
        print(f"=== Making cornerplot for id = {id}")
        success_counter = 0

        try: 
            samples = load_samples(id, which_run="Rubin")
            make_corner_plot(samples, id)
            success_counter += 1
        except Exception as e:
            print(f"Issue: {e} -- failed run?")
    
    print(f"Percentage of successful runs: {success_counter/(N+1)*100:.2f}%")
    
    return

if __name__ == '__main__':
    main()
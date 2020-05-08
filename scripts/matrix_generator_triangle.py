import os
import graco
import numpy as np
import pandas as pd
import networkx as nx

"""
Takes network, feature and metric as input and calculates distance matrix.
"""

# =============================================================================
#  ---------------------------------- MAIN -----------------------------------
# =============================================================================

def main(network, feature, metric):
 # Start of computations
    G_nx = nx.read_edgelist(f"{NETWORK_DIRECTORY}/{network}.txt")
    D_arr = graco.triangle_distance(G_nx)

    np.savetxt(f"{MATRIX_DIRECTORY}/{network}/{feature}/{metric}.txt", D_arr,
           fmt='%.7f', header=' '.join(G_nx), comments='')

if __name__ == '__main__':
    from itertools import product
    from multiprocessing import Pool

    print('Number of available cores:', os.cpu_count())

 # Global constants
    DATA_DIRECTORY = "/Users/markusyoussef/Desktop/git/supplements/data"
    YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed_data/yeast"
    NETWORK_DIRECTORY = f"{YEAST_DIRECTORY}/networks"
    MATRIX_DIRECTORY  = f"{YEAST_DIRECTORY}/distance_matrices"

 # Input parameters
    with open("input_parameters.py") as f:
        for line in f:
            exec(line.strip())

 # Define necessary directories
    for network, feature in product(networks, features):
        if not os.path.exists(f"{MATRIX_DIRECTORY}/{network}/{feature}"):
            os.makedirs(f"{MATRIX_DIRECTORY}/{network}/{feature}/")

    with Pool() as p:
        p.starmap(main,product(networks, features, metrics))

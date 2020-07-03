import os
import graco
import numpy as np
import pandas as pd
import networkx as nx

"""
Takes network, feature and metric as input and calculates distance matrix.
"""

# =============================================================================
#  -------------------------------- FUNCTIONS --------------------------------
# =============================================================================

def get_feature_matrix(feature, G_nx):
    if feature == 'GDV':
        GDV = graco.orbits(G_nx)
        return GDV
    else:
        GCV = graco.coefficients(G_nx)

 # Single equation sets
    if   feature == 'GCV-D':
        feature_matrix = GCV['D']
    elif feature == 'GCV-A':
        feature_matrix = GCV['A']
    elif feature == 'GCV-G':
        feature_matrix = GCV['G']
    elif feature == 'GCV-G-sym':
        feature_matrix = GCV['G'][['0-0','1-1','3-3']]
    elif feature == 'GCV-O':
        feature_matrix = GCV['O']
    elif feature == 'GCV-3':
        feature_matrix = GCV[['A','D']].xs('0', axis=1, level='Equation')

 # Combined equation sets
    elif feature == 'GCV-DA':
        feature_matrix = GCV[['D','A']]
    elif feature == 'GCV-DG':
        feature_matrix = GCV[['D','G']]
    elif feature == 'GCV-DO':
        feature_matrix = GCV[['D','O']]
    elif feature == 'GCV-all':
        feature_matrix = GCV.drop(('G','0-0'), axis=1)
    elif feature == 'GCV-DG-2':
        feature_matrix = GCV[['D','G']].drop(('G','2-1'), axis=1)
    elif feature == 'GCV-DG-3':
        feature_matrix = GCV[['D','G']].drop(('G','3-3'), axis=1)
    elif feature == 'GCV-DG-sym':
        GCV_G_sym = GCV['G'][['0-0','1-1','3-3']]
        feature_matrix = pd.concat([GCV['D'],GCV_G_sym], axis=1)
    elif feature == 'GCV-DAG':
        feature_matrix = GCV[['D','A','G']].drop(('G','0-0'), axis=1)
    elif feature == 'GCV-DAG-reduced':
        G_red = [('G','0-0'), ('G','1-2'), ('G','2-1'), ('G', '3-3')]
        feature_matrix = GCV[['D','A','G']].drop(G_red, axis=1)
    elif feature == 'GCV-O+':
        GCV_3 = GCV[['A','D']].xs('0', axis=1, level='Equation')
        feature_matrix = pd.concat([GCV['O'],GCV_3], axis=1)

    return feature_matrix


# =============================================================================
#  ---------------------------------- MAIN -----------------------------------
# =============================================================================

def main(network, feature, metric):
 # Start of computations
    G_nx = nx.read_edgelist(f"{NETWORK_DIRECTORY}/{network}.txt")
    feature_matrix = get_feature_matrix(feature, G_nx)
    D_arr = graco.GCV_distance_matrix(feature_matrix, metric)

    np.savetxt(f"{MATRIX_DIRECTORY}/{network}/{feature}/{metric}.txt", D_arr,
           fmt='%.7f', header=' '.join(G_nx), comments='')

if __name__ == '__main__':
    from itertools import product
    from multiprocessing import Pool

    print('Number of available cores:', os.cpu_count())

 # Global constants
    DATA_DIRECTORY = "/media/clusterduck123/joe/data"
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

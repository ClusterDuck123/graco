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
    elif feature == 'GCV-DG-sym':
        GCV_G_sym = GCV['G'][['0-0','1-1','3-3']]
        feature_matrix = pd.concat([GCV['D'],GCV_G_sym], axis=1)
    elif feature == 'GCV-DAG':
        feature_matrix = GCV[['D','A','G']].drop(('G','0-0'), axis=1)
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
    D_arr = graco.distance_matrix(feature_matrix, metric)

    np.savetxt(f"{MATRIX_DIRECTORY}/{network}/{feature}/{metric}.txt", D_arr,
           fmt='%.7f', header=' '.join(G_nx), comments='')

if __name__ == '__main__':
    from itertools import product
    from multiprocessing import Pool

    print(os.cpu_count())

 # Global constants
    DATA_DIRECTORY = "/media/clusterduck123/joe/data"
    YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed_data/yeast"
    NETWORK_DIRECTORY = f"{YEAST_DIRECTORY}/networks"
    MATRIX_DIRECTORY  = f"{YEAST_DIRECTORY}/distance_matrices"

 # Input parameters
    networks = {'systematic_PPI_BioGRID', 'systematic_CoEx_COEXPRESdb'}
    features = {'GCV-O+', 'GDV'}
    metrics  = {'canberra', 'cityblock'}

 # Define necessary directories
    for network, feature in product(networks, features):
        if not os.path.exists(f"{MATRIX_DIRECTORY}/{network}/{feature}"):
            os.makedirs(f"{MATRIX_DIRECTORY}/{network}/{feature}/")

    with Pool() as p:
        p.starmap(main,product(networks, features, metrics))

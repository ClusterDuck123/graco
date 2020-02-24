from pyclustering.cluster.kmedoids import kmedoids

import os
import random
import numpy as np
import pandas as pd

"""
Takes network, feature and metric as input and calculates distance matrix.
"""

# =============================================================================
#  ---------------------------- GLOBAL PARAMETERS ----------------------------
# =============================================================================

RUN = 3
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

# =============================================================================
#  -------------------------------- FUNCTIONS --------------------------------
# =============================================================================




# =============================================================================
#  ---------------------------------- MAIN -----------------------------------
# =============================================================================

def main(network, feature, metric, method):
    CLUSTER_DIRECTORY = f"{YEAST_DIRECTORY}/clusterings/{network}/{feature}/{metric}/{method}"

    D_df  = pd.read_csv(f"{MATRIX_DIRECTORY}/{network}/{feature}/{metric}.txt", delimiter=' ')
    D_arr = D_df.values.astype(float)

    for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS+1):
        initial_medoids   = random.sample(range(len(D_arr)), n_clusters)
        kmedoids_instance = kmedoids(D_arr, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()

        with open(f"{CLUSTER_DIRECTORY}/{RUN}_{n_clusters}.txt", 'w') as f:
            for cluster in kmedoids_instance.get_clusters():
                f.write(' '.join(D_df.columns[cluster]) + '\n')


if __name__ == '__main__':
    from itertools import product
    from multiprocessing import Pool

    print('Number of available cores:', os.cpu_count())

 # Global constants

    DATA_DIRECTORY = "/media/clusterduck123/joe/data"
    YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed_data/yeast"
    MATRIX_DIRECTORY  = f"{YEAST_DIRECTORY}/distance_matrices"

 # Input parameters
    networks = {'systematic_PPI_BioGRID', 'systematic_CoEx_COEXPRESdb'}
    features = {'GCV-O+', 'GDV'}
    metrics  = {'canberra'}
    methods   = {'kmedoid'}

 # Define necessary directories
    for network, feature, metric, method in product(networks, features, metrics, methods):
        CLUSTER_DIRECTORY = f"{YEAST_DIRECTORY}/clusterings/{network}/{feature}/{metric}/{method}"
        if not os.path.exists(CLUSTER_DIRECTORY):
            os.makedirs(CLUSTER_DIRECTORY)

    with Pool() as p:
        p.starmap(main,product(networks, features, metrics, methods))

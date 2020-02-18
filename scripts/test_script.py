from pyclustering.cluster.kmedoids import kmedoids
from functools import partial
from random import sample

import os
import time
import sys
import numpy as np
import pandas as pd
import networkx as nx


# =============================================================================
# =============================== ABSOLUTE PATHS ==============================
# =============================================================================

DATA_DIRECTORY = "/media/clusterduck123/joe/data"
HUMAN_DIRECTORY = f"{DATA_DIRECTORY}/processed-data/human"
NETWORK_DIRECTORY = f"{HUMAN_DIRECTORY}/networks"
MATRIX_DIRECTORY  = f"{HUMAN_DIRECTORY}/distance-matrices"
ANNOTATION_DIRECTORY = f"{HUMAN_DIRECTORY}/annotations"



# =============================================================================
# ================================= FUNCTIONS =================================
# =============================================================================

def get_number_of_pre_runs(CLUSTER_DIRECTORY, distance, n_clusters = 99):
    splitted_file_names = [name.split('_') for name in os.listdir(CLUSTER_DIRECTORY)]
    pre_runs = [int(run) for run, ncluster, db_txt in splitted_file_names if ncluster == str(n_clusters)]
    if pre_runs:
        return max(pre_runs)
    else:
        return -1



# =============================================================================
# ================================= PARAMETERS ================================
# =============================================================================
MIN_CLUSTERS = 2
MAX_CLUSTERS = 100

feature = sys.argv[1]
method = 'kmedoid'



# =============================================================================
# =================================== MAIN ====================================
# =============================================================================

for run in range(5):
    for distance in {sys.argv[2]}:
        print(distance)

        CLUSTER_DIRECTORY = f"{HUMAN_DIRECTORY}/clusterings/{feature}/{distance}/{method}"
        if not os.path.exists(CLUSTER_DIRECTORY):
            os.makedirs(CLUSTER_DIRECTORY)

        df = pd.read_csv(f"{MATRIX_DIRECTORY}/{feature}/{distance}_BioGRID.txt", delimiter=' ')
        D  = df.values.astype(float)

        t1 = time.time()
        for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS+1):
            initial_medoids = sample(range(len(D)), n_clusters)
            kmedoids_instance = kmedoids(D, initial_medoids, data_type='distance_matrix')
            kmedoids_instance.process()

            nr = get_number_of_pre_runs(CLUSTER_DIRECTORY, distance, MAX_CLUSTERS)

            with open(f"{CLUSTER_DIRECTORY}/{nr+1}_{n_clusters}_BioGRID.txt", 'w') as f:
                for cluster in kmedoids_instance.get_clusters():
                    f.write(' '.join(df.columns[cluster]) + '\n')
            t2 = time.time()
            print(f'{n_clusters}: {t2-t1:.2f}sec', end='\r')
        print()

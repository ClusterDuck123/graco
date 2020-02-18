from scipy.spatial.distance import squareform, cdist, pdist
from itertools import combinations
from functools import partial

import os
import sys
import graco
import numpy as np
import pandas as pd
import networkx as nx

"""
Takes network, feature and metric as input and calculates distance matrix.
"""

# Global constants
DATA_DIRECTORY = "/media/clusterduck123/joe/data"
YEAST_DIRECTORY = f"{DATA_DIRECTORY}/processed-data/yeast"
NETWORK_DIRECTORY = f"{YEAST_DIRECTORY}/networks"
MATRIX_DIRECTORY  = f"{YEAST_DIRECTORY}/distance-matrices"

# Input parameters
network = sys.argv[1]
feature = sys.argv[2]
metric  = sys.argv[3]

# preparation
if not os.path.exists(f"{MATRIX_DIRECTORY}/{network}"):
    os.makedirs(f"{MATRIX_DIRECTORY}/{network}/")

def get_feature_matrix(feature):
    G_nx = nx.read_edgelist(f"{NETWORK_DIRECTORY}/{network}.txt")
    if feature == 'GDV':
        GDV = graco.orbits(G_nx)
        return GDV
    else:
        GCV = graco.coefficients(G_nx)

    if   feature == 'GCV-D':
        feature_matrix = GCV['D']
    elif feature == 'GCV-A':
        feature_matrix = GCV['A']
    elif feature == 'GCV-G':
        feature_matrix = GCV['G']
    elif feature == 'GCV-O':
        feature_matrix = GCV['O']

    elif feature == 'GCV-3':
        feature_matrix = GCV['A']

    return feature_matrix

G_nx = nx.read_edgelist(f"{NETWORK_DIRECTORY}/{network}.txt")
GDV = graco.orbits(G_nx)
GCV = graco.coefficients(GDV)

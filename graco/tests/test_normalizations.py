from scipy.spatial.distance import squareform, pdist
from itertools import combinations, product
from functools import partial

import graco
import unittest
import numpy as np
import pandas as pd
import networkx as nx


distances = ['cityblock', 'euclidean', 'sqeuclidean', 'chebyshev',
             'cosine', 'correlation', 'canberra', 'braycurtis',
             'hellinger', 'seuclidean', 'js_divergence']

# ============================================================================
#                              Distance Functions
# ============================================================================

class TestNormalizations(unittest.TestCase):
    def test_length2_convex_combinations(self):
        gcv1 = pd.DataFrame([[a,1-a] for a in np.linspace(0,1,10)])
        gcv2 = np.random.uniform(size=[10,2])
        gcv2 = pd.DataFrame(gcv2.T/gcv2.sum(axis=1)).T

        GCV = pd.concat([gcv1, gcv2])
        GCV.index = range(len(gcv1)+len(gcv2))

        for distance in distances:
            max_value = np.max(np.max(graco.GCV_distance_matrix(GCV, distance)))
            assert np.isclose(max_value, 1)

    def test_length3_convex_combinations(self):
        gcv1 = pd.DataFrame([[a,b,1-a-b]
                            for (a,b) in product(np.linspace(0,1,10),
                                                 repeat=2)
                                if a+b <= 1])
        gcv2 = np.random.uniform(size=[10,3])
        gcv2 = pd.DataFrame(gcv2.T/gcv2.sum(axis=1)).T

        GCV = pd.concat([gcv1, gcv2])
        GCV.index = range(len(gcv1)+len(gcv2))

        for distance in distances:
            max_value = np.max(np.max(graco.GCV_distance_matrix(GCV, distance)))
            assert np.isclose(max_value, 1)

    def test_length4_convex_combinations(self):
        gcv1 = pd.DataFrame([[a,b,c,1-a-b-c]
                            for (a,b,c) in product(np.linspace(0,1,10),
                                                   repeat=3)
                                if a+b+c <= 1])

        gcv2 = np.random.uniform(size=[10,4])
        gcv2 = pd.DataFrame(gcv2.T/gcv2.sum(axis=1)).T

        GCV = pd.concat([gcv1, gcv2])
        GCV.index = range(len(gcv1)+len(gcv2))

        for distance in distances:
            max_value = np.max(np.max(graco.GCV_distance_matrix(GCV, distance)))
            assert np.isclose(max_value, 1)

    def test_length5_convex_combinations(self):
        gcv1 = pd.DataFrame([[a,b,c,d,1-a-b-c-d]
                            for (a,b,c,d) in product(np.linspace(0,1,10),
                                                     repeat=4)
                                if a+b+c+d <= 1])
        gcv2 = np.random.uniform(size=[10,5])
        gcv2 = pd.DataFrame(gcv2.T/gcv2.sum(axis=1)).T

        GCV = pd.concat([gcv1, gcv2])
        GCV.index = range(len(gcv1)+len(gcv2))

        for distance in distances:
            max_value = np.max(np.max(graco.GCV_distance_matrix(GCV, distance)))
            assert np.isclose(max_value, 1)

if __name__ == '__main__':
    unittest.main()

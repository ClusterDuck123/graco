from scipy.spatial.distance import squareform, pdist
from itertools import combinations
from functools import partial

import graco
import unittest
import numpy as np
import networkx as nx


# ============================================================================
#                              Distance Functions
# ============================================================================

class TestIntCompabilities(unittest.TestCase):
    def setUp(self):
        self.GDV = np.random.randint(2**62, size=[100,15])

    def test_int_GDV_similarity(self):
        D1 = graco.distance_matrices.GDV_similarity(self.GDV)
        D2 = graco.distance_matrix(self.GDV, 'GDV_similarity')
        D3 = squareform([graco.distances.GDV_similarity(u,v)
                        for u,v in combinations(self.GDV,2)])

        np.testing.assert_almost_equal(D1, D2, decimal=4)
        np.testing.assert_almost_equal(D1, D3, decimal=4)
        np.testing.assert_almost_equal(D2, D3, decimal=4)

    def test_int_pdist(self):
        for distance in ['euclidean', 'cityblock', 'sqeuclidean',
                         'cosine', 'correlation', 'chebyshev',
                         'canberra', 'braycurtis']:
            D1 = graco.distance_matrix(self.GDV, distance)
            D2 = squareform(pdist(self.GDV, distance))
            D3 = squareform([graco.distance(u,v, distance)
                            for u,v in combinations(self.GDV,2)])

            np.testing.assert_allclose(D1, D2)
            np.testing.assert_allclose(D1, D3)
            np.testing.assert_allclose(D2, D3)

class TestFloatCompabilities(unittest.TestCase):
    def setUp(self):
        M = np.random.uniform(size=[100,4])
        self.GCV = (M.T / M.sum(axis=1)).T

    def test_float_hellinger(self):
        D1 = graco.distance_matrices.hellinger(self.GCV)
        D2 = graco.distance_matrix(self.GCV, 'hellinger')
        D3 = squareform([graco.distances.hellinger(u,v)
                        for u,v in combinations(self.GCV,2)])

        np.testing.assert_almost_equal(D1, D2, decimal=4)
        np.testing.assert_almost_equal(D1, D3, decimal=4)
        np.testing.assert_almost_equal(D2, D3, decimal=4)

    def test_float_pdist(self):
        for distance in ['euclidean', 'cityblock', 'sqeuclidean',
                         'cosine', 'correlation', 'chebyshev',
                         'canberra', 'braycurtis']:
            D1 = graco.distance_matrix(self.GCV, distance)
            D2 = squareform(pdist(self.GCV, distance))
            D3 = squareform([graco.distance(u,v, distance)
                            for u,v in combinations(self.GCV,2)])

            np.testing.assert_allclose(D1, D2)
            np.testing.assert_allclose(D1, D3)
            np.testing.assert_allclose(D2, D3)

class TestGCVDistance(unittest.TestCase):
    def setUp(self):
        N = 2**4
        m = 2
        G = nx.barabasi_albert_graph(N,m)
        self.GCV = graco.coefficients(G)

    def test_individual_vs_all(self):
        for metric in ['canberra', 'hellinger']:
            D1 = graco.GCV_distance_matrix(self.GCV, metric)
            D2 = squareform([graco.GCV_distance(u, v, metric)
                        for (_,u),(_,v) in combinations(self.GCV.iterrows(), 2)])
            np.testing.assert_almost_equal(D1, D2, decimal=4)


if __name__ == '__main__':
    unittest.main()

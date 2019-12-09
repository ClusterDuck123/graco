from scipy.spatial.distance import squareform, pdist
from itertools import combinations
from functools import partial

import graco
import unittest
import numpy as np


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

    def test_int_lormalized1_lp(self):
        p = np.random.randint(1,10)
        D1 = graco.distance_matrices.normalized1_lp(self.GDV,p)
        D2 = graco.distance_matrix(self.GDV, 'normalized1_l' + str(p))
        D3 = squareform([graco.distances.normalized1_lp(u,v,p)
                        for u,v in combinations(self.GDV,2)])

        np.testing.assert_almost_equal(D1, D2, decimal=4)
        np.testing.assert_almost_equal(D1, D3, decimal=4)
        np.testing.assert_almost_equal(D2, D3, decimal=4)

    def test_int_lormalized1_linf(self):
        D1 = graco.distance_matrices.normalized1_lp(self.GDV, np.inf)
        D2 = graco.distance_matrix(self.GDV, 'normalized1_linf')
        D3 = squareform([graco.distances.normalized1_lp(u,v, np.inf)
                        for u,v in combinations(self.GDV,2)])

        np.testing.assert_almost_equal(D1, D2, decimal=4)
        np.testing.assert_almost_equal(D1, D3, decimal=4)
        np.testing.assert_almost_equal(D2, D3, decimal=4)

    def test_int_pdist(self):
        for distance in ['euclidean', 'cityblock', 'sqeuclidean',
                         'cosine', 'correlation', 'chebyshev',
                         'canberra', 'braycurtis']:
            D1 = graco.distance_matrix(self.GDV, distance)
            D2 = pdist(self.GDV, distance)
            np.testing.assert_almost_equal(D1, D2, decimal=4)

class TestFloatCompabilities(unittest.TestCase):
    def setUp(self):
        self.GCV = np.random.uniform(size=[100,4])

    def test_float_lormalized1_lp(self):
        p = np.random.randint(1,10)
        D1 = graco.distance_matrices.normalized1_lp(self.GCV,p)
        D2 = graco.distance_matrix(self.GCV, 'normalized1_l' + str(p))
        D3 = squareform([graco.distances.normalized1_lp(u,v,p)
                        for u,v in combinations(self.GCV,2)])

        np.testing.assert_almost_equal(D1, D2, decimal=4)
        np.testing.assert_almost_equal(D1, D3, decimal=4)
        np.testing.assert_almost_equal(D2, D3, decimal=4)

    def test_float_lormalized1_linf(self):
        D1 = graco.distance_matrices.normalized1_lp(self.GCV, np.inf)
        D2 = graco.distance_matrix(self.GCV, 'normalized1_linf')
        D3 = squareform([graco.distances.normalized1_lp(u,v, np.inf)
                        for u,v in combinations(self.GCV,2)])

        np.testing.assert_almost_equal(D1, D2, decimal=4)
        np.testing.assert_almost_equal(D1, D3, decimal=4)
        np.testing.assert_almost_equal(D2, D3, decimal=4)

    def test_float_pdist(self):
        for distance in ['euclidean', 'cityblock', 'sqeuclidean',
                         'cosine', 'correlation', 'chebyshev',
                         'canberra', 'braycurtis']:
            D1 = graco.distance_matrix(self.GCV, distance)
            D2 = pdist(self.GCV, distance)
            np.testing.assert_almost_equal(D1, D2, decimal=4)

if __name__ == '__main__':
    unittest.main()

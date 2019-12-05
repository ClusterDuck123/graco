from scipy.spatial.distance import squareform
from itertools import combinations
from functools import partial

import graco
import unittest
import numpy as np


# ============================================================================
#                              Distance Functions
# ============================================================================

class TestIntDistances(unittest.TestCase):
    def setUp(self):
        self.GDV = np.random.randint(2**62, size=[100,15])

    def test_int_GDV_similarity(self):
        D1 = graco.distance_matrix.GDV_similarity(self.GDV)
        D2 = squareform([graco.distance.GDV_similarity(u,v)
                        for u,v in combinations(self.GDV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)

    def test_int_lormalized1_l1(self):
        D1 = graco.distance_matrix.normalized1_lp(self.GDV, 1)
        D2 = squareform([graco.distance.normalized1_lp(u,v, 1)
                        for u,v in combinations(self.GDV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)

    def test_int_lormalized1_l2(self):
        D1 = graco.distance_matrix.normalized1_lp(self.GDV, 2)
        D2 = squareform([graco.distance.normalized1_lp(u,v, 2)
                        for u,v in combinations(self.GDV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)


    def test_int_lormalized1_linf(self):
        D1 = graco.distance_matrix.normalized1_lp(self.GDV, np.inf)
        D2 = squareform([graco.distance.normalized1_lp(u,v, np.inf)
                        for u,v in combinations(self.GDV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)

class TestFloatDistances(unittest.TestCase):
    def setUp(self):
        self.GCV = np.random.uniform(size=[100,4])

    def test_float_lormalized1_l1(self):
        D1 = graco.distance_matrix.normalized1_lp(self.GCV, 1)
        D2 = squareform([graco.distance.normalized1_lp(u,v, 1)
                        for u,v in combinations(self.GCV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)


    def test_float_lormalized1_l2(self):
        D1 = graco.distance_matrix.normalized1_lp(self.GCV, 2)
        D2 = squareform([graco.distance.normalized1_lp(u,v, 2)
                        for u,v in combinations(self.GCV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)


    def test_float_lormalized1_linf(self):
        D1 = graco.distance_matrix.normalized1_lp(self.GCV, np.inf)
        D2 = squareform([graco.distance.normalized1_lp(u,v, np.inf)
                        for u,v in combinations(self.GCV,2)])
        np.testing.assert_almost_equal(D1, D2, decimal=4)

if __name__ == '__main__':
    unittest.main()

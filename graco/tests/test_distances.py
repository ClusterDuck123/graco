from scipy.spatial.distance import pdist

import graco
import unittest
import numpy as np


# ============================================================================
#                              Distance Functions
# ============================================================================

all_distances = ['euclidean', 'cityblock', 'sqeuclidean',
                 'cosine', 'correlation', 'chebyshev',
                 'canberra', 'braycurtis']

class TestIntCompabilities(unittest.TestCase):
    def setUp(self):
        self.u = np.random.randint(100,size=15)
        self.v = np.random.randint(100,size=15)

    def test_int_GDV_similarity(self):
        d1 = graco.distances.GDV_similarity(self.u,self.v)
        d2 = graco.distance(self.u,self.v, 'GDV_similarity')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_normalized1_lp(self):
        p = np.random.randint(1,10)
        d1 = graco.distances.normalized1_lp(self.u,self.v,p)
        d2 = graco.distance(self.u,self.v, 'normalized1_l' + str(p))
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_normalized1_linf(self):
        d1 = graco.distances.normalized1_lp(self.u,self.v,np.inf)
        d2 = graco.distance(self.u,self.v, 'normalized1_linf')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_normalized2_lp(self):
        p = np.random.randint(1,10)
        d1 = graco.distances.normalized2_lp(self.u,self.v,p)
        d2 = graco.distance(self.u,self.v, 'normalized2_l' + str(p))
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_normalized2_linf(self):
        d1 = graco.distances.normalized2_lp(self.u,self.v,np.inf)
        d2 = graco.distance(self.u,self.v, 'normalized2_linf')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_hellinger(self):
        d1 = graco.distances.hellinger(self.u,self.v)
        d2 = graco.distance(self.u,self.v, 'hellinger')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_pdist(self):
        for distance in all_distances:
            d2 = graco.distance(self.u,self.v, distance)
            d1 = float(pdist([self.u,self.v], distance))
            np.testing.assert_almost_equal(d1, d2, decimal=4)


class TestFloatCompabilities(unittest.TestCase):
    def setUp(self):
        n = np.random.randint(100)
        self.u = np.random.uniform(size=n)
        self.v = np.random.uniform(size=n)

    def test_float_normalized1_lp(self):
        p = np.random.randint(1,10)
        d1 = graco.distances.normalized1_lp(self.u,self.v,p)
        d2 = graco.distance(self.u,self.v, 'normalized1_l' + str(p))
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_float_normalized1_linf(self):
        d1 = graco.distances.normalized1_lp(self.u,self.v,np.inf)
        d2 = graco.distance(self.u,self.v, 'normalized1_linf')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_float_normalized2_lp(self):
        p = np.random.randint(1,10)
        d1 = graco.distances.normalized2_lp(self.u,self.v,p)
        d2 = graco.distance(self.u,self.v, 'normalized2_l' + str(p))
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_float_normalized2_linf(self):
        d1 = graco.distances.normalized2_lp(self.u,self.v,np.inf)
        d2 = graco.distance(self.u,self.v, 'normalized2_linf')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_hellinger(self):
        d1 = graco.distances.hellinger(self.u,self.v)
        d2 = graco.distance(self.u,self.v, 'hellinger')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_float_pdist(self):
        for distance in all_distances:
            d2 = graco.core.distance(self.u,self.v, distance)
            d1 = float(pdist([self.u,self.v], distance))
            np.testing.assert_almost_equal(d1, d2, decimal=4)


class TestGCVDistance(unittest.TestCase):
    def setUp(self):
        self.GDV = np.array([
        #                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14
                            [2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [3, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                            ])

        self.GCV = graco.coefficients(self.GDV)

    def test_against_dummy_node(self):
        pass

#     INCOMPLETE !!!

if __name__ == '__main__':
    unittest.main()

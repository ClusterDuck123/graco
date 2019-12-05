from scipy.spatial.distance import pdist

import graco
import unittest
import numpy as np


# ============================================================================
#                              Distance Functions
# ============================================================================

class TestIntCompabilities(unittest.TestCase):
    def setUp(self):
        n = np.random.randint(100)
        self.u = np.random.randint(100,size=n)
        self.v = np.random.randint(100,size=n)

    def test_int_normalized1_lp(self):
        p = np.random.randint(10)
        d1 = graco.distances.normalized1_lp(self.u,self.v,p)
        d2 = graco.core.distance(self.u,self.v, 'normalized1_l' + str(p))
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_normalized1_linf(self):
        d1 = graco.distances.normalized1_lp(self.u,self.v,np.inf)
        d2 = graco.core.distance(self.u,self.v, 'normalized1_linf')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_cdist(self):
        for distance in ['euclidean', 'cityblock', 'sqeuclidean',
                         'cosine', 'correlation', 'chebyshev',
                         'canberra', 'braycurtis']:
            d2 = graco.core.distance(self.u,self.v, distance)
            d1 = float(pdist([self.u,self.v], distance))
            np.testing.assert_almost_equal(d1, d2, decimal=4)


class TestFloatCompabilities(unittest.TestCase):
    def setUp(self):
        n = np.random.randint(100)
        self.u = np.random.uniform(size=n)
        self.v = np.random.uniform(size=n)

    def test_int_normalized1_lp(self):
        p = np.random.randint(10)
        d1 = graco.distances.normalized1_lp(self.u,self.v,p)
        d2 = graco.core.distance(self.u,self.v, 'normalized1_l' + str(p))
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_normalized1_linf(self):
        d1 = graco.distances.normalized1_lp(self.u,self.v,np.inf)
        d2 = graco.core.distance(self.u,self.v, 'normalized1_linf')
        np.testing.assert_almost_equal(d1, d2, decimal=4)

    def test_int_cdist(self):
        for distance in ['euclidean', 'cityblock', 'sqeuclidean',
                         'cosine', 'correlation', 'chebyshev',
                         'canberra', 'braycurtis']:
            d2 = graco.core.distance(self.u,self.v, distance)
            d1 = float(pdist([self.u,self.v], distance))
            np.testing.assert_almost_equal(d1, d2, decimal=4)


if __name__ == '__main__':
    unittest.main()

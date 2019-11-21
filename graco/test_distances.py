from scipy.spatial.distance import squareform, cdist
from scipy.spatial.distance import cdist
from itertools import combinations
from functools import partial

import graco
import unittest
import numpy as np


# ============================================================================
#                              Distance Functions
# ============================================================================

def normalized1_lp(P,Q,p=1):
    v1 = np.divide(P, P+Q, out=np.zeros_like(P), where=(P+Q)!=0)
    v2 = np.divide(Q, P+Q, out=np.zeros_like(Q), where=(P+Q)!=0)
    return np.linalg.norm(v1-v2,p)

def GDV_similarity(GDV):
    LogGDV = np.log(np.array(GDV+1))

    orbit_dependencies = np.array((1,2,2,2,3,4,3,3,4,3,4,4,4,4,3))
    weights = 1 - np.log(orbit_dependencies) / np.log(len(orbit_dependencies))

    sqD = [np.sum(weights*np.abs(LogGDV[i,:]-LogGDV[j,:]) /
           np.log(np.max([GDV[i,:],GDV[j,:]], axis=0)+2))
                for (i,j) in combinations(range(len(GDV)), 2)]
    return squareform(sqD) / np.sum(weights)

class TestGDVDistances(unittest.TestCase):
    def setUp(self):
        self.GDV = np.random.randint(100, size=[100,15])

    def test_GDV_similarity(self):
        D1 = GDV_similarity(self.GDV)
        D2 = graco.distances.GDV_similarity(self.GDV)
        np.testing.assert_almost_equal(D1, D2, decimal=4)

    def test_lormalized1_l1(self):
        D1 = cdist(self.GDV.astype(float),
                   self.GDV.astype(float),
                   normalized1_lp)
        D2 = graco.distances.normalized1_lp(self.GDV)
        np.testing.assert_almost_equal(D1, D2, decimal=4)

    def test_lormalized1_l2(self):
        D1 = cdist(self.GDV.astype(float),
                   self.GDV.astype(float),
                   partial(normalized1_lp, p=2))
        D2 = graco.distances.normalized1_lp(self.GDV, 2)
        np.testing.assert_almost_equal(D1, D2, decimal=4)


    def test_lormalized1_linf(self):
        D1 = cdist(self.GDV.astype(float),
                   self.GDV.astype(float),
                   partial(normalized1_lp, p=np.inf))
        D2 = graco.distances.normalized1_lp(self.GDV, np.inf)
        np.testing.assert_almost_equal(D1, D2, decimal=4)

if __name__ == '__main__':
    unittest.main()

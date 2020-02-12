import os
import graco
import unittest
import numpy as np
import pandas as pd
import networkx as nx

GRACO_PATH = os.path.dirname(graco.__file__)
DATA_PATH   = f"{GRACO_PATH}/tests/data"

# ============================================================================
#                              Distance Functions
# ============================================================================

class TestOrca(unittest.TestCase):
    def test_against_golden_standard(self):
        GDV1 = pd.read_csv(f"{DATA_PATH}/golden_GDV.txt", index_col=0)
        GDV1.columns.name = 'Orbit'

        G = nx.read_edgelist(f"{DATA_PATH}/golden_edgelist.txt", nodetype=int)
        GDV2 = graco.orbits(G)

        pd.testing.assert_frame_equal(GDV1.sort_index(axis=0),
                                      GDV2.sort_index(axis=0))

class TestGCV(unittest.TestCase):
    def setUp(self):
        self.G   = nx.erdos_renyi_graph(2**9,0.05)
        self.GCV = graco.coefficients(self.G)

    def test_convexity(self):
        for eq in set(zip(*map(self.GCV.columns.get_level_values, [0,1]))):
            assert np.isclose(self.GCV[eq].dropna().sum(axis=1), 1).all()


if __name__ == '__main__':
    unittest.main()

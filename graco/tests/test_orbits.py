import os
import graco
import unittest
import pandas as pd
import networkx as nx

GRACO_PATH = os.path.dirname(graco.__file__)
DATA_PATH   = f"{GRACO_PATH}/tests/data"

# ============================================================================
#                              Distance Functions
# ============================================================================

class TestOrca(unittest.TestCase):
    def test_orca(self):
        GDV1 = pd.read_csv(f"{DATA_PATH}/golden_GDV.txt", index_col=0)
        GDV1.columns.name = 'Orbit'

        G = nx.read_edgelist(f"{DATA_PATH}/golden_edgelist.txt", nodetype=int)
        GDV2 = graco.orbits(G)

        pd.testing.assert_frame_equal(GDV1.sort_index(axis=0),
                                      GDV2.sort_index(axis=0))


if __name__ == '__main__':
    unittest.main()

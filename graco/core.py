import graco
import numpy as np
from scipy.spatial.distance import pdist


def distance(u,v, dist):
    if dist.startswith('normalized1_l'):
        p = dist.split('normalized1_l')[1]
        if p == 'inf':
            return graco.distances.normalized1_lp(u,v, p = np.inf)
        else:
            return graco.distances.normalized1_lp(u,v, p = int(p))
    elif dist == 'GDV_similarity':
        return graco.distances.GDV_similarity(u,v)
    else:
        return float(pdist([u,v], dist))

def distance_matrix(M, dist):
    if dist.startswith('normalized1_l'):
        p = dist.split('normalized1_l')[1]
        if p == 'inf':
            return graco.distance_matrices.normalized1_lp(M, p = np.inf)
        else:
            return graco.distance_matrices.normalized1_lp(M, p = int(p))
    elif dist == 'GDV_similarity':
        return graco.distance_matrices.GDV_similarity(M)
    else:
        return pdist(M, dist)

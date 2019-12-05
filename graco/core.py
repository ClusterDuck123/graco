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
    else:
        return float(pdist([u,v], dist))

from scipy.spatial.distance import pdist
import numpy as np
import sys

def normalized1_lp(u,v,p=1):
    u0 = np.divide(u, abs(u)+abs(v),
                   out   = np.zeros(u.shape),
                   where = (u+v)!=0)
    v0 = np.divide(v, abs(u)+abs(v),
                   out   = np.zeros(v.shape),
                   where = (u+v)!=0)
    return np.linalg.norm(u0-v0, p)


def GDV_similarity(u,v):
    log_u = np.log(np.array(u+1))
    log_v = np.log(np.array(v+1))

    orbit_dependencies = np.array((1,2,2,2,3,4,3,3,4,3,4,4,4,4,3))
    weights = 1 - np.log(orbit_dependencies) / np.log(len(orbit_dependencies))

    res = np.sum(weights*np.abs(log_u-log_v) /
          np.log(np.max([u,v], axis=0)+2))
    return res / np.sum(weights)

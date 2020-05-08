from scipy.spatial.distance import pdist
import numpy as np
import sys

_SQRT2 = np.sqrt(2)

def hellinger(p, q):
    p = p/np.sum(p)
    q = q/np.sum(q)
    tmp = np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)
    return np.sqrt(tmp) / _SQRT2

def js_divergence(p,q):
    p = p/np.sum(p)
    q = q/np.sum(q)
    m = (p+q)/2

    log_arg1 = np.true_divide(p,m, out   = np.ones_like(p),
                                   where = p!=0)
    log_arg2 = np.true_divide(q,m, out   = np.ones_like(q),
                                   where = q!=0)

    log1 = np.log(log_arg1)
    log2 = np.log(log_arg2)

    return np.sum(p*log1 +q*log2)/(2*np.log(2))


def GDV_similarity(u,v):
    log_u = np.log(np.array(u+1))
    log_v = np.log(np.array(v+1))

    orbit_dependencies = np.array((1,2,2,2,3,4,3,3,4,3,4,4,4,4,3))
    weights = 1 - np.log(orbit_dependencies) / np.log(len(orbit_dependencies))

    res = np.sum(weights*np.abs(log_u-log_v) /
          np.log(np.max([u,v], axis=0)+2))
    return res / np.sum(weights)

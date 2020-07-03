from scipy.spatial.distance import cdist, pdist, squareform

import ot
import graco
import scipy
import numpy as np
import pandas as pd
import networkx as nx

def triangle_signature(G):
    """
    Returns clustering coefficient and community coefficient.
    This function will ignore self-loops.
    """
    A = nx.to_scipy_sparse_matrix(G, format='lil')
    A.setdiag(0)
    A = scipy.sparse.csc_matrix(A)
    A2 = A@A
    o0 = A.sum(axis=1).A
    numer = A.multiply(A2).sum(axis=1).A

    D_denom = o0*o0 - o0
    A_denom = A @o0 - o0

    C_D = np.true_divide(numer, D_denom,
                         out   = np.nan*np.empty_like(numer, dtype=float),
                         where = D_denom!=0).flatten()
    C_A = np.true_divide(numer, A_denom,
                         out   = np.nan*np.empty_like(numer, dtype=float),
                         where = A_denom!=0).flatten()

    return C_D, C_A

def GDV_to_GCM11(GDV):
    GCM11 = GDV.loc[:, map(str,set(range(15)) - {3,12,13,14})]
    GCM11.loc['NULL'] = 1.
    return GCM11

def GCD11(G1, G2, metric='euclidean'):
    GDV1, GDV2 = graco.orbits(G1)  , graco.orbits(G2)
    GCM1 = squareform(GDV_to_GCM11(GDV1).corr(), checks=False)
    GCM2 = squareform(GDV_to_GCM11(GDV2).corr(), checks=False)
    return distance(GCM1,GCM2,metric)

def emd(xs, xt, metric='euclidean'):
    if len(xs) > len(xt):
        xs, xt = xt, xs

    M  = cdist(xs,xt,metric)

    if (M == 0).all():
        return 0.

    M2 = M**2

    for j in range(10):
        a = np.ones(len(xs))/len(xs)
        b = np.ones(len(xt))/len(xt)

        F = ot.emd(a,b, M2)

        if np.isclose(np.sum(F), 1):
            break
        else:
            xs = np.vstack([xs,xs])
            M  = np.vstack([M,M])
            M2 = np.vstack([M2,M2])

    assert np.isclose(np.sum(F), 1)

    return np.sum(M*F)

def triangle_distance(Gs, Gt, metric='euclidean'):
    xs = np.nan_to_num(np.array(triangle_signature(Gs)).T)
    xt = np.nan_to_num(np.array(triangle_signature(Gt)).T)
    return emd(xs,xt, metric)

def iter_equations(GCV):
    lowest_two_levels = list(range(GCV.columns.nlevels-1))
    for eq, coeffs in GCV.groupby(
                                level = lowest_two_levels,
                                axis  = 1):
        yield eq

def iter_equation_coefficients(GCV):
    lowest_two_levels = list(range(GCV.columns.nlevels-1))
    for eq, coeffs in GCV.groupby(
                                level = lowest_two_levels,
                                axis  = 1):
        yield eq, coeffs

def _get_value(value, coeffs):
    if   value == 'barycenter':
        return 1/len(coeffs.T)
    elif value == 'mean':
        return coeffs.mean()
    else:
        return value

def fill_nan(GCV, value='barycenter'):

    lowest_two_levels = list(range(GCV.columns.nlevels-1))

    if not lowest_two_levels:
#       CALCULATE WITHOUT LOOP
        pass
    else:
        for eq, coeffs in GCV.groupby(
                                level = lowest_two_levels,
                                axis  = 1):
            GCV.loc[:,eq] = coeffs.fillna(_get_value(value,coeffs))

def normalizer(metric, length):
    if   metric == 'cityblock'   : return 2
    elif metric == 'euclidean'   : return np.sqrt(2)
    elif metric == 'sqeuclidean' : return 2
    elif metric == 'chebyshev'   : return 1

    elif metric == 'cosine'      : return 1
    elif metric == 'correlation' : return 2

    elif metric == 'canberra'   : return length
    elif metric == 'braycurtis' : return 1

    elif metric == 'mahalanobis': return 1
    elif metric == 'seuclidean' : return 1

    elif metric == 'hellinger'     : return 1
    elif metric == 'js_divergence' : return 1

    else: raise Exception(f"Metric {metric} not known!")

def distance(u,v, metric):
    if   metric == 'GDV_similarity':
        return graco.distances.GDV_similarity(u,v)
    elif metric == 'hellinger':
        return graco.distances.hellinger(u,v)
    elif metric == 'js_divergence':
        return graco.distances.js_divergence(u,v)
    else:
        u, v = np.array(u).flatten(), np.array(v).flatten()
        return float(pdist([u,v], metric))

def convex_distance(u,v,metric):
    """
    Normalized distance of two convex vectors (function asserts this).
    Not suited for NaN entries.
    """

    assert np.isclose(sum(u),1), "u is not a convex combination! ({sum(u)})"
    assert np.isclose(sum(v),1), "v is not a convex combination! ({sum(v)})"

    if metric == 'seuclidean':
        return distance(u,v,metric)
    else:
        return distance(u,v,metric) / normalizer(metric, len(u))


def distance_matrix(M, metric):
    if   metric == 'GDV_similarity':
        return graco.distance_matrices.GDV_similarity(M)
    elif metric == 'hellinger':
        return graco.distance_matrices.hellinger(M)
    elif metric == 'js_divergence':
        return graco.distance_matrices.js_divergence(M)
    else:
        return squareform(pdist(M, metric))

def GCV_distance(u, v, metric, nan='include'):
    gdv = pd.concat([u, v], axis=1).T.dropna(axis=1)
    return np.mean([convex_distance(u[eq], v[eq], metric)
                                                for eq in iter_equations(gdv)])

def GCV_distance_matrix(GCV, distance, nan='include'):
    """
    Calculates distances equation-wise with optional parameter to controll the
    behaviour of NaNs. Carefull with 'correlation-distances', since they require
    a positive variation in the coefficients within an equation.
    """
    if nan == 'include':
        if type(GCV.columns) == pd.MultiIndex:
            D_all   = pd.DataFrame(0, index=GCV.index, columns=GCV.index)
            Divisor = pd.DataFrame(0, index=GCV.index, columns=GCV.index)

            for eq, coeffs in iter_equation_coefficients(GCV):
                gcv = coeffs.dropna()
                if gcv.empty:
                    continue

                not_nan_indices = gcv.index
                nan_indices = GCV.index[coeffs.isna().any(axis=1)]

                assert (coeffs.isna().any(axis=1) == coeffs.isna().all(axis=1)).all()
                assert len(nan_indices) + len(not_nan_indices) == len(GCV)

                D_sub = graco.distance_matrix(gcv, distance)
                if distance == 'seuclidean':
                    s1, s2 = sorted(gcv.var(ddof=1))[:2]
                    d_max = np.sqrt(1/s1 +1/s2)
                else:
                    d_max = normalizer(distance,len(gcv.T))
                D_all.loc[  not_nan_indices, not_nan_indices] += D_sub / d_max
                Divisor.loc[not_nan_indices, not_nan_indices] += 1

            return D_all / Divisor
        else:
            D = pd.DataFrame(np.nan, index   = GCV.index,
                                     columns = GCV.index)
            length = len(GCV.T)
            gcv = GCV.dropna()
            not_nan_indices = gcv.index
            nan_indices = GCV.index[GCV.isna().any(axis=1)]

            assert (GCV.isna().any(axis=1) == GCV.isna().all(axis=1)).all()
            assert len(nan_indices) + len(not_nan_indices) == len(GCV)

            D_sub = graco.distance_matrix(gcv, distance)
            if distance == 'seuclidean':
                s1, s2 = sorted(gcv.var(ddof=1))[:2]
                d_max = np.sqrt(1/s1 +1/s2)
            else:
                d_max = normalizer(distance,length)
            D.loc[not_nan_indices,not_nan_indices] = D_sub / d_max
            return D
    else:
        raise Exception("Only 'nan=include' implemented yet!")

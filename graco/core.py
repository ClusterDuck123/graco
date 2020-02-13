from scipy.spatial.distance import pdist, squareform

import graco
import numpy as np
import pandas as pd

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

def normalizer(distance, length):
    if   distance == 'normalized1_l1'  : return length
    elif distance == 'normalized1_l2'  : return np.sqrt(length)
    elif distance == 'normalized1_linf': return 1

    elif distance == 'normalized2_l1'  : return length
    elif distance == 'normalized2_l2'  : return np.sqrt(2)
    elif distance == 'normalized2_linf': return 1

    elif distance == 'cityblock'   : return 2
    elif distance == 'euclidean'   : return np.sqrt(2)
    elif distance == 'sqeuclidean' : return 2
    elif distance == 'chebyshev'   : return 1

    elif distance == 'cosine'      : return 1
    elif distance == 'correlation' : return 2

    elif distance == 'canberra'   : return length
    elif distance == 'braycurtis' : return 1

    elif distance == 'mahalanobis': return 1
    elif distance == 'seuclidean' : return 1

    elif distance == 'hellinger' : return 1

def distance(u,v, dist):
    if   dist.startswith('normalized1_l'):
        p = dist.split('normalized1_l')[1]
        if p == 'inf':
            return graco.distances.normalized1_lp(u,v, p = np.inf)
        else:
            return graco.distances.normalized1_lp(u,v, p = int(p))
    elif dist.startswith('normalized2_l'):
        p = dist.split('normalized2_l')[1]
        if p == 'inf':
            return graco.distances.normalized2_lp(u,v, p = np.inf)
        else:
            return graco.distances.normalized2_lp(u,v, p = int(p))
    elif dist == 'GDV_similarity':
        return graco.distances.GDV_similarity(u,v)
    elif dist == 'hellinger':
        return graco.distances.hellinger(u,v)
    else:
        return float(pdist([u,v], dist))


def distance_matrix(M, dist):
    if   dist.startswith('normalized1_l'):
        p = dist.split('normalized1_l')[1]
        if p == 'inf':
            return graco.distance_matrices.normalized1_lp(M, p = np.inf)
        else:
            return graco.distance_matrices.normalized1_lp(M, p = int(p))
    elif dist.startswith('normalized2_l'):
        p = dist.split('normalized2_l')[1]
        if p == 'inf':
            return graco.distance_matrices.normalized2_lp(M, p = np.inf)
        else:
            return graco.distance_matrices.normalized2_lp(M, p = int(p))
    elif dist == 'GDV_similarity':
        return graco.distance_matrices.GDV_similarity(M)
    elif dist == 'hellinger':
        return graco.distance_matrices.hellinger(M)
    else:
        return squareform(pdist(M, dist))

def GCV_distance(u, v, distance, nan='include'):
    gdv = pd.concat([u, v], axis=1).T.dropna(axis=1)

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
                D_all.loc[not_nan_indices,not_nan_indices] += \
                                            D_sub / normalizer(distance,len(gcv.T))
                Divisor.loc[not_nan_indices,not_nan_indices] += 1

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
            D.loc[not_nan_indices,not_nan_indices] = \
                                            D_sub / normalizer(distance,length)
            return D
    else:
        raise Exception

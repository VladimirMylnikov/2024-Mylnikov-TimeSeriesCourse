import numpy as np

from modules.utils import *
import stumpy

def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """
    if ts2 is None:
        # Cчитаем матричный профиль для одного ряда 
        mp = stumpy.stump(ts1, m)
    else:
        # Cчитаем матричный профиль между двумя временными рядами 
        mp = stumpy.stump(ts1, m, ts2)

    return {'mp': mp[:, 0],
            'mpi': mp[:, 1],
            'm' : m,
            'excl_zone': exclusion_zone,
            'data': {'ts1' : ts1, 'ts2' : ts2}
            }

def top_k_motifs(mp, ts, top_k=3):
    """
    Find top-k motifs in the time series using the matrix profile.

    Parameters
    ----------
    mp: matrix profile structure
    ts: the time series
    top_k: number of top motifs to find

    Returns
    -------
    output: dictionary with 'indices' and 'distances' of the top-k motifs
    """
    motifs = stumpy.motifs(ts, mp['mp'], max_matches=top_k)
    indices = [(motifs[1][0][i], motifs[1][0][i + 1]) for i in range(0, len(motifs[1][0]) - 1, 2)]
    distances = motifs[0]
    return {'indices': indices, 'distances': distances}

import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """

    if ts1.shape != ts2.shape:
        raise ValueError("Временные ряды должны быть одной длины!")
    
    ed_dist = 0

    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    n = len(ts1)
    
    mu_ts1 = np.mean(ts1)
    mu_ts2 = np.mean(ts2)
    sigma_ts1 = np.std(ts1)
    sigma_ts2 = np.std(ts2)
    
    dot_product = np.dot(ts1, ts2)
    
    norm_ed_dist = np.sqrt(np.abs(2 * n * (1 - (dot_product - n * mu_ts1 * mu_ts2) / (n * sigma_ts1 * sigma_ts2))))
    
    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n = len(ts1)
    m = len(ts2)

    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[:, :] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(max(1, i-int(np.floor(m*r))), min(m, i+int(np.floor(m*r))) + 1):
            cost = np.square(ts1[i-1] - ts2[j-1])
            dtw_matrix[i, j] = cost + \
                min(dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1])

    dtw_dist = dtw_matrix[n, m]

    return dtw_dist
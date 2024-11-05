import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
    mp = matrix_profile['mp']
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']
    
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    for _ in range(top_k):
        # Найти индекс с максимальным значением в матричном профиле
        discord_idx = np.argmax(mp)
        discord_dist = mp[discord_idx]
        nn_idx = mpi[discord_idx]
        
        # Добавить найденный диссонанс в список
        discords_idx.append(discord_idx)
        discords_dist.append(discord_dist)
        discords_nn_idx.append(nn_idx)
        
        # Применить зону исключения
        mp = apply_exclusion_zone(mp, discord_idx, excl_zone, np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
    }
import numpy as np
import math
import copy

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw

def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """
    Apply an exclusion zone to an array (inplace)
    
    Parameters
    ----------
    array: the array to apply the exclusion zone to
    idx: the index around which the window should be centered
    excl_zone: size of the exclusion zone
    
    Returns
    -------
    array: the array which is applied the exclusion zone
    """

    zone_start = max(0, idx - excl_zone)
    zone_stop = min(array.shape[-1], idx + excl_zone)
    array[zone_start : zone_stop + 1] = np.inf

    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3, max_distance: float = np.inf) -> dict:
    """
    Search the topK match subsequences based on distance profile
    
    Parameters
    ----------
    dist_profile: distances between query and subsequences of time series
    excl_zone: size of the exclusion zone
    topK: count of the best match subsequences
    max_distance: maximum distance between query and a subsequence `S` for `S` to be considered a match
    
    Returns
    -------
    topK_match_results: dictionary containing results of algorithm
    """

    topK_match_results = {
        'indices': [],
        'distances': []
    } 

    dist_profile_len = len(dist_profile)
    dist_profile = np.copy(dist_profile).astype(float)

    for k in range(topK):
        min_idx = np.argmin(dist_profile)
        min_dist = dist_profile[min_idx]

        if (np.isnan(min_dist)) or (np.isinf(min_dist)) or (min_dist > max_distance):
            break

        dist_profile = apply_exclusion_zone(dist_profile, min_idx, excl_zone)

        topK_match_results['indices'].append(min_idx)
        topK_match_results['distances'].append(min_dist)

    return topK_match_results


class BestMatchFinder:
    """
    Base Best Match Finder
    
    Parameters
    ----------
    excl_zone_frac: exclusion zone fraction
    topK: number of the best match subsequences
    is_normalize: z-normalize or not subsequences before computing distances
    r: warping window size
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05) -> None:
        """ 
        Constructor of class BestMatchFinder
        """

        self.excl_zone_frac: float = excl_zone_frac
        self.topK: int = topK
        self.is_normalize: bool = is_normalize
        self.r: float = r

    def _calculate_excl_zone(self, m: int) -> int:
        """
        Calculate the exclusion zone
        
        Parameters
        ----------
        m: length of subsequence
        
        Returns
        -------
        excl_zone: exclusion zone
        """
        return math.ceil(m * self.excl_zone_frac)

    def perform(self):
        raise NotImplementedError


class BestMatchPredictor(BestMatchFinder):
    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05, aggr_func: str = 'average') -> None:
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        self.aggr_func = aggr_func

    def _z_normalize(self, subsequence):
        return (subsequence - np.mean(subsequence)) / np.std(subsequence)



    def _calculate_distance(self, Q, subsequence):
        if self.is_normalize:
            Q = self._z_normalize(Q)
            subsequence = self._z_normalize(subsequence)

        # Ensure both are 1-D arrays
        Q = np.ravel(Q)
        subsequence = np.ravel(subsequence)

        # Use dtaidistance instead of fastdtw
        distance = dtw.distance(Q, subsequence)
        return distance

    def perform(self, T_train, Q, h):
        excl_zone = self._calculate_excl_zone(len(Q))
        distances = []

        for i in range(len(T_train) - len(Q) + 1):
            if i < excl_zone:
                continue
            subsequence = T_train[i:i+len(Q)]
            distance = self._calculate_distance(Q, subsequence)
            distances.append((distance, i))

        distances.sort(key=lambda x: x[0])
        topK_indices = [idx for _, idx in distances[:self.topK]]

        predictions = []
        for idx in topK_indices:
            future_values = T_train[idx+len(Q):idx+len(Q)+h]
            # Only include future values if they match the length of h
            if len(future_values) == h:
                predictions.append(future_values)

        if len(predictions) == 0:
            raise ValueError("No valid predictions found. Ensure enough future data exists for the forecast horizon.")

        predictions = np.array(predictions)
        
        if self.aggr_func == 'average':
            return np.mean(predictions, axis=0)
        elif self.aggr_func == 'median':
            return np.median(predictions, axis=0)
        else:
            raise ValueError("Unsupported aggregation function")



class NaiveBestMatchFinder(BestMatchFinder):
    """
    Naive Best Match Finder
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class NaiveBestMatchFinder
        """


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using the naive algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of the naive algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape
        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf

        bestmatch = {
            'indices' : [],
            'distance' : []
        }
        
        # Normalize query if needed
        if self.is_normalize:
            query = z_normalize(query)

        # Compute distance profile for each subsequence
        for i in range(N):
            subsequence = ts_data[i]
            if self.is_normalize:
                subsequence = z_normalize(subsequence)
            distance = DTW_distance(subsequence, query, r=self.r)
            dist_profile[i] = distance

        # Find topK matches
        topK_results = topK_match(dist_profile, excl_zone, self.topK, max_distance=bsf)

        bestmatch['indices'] = topK_results['indices']
        bestmatch['distance'] = topK_results['distances']



        return bestmatch


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        return (subs1[0] - subs2[0])**2 + (subs1[-1] - subs2[-1])**2



    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size
        
        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        n = len(subs1)
        lb_Keogh = 0
        
        # Calculate upper and lower envelopes for subs1
        U = np.zeros(n)
        L = np.zeros(n)
        for i in range(n):
            U[i] = np.max(subs1[max(0, i-r):min(n, i+r+1)])
            L[i] = np.min(subs1[max(0, i-r):min(n, i+r+1)])
        
        # Calculate LB_Keogh
        for i in range(n):
            if subs2[i] > U[i]:
                lb_Keogh += (subs2[i] - U[i])**2
            elif subs2[i] < L[i]:
                lb_Keogh += (subs2[i] - L[i])**2
        
        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf
        
        bestmatch = {
            'indices' : [],
            'distance' : []
        }

        for i in range(N):
            if i < excl_zone:
                continue
            
            subs = ts_data[i]
            
            # Step 1: LB_KimFL
            lb_Kim = self._LB_Kim(query, subs)
            self.lb_Kim_num += 1
            if lb_Kim > bsf:
                continue
            
            # Step 2: LB_KeoghQC
            lb_KeoghQC = self._LB_Keogh(query, subs, int(self.r * m))
            self.lb_KeoghQC_num += 1
            if lb_KeoghQC > bsf:
                continue
            
            # Step 3: LB_KeoghCQ
            lb_KeoghCQ = self._LB_Keogh(subs, query, int(self.r * m))
            self.lb_KeoghCQ_num += 1
            if lb_KeoghCQ > bsf:
                continue
            
            # Step 4: DTW
            dtw_dist = DTW_distance(query, subs, self.r)
            self.not_pruned_num += 1
            
            if dtw_dist < bsf:
                bsf = dtw_dist
                bestmatch['indices'].append(i)
                bestmatch['distance'].append(dtw_dist)
        
        # Sort results by distance
        bestmatch['indices'] = [x for _, x in sorted(zip(bestmatch['distance'], bestmatch['indices']))]
        bestmatch['distance'] = sorted(bestmatch['distance'])
        
        return bestmatch
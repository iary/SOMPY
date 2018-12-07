import itertools
from functools import partial
import warnings

import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.preprocessing import normalize
### when we will parallelize:
#from sklearn.utils import gen_even_slices
#from sklearn.externals.joblib import Parallel
#from sklearn.externals.joblib import delayed
#from sklearn.externals.joblib import cpu_count


_VALID_METRICS = ['euclidean']
                  ## All the other metrics still not implemented:
                  #  'l2', 'l1', 'manhattan', 'cityblock',
                  #'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  #'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  #'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  #'russellrao', 'seuclidean', 'sokalmichener',
                  #'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]

def euclidean_chidistances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None, sigma=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::
        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.
    However, this is not the most precise way of doing this computation, and
    the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.
    Read more in the :ref:`User Guide <metrics>`.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)
    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)
    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
    squared : boolean, optional
        Return squared Euclidean distances.
    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
    Returns
    -------
    distances : {array, sparse matrix}, shape (n_samples_1, n_samples_2)
    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    X, Y = check_pairwise_arrays(X, Y)
    if sigma is not None:
        if type(sigma) is not np.ndarray : 
            sigma = np.array(sigma)
        if sigma.shape!=X.shape:
            raise ValueError("Std dev array 'sigma' must have same shape of data array 'X' ")
    else:
        print 'Euclidean distance with chi^2 weights, but errors NOT DEFINED!!'
        sigma = np.ones(X.shape)
    #IY: X is always a np.ndarry (at the moment)
    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
    else:
        XX = row_norms(X/sigma, squared=True)[:, np.newaxis]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        print "X is Y"
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]
    
    distances = safe_sparse_dot(X/sigma/sigma, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    for i in range(distances.shape[0]):
        distances[i,:] += row_norms(Y/sigma[i,:], squared=True)
    
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)

#In this wrapper we allow for one dist func only (euclidean)
PAIRWISE_DISTANCE_FUNCTIONS = {
    #'cityblock': manhattan_distances,
    #'cosine': cosine_distances,
    'euclidean': euclidean_chidistances,
    #'l2': euclidean_distances,
    #'l1': manhattan_distances,
    #'manhattan': manhattan_distances,
    #'precomputed': None,  # HACK: precomputed is always allowed, never called
}


def pairwise_chidistances(X, Y=None, metric="euclidean", n_jobs=1, sigma=None, **kwds):
    """ *** This is a modified version of pairwise_distance() *** 
    It has been devised to work with sompy package in the assumption of
    euclidean metric. Then, distances are always computed in euclidean style,
    see example below. 
    Compute the distance matrix from a vector array X and optional Y.
    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.
    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.
    Parameters  AS CALLED BY SOMPY
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise \
        Feature array with new data to map.
    Y : array [n_samples_b, n_features] \
        An optional second feature array. 
    metric : default is "euclidean", CANNOT BE CHANGED at the moment
    n_jobs : int  NOT IMPLEMENTED YET
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    E : array [n_samples_a] \
        Optional array with the VARIANCE (sigma**2) of each element of X
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.
    See also
    --------
    pairwise_distances_chunked : performs the same calculation as this funtion,
        but returns a generator of chunks of the distance matrix, in order to
        limit memory usage.
    paired_distances : Computes the distances between corresponding
                       elements of two arrays
    """
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]   #IY: func is only euclidian_distance at the moment
    else:
        raise TypeError("In this beta version only euclidian metric is supported")

    # TODO: parallelize
    #return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
    return func(X, Y, sigma=sigma, **kwds)


def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    """ Set X and Y appropriately and checks inputs
    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.
    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.
    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.
        .. versionadded:: 0.18
    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.
    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    X, Y, dtype_float = _return_float_dtype(X, Y)

    warn_on_dtype = dtype is not None
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype,
                            warn_on_dtype=warn_on_dtype, estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype,
                        warn_on_dtype=warn_on_dtype, estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y

# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype


def check_paired_arrays(X, Y):
    """ Set X and Y appropriately and checks inputs for paired distances
    All paired distance metrics should use this function first to assert that
    the given parameters are correct and safe to use.
    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the dimensions of the two arrays are equal.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.
    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    X, Y = check_pairwise_arrays(X, Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y should be of same shape. They were "
                         "respectively %r and %r long." % (X.shape, Y.shape))
    return X, Y


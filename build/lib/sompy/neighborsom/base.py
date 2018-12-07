"""Base and mixin classes for nearest neighbors"""
# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck
#          Multi-output support by Arnaud Joly <a.joly@ulg.ac.be>
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, issparse

from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.kd_tree import KDTree
from sklearn.base import BaseEstimator
#from sklearn.metrics import pairwise_distances
from pairwisesom import pairwise_chidistances
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils import check_X_y, check_array, _get_n_jobs, gen_even_slices
from sklearn.utils.multiclass import check_classification_targets
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.exceptions import NotFittedError
from sklearn.exceptions import DataConversionWarning

VALID_METRICS = dict(ball_tree=BallTree.valid_metrics,
                     kd_tree=KDTree.valid_metrics,
                     # The following list comes from the
                     # sklearn.metrics.pairwise doc string
                     brute=(list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) +
                            ['braycurtis', 'canberra', 'chebyshev',
                             'correlation', 'cosine', 'dice', 'hamming',
                             'jaccard', 'kulsinski', 'mahalanobis',
                             'matching', 'minkowski', 'rogerstanimoto',
                             'russellrao', 'seuclidean', 'sokalmichener',
                             'sokalsneath', 'sqeuclidean',
                             'yule', 'wminkowski']))


VALID_METRICS_SPARSE = dict(ball_tree=[],
                            kd_tree=[],
                            brute=PAIRWISE_DISTANCE_FUNCTIONS.keys())


def _check_weights(weights):
    """Check to make sure weights are valid"""
    if weights in (None, 'uniform', 'distance'):
        return weights
    elif callable(weights):
        return weights
    else:
        raise ValueError("weights not recognized: should be 'uniform', "
                         "'distance', or a callable function")


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``

    Parameters
    ===========
    dist : ndarray
        The input distances
    weights : {'uniform', 'distance' or a callable}
        The kind of weighting used

    Returns
    ========
    weights_arr : array of the same shape as ``dist``
        if ``weights == 'uniform'``, then returns None
    """
    if weights in (None, 'uniform'):
        return None
    elif weights == 'distance':   #IY notes
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, '__contains__') and 0. in point_dist:
                    dist[point_dist_i] = point_dist == 0.
                else:
                    dist[point_dist_i] = 1. / point_dist
        else:
            with np.errstate(divide='ignore'):
                dist = 1. / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist
    elif callable(weights):
        return weights(dist)
    else:
        raise ValueError("weights not recognized: should be 'uniform', "
                         "'distance', or a callable function")


class NeighborsBase(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for nearest neighbors estimators."""

    @abstractmethod
    def __init__(self):
        pass

    def _init_params(self, n_neighbors=None, radius=None,
                     algorithm='auto', leaf_size=30, metric='minkowski',
                     p=2, metric_params=None, n_jobs=1):

        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

        if algorithm not in ['auto', 'brute',
                             'kd_tree', 'ball_tree']:
            raise ValueError("unrecognized algorithm: '%s'" % algorithm)

        if algorithm == 'auto':
            if metric == 'precomputed':
                alg_check = 'brute'
            elif callable(metric) or metric in VALID_METRICS['ball_tree']:
                alg_check = 'ball_tree'
            else:
                alg_check = 'brute'
        else:
            alg_check = algorithm

        if callable(metric):
            if algorithm == 'kd_tree':
                # callable metric is only valid for brute force and ball_tree
                raise ValueError(
                    "kd_tree algorithm does not support callable metric '%s'"
                    % metric)
        elif metric not in VALID_METRICS[alg_check]:
            raise ValueError("Metric '%s' not valid for algorithm '%s'"
                             % (metric, algorithm))

        if self.metric_params is not None and 'p' in self.metric_params:
            warnings.warn("Parameter p is found in metric_params. "
                          "The corresponding parameter from __init__ "
                          "is ignored.", SyntaxWarning, stacklevel=3)
            effective_p = metric_params['p']
        else:
            effective_p = self.p

        if self.metric in ['wminkowski', 'minkowski'] and effective_p < 1:
            raise ValueError("p must be greater than one for minkowski metric")

        self._fit_X = None
        self._tree = None
        self._fit_method = None

    def _fit(self, X):
        #use Euclidean metric if possible, or raise error [IY]
        #note that in sompy.project_realdata() the algorithm is set by default
        # (e.g. to 'brute' or 'kd_tree')
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        if self.metric not in ['euclidean', 'minkowski']:
            raise ValueError("Using Euclidean distance with the wrong metric") 
        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == 'minkowski':
            p = self.effective_metric_params_.pop('p', 2)
            if p == 2:
                self.effective_metric_ = 'euclidean'
            else :
                raise ValueError("cannot replace Minkowski with Euclidian metric")

        X = check_array(X, accept_sparse='csr')

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("n_samples must be greater than 0")

        if issparse(X) and self.effective_metric_ not in VALID_METRICS_SPARSE['brute']:
            raise ValueError("metric '%s' not valid for sparse input"
                                 % self.effective_metric_)

        self._fit_method = self.algorithm
        self._fit_X = X

        if self._fit_method == 'ball_tree':
            self._tree = BallTree(X, self.leaf_size,
                                  metric=self.effective_metric_,
                                  **self.effective_metric_params_)
        elif self._fit_method == 'kd_tree':
            self._tree = KDTree(X, self.leaf_size,
                                metric=self.effective_metric_,
                                **self.effective_metric_params_)
        elif self._fit_method == 'brute':
            self._tree = None
        else:
            raise ValueError("algorithm = '%s' not recognized"
                             % self.algorithm)

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    "Expected n_neighbors > 0. Got %d" %
                    self.n_neighbors
                )

        return self

    @property
    def _pairwise(self):
        # For cross-validation routines to split data correctly
        return self.metric == 'precomputed'


class KNeighborsMixin(object):
    """Mixin for k-neighbors searches"""

    def kneighbors(self, X=None, E=None, n_neighbors=None, return_distance=True):  #IY modified to account std dev
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True

        ind : array
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]

        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> print(neigh.kneighbors([[1., 1., 1.]])) # doctest: +ELLIPSIS
        (array([[ 0.5]]), array([[2]]...))

        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:

        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False) # doctest: +ELLIPSIS
        array([[1],
               [2]]...)

        """
        if self._fit_method is None:  #IY: usually in SOMPY is set to 'auto'
            raise NotFittedError("Must fit neighbors before querying.")

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr')
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later. Results don't consider the test point itself as a neighbor
            n_neighbors += 1
        train_size = self._fit_X.shape[0]
        if n_neighbors > train_size:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (train_size, n_neighbors)
            )
        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, None]

        n_jobs = _get_n_jobs(self.n_jobs)  #IY: single core at the moment
        if self._fit_method == 'brute':
            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == 'euclidean':
                dist = pairwise_chidistances(X, self._fit_X, 'euclidean',  #IY
                                          n_jobs=n_jobs, sigma=E, squared=True)
            else:
                raise ValueError("kneighbor for project_realdata() works only"
                                 " with euclidean metric")

            neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
            neigh_ind = neigh_ind[:, :n_neighbors]
            # argpartition doesn't guarantee sorted order, so we sort again
            neigh_ind = neigh_ind[
                sample_range, np.argsort(dist[sample_range, neigh_ind])]

            if return_distance:
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = neigh_ind

        ## IY: only brute force at the moment...
        #elif self._fit_method in ['ball_tree', 'kd_tree']:
        #    if issparse(X):
        #        raise ValueError(
        #            "%s does not work with sparse matrices. Densify the data, "
        #            "or set algorithm='brute'" % self._fit_method)
        #    result = Parallel(n_jobs, backend='threading')(
        #        delayed(self._tree.query, check_pickle=False)(
        #            X[s], n_neighbors, return_distance)
        #        for s in gen_even_slices(X.shape[0], n_jobs)
        #    )
        #    if return_distance:
        #        dist, neigh_ind = tuple(zip(*result))
        #        result = np.vstack(dist), np.vstack(neigh_ind)
        #    else:
        #        result = np.vstack(result)
        else:
            raise ValueError("only brute force algorithm accepted as _fit_method ")

        if not query_is_train:
            return result
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = result
            else:
                neigh_ind = result

            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False

            neigh_ind = np.reshape(
                neigh_ind[sample_mask], (n_samples, n_neighbors - 1))

            if return_distance:
                dist = np.reshape(
                    dist[sample_mask], (n_samples, n_neighbors - 1))
                return dist, neigh_ind
            return neigh_ind


class SupervisedIntegerMixin(object):
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        if not isinstance(X, (KDTree, BallTree)):
            X, y = check_X_y(X, y, "csr", multi_output=True)

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn("A column-vector y was passed when a 1d array "
                              "was expected. Please change the shape of y to "
                              "(n_samples, ), for example using ravel().",
                              DataConversionWarning, stacklevel=2)

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        check_classification_targets(y)
        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        return self._fit(X)


class UnsupervisedMixin(object):
    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.
        """
        return self._fit(X)

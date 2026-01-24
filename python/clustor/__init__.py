# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


"""Clustor: Rust-accelerated clustering with a Python-friendly API.

Clustor is intentionally different from dataset balancing libraries. Use it for fast
unsupervised learning primitives inside ML pipelines.
"""

from __future__ import annotations

import numpy as _np

from ._clustor import DBSCAN as _DBSCAN
from ._clustor import OPTICS as _OPTICS
from ._clustor import AffinityPropagation as _AffinityPropagation
from ._clustor import Birch as _Birch
from ._clustor import BisectingKMeans as _BisectingKMeans
from ._clustor import GaussianMixture as _GaussianMixture
from ._clustor import KMeans as _KMeans  # type: ignore
from ._clustor import MiniBatchKMeans as _MiniBatchKMeans
from ._clustor import (
    __version__,
)
from ._clustor import affinity_propagation as _affinity_propagation
from ._clustor import birch as _birch
from ._clustor import calinski_harabasz_score as _calinski_harabasz_score
from ._clustor import davies_bouldin_score as _davies_bouldin_score
from ._clustor import dbscan as _dbscan
from ._clustor import gaussian_mixture as _gaussian_mixture
from ._clustor import hac_dendrogram as _hac_dendrogram
from ._clustor import kmeans as _kmeans
from ._clustor import optics as _optics
from ._clustor import silhouette_score as _silhouette_score


def _as_f64_2d(x) -> _np.ndarray:
    """Coerce input data to contiguous float64 2D arrays.

    Args:
        x: Array-like input of shape (n_samples, n_features).

    Returns:
        np.ndarray: Contiguous array with dtype float64 and shape (n_samples, n_features).

    Examples:
        >>> import numpy as np
        >>> from clustor import _as_f64_2d
        >>> _as_f64_2d(np.array([[1, 2], [3, 4]])).dtype
        dtype('float64')
    """
    arr = _np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}")
    return _np.ascontiguousarray(arr, dtype=_np.float64)


def _as_f64_1d(x, n: int | None = None) -> _np.ndarray:
    """Coerce input data to contiguous float64 1D arrays.

    Args:
        x: Array-like input of shape (n_samples,).
        n: Optional expected length for validation.

    Returns:
        np.ndarray: Contiguous array with dtype float64 and shape (n_samples,).

    Examples:
        >>> import numpy as np
        >>> from clustor import _as_f64_1d
        >>> _as_f64_1d(np.array([1, 2, 3]), n=3).shape
        (3,)
    """
    arr = _np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape={arr.shape}")
    arr = _np.ascontiguousarray(arr, dtype=_np.float64)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"Expected length {n}, got {arr.shape[0]}")
    return arr


class KMeans(_KMeans):
    """KMeans clustering with KMeans++ initialization.

    Args:
        n_clusters: Number of clusters to form.
        n_init: Number of initializations to run.
        max_iter: Maximum iterations per initialization.
        tol: Convergence tolerance.
        metric: Distance metric ("euclidean" or "cosine").
        normalize_input: Optional input normalization for cosine distance.
        normalize_centers: Optional center normalization for cosine distance.
        random_state: Seed for reproducibility.
        verbose: Emit iteration diagnostics when True.

    Returns:
        KMeans: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.KMeans(n_clusters=1, n_init=1, random_state=0).fit(X)["labels"]
        array([0, 0])
    """

    def fit(self, X, *, sample_weight=None):  # noqa: N802
        Xc = _as_f64_2d(X)
        sw = None if sample_weight is None else _as_f64_1d(sample_weight, Xc.shape[0])
        return super().fit(Xc, sample_weight=sw)

    def fit_predict(self, X, *, sample_weight=None):  # noqa: N802
        Xc = _as_f64_2d(X)
        sw = None if sample_weight is None else _as_f64_1d(sample_weight, Xc.shape[0])
        return super().fit_predict(Xc, sample_weight=sw)

    def predict(self, X):  # noqa: N802
        return super().predict(_as_f64_2d(X))


class MiniBatchKMeans(_MiniBatchKMeans):
    """Mini-batch KMeans with streaming ``partial_fit`` support.

    Args:
        n_clusters: Number of clusters to form.
        batch_size: Batch size for updates.
        max_steps: Maximum number of update steps.
        metric: Distance metric ("euclidean" or "cosine").
        normalize_input: Optional input normalization for cosine distance.
        normalize_centers: Optional center normalization for cosine distance.
        random_state: Seed for reproducibility.
        verbose: Emit iteration diagnostics when True.

    Returns:
        MiniBatchKMeans: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> mb = clustor.MiniBatchKMeans(n_clusters=1, batch_size=2, random_state=0)
        >>> mb.fit(X)["labels"]
        array([0, 0])
    """

    def fit(self, X):  # noqa: N802
        return super().fit(_as_f64_2d(X))

    def partial_fit(self, X):  # noqa: N802
        return super().partial_fit(_as_f64_2d(X))

    def predict(self, X):  # noqa: N802
        return super().predict(_as_f64_2d(X))


class BisectingKMeans(_BisectingKMeans):
    """Bisecting KMeans (hierarchical top-down splitting).

    Args:
        n_clusters: Number of clusters to form.
        n_init: Number of initializations to run per split.
        max_iter: Maximum iterations per split.
        tol: Convergence tolerance.
        metric: Distance metric ("euclidean" or "cosine").
        normalize_input: Optional input normalization for cosine distance.
        normalize_centers: Optional center normalization for cosine distance.
        random_state: Seed for reproducibility.
        verbose: Emit iteration diagnostics when True.

    Returns:
        BisectingKMeans: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.BisectingKMeans(n_clusters=1, n_init=1, random_state=0).fit(X)["labels"]
        array([0, 0])
    """

    def fit(self, X, *, sample_weight=None):  # noqa: N802
        Xc = _as_f64_2d(X)
        sw = None if sample_weight is None else _as_f64_1d(sample_weight, Xc.shape[0])
        return super().fit(Xc, sample_weight=sw)

    def fit_predict(self, X, *, sample_weight=None):  # noqa: N802
        Xc = _as_f64_2d(X)
        sw = None if sample_weight is None else _as_f64_1d(sample_weight, Xc.shape[0])
        return super().fit_predict(Xc, sample_weight=sw)

    def predict(self, X):  # noqa: N802
        return super().predict(_as_f64_2d(X))


class DBSCAN(_DBSCAN):
    """Density-based spatial clustering of applications with noise (DBSCAN).

    Args:
        eps: Maximum distance between two samples for neighborhood inclusion.
        min_samples: Minimum samples required to form a core point.
        metric: Distance metric ("euclidean" or "cosine").
        normalize_input: Optional input normalization for cosine distance.
        verbose: Emit diagnostics when True.

    Returns:
        DBSCAN: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        >>> clustor.DBSCAN(eps=0.5, min_samples=1).fit(X)["labels"]
        array([0, 1])
    """

    def fit(self, X):  # noqa: N802
        return super().fit(_as_f64_2d(X))

    def fit_predict(self, X):  # noqa: N802
        return super().fit_predict(_as_f64_2d(X))


class GaussianMixture(_GaussianMixture):
    """Diagonal-covariance Gaussian Mixture Model with EM.

    Args:
        n_components: Number of mixture components.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        reg_covar: Non-negative regularization for diagonal covariance.
        init: Initialization strategy ("kmeans++" or "random").
        random_state: Seed for reproducibility.
        verbose: Emit iteration diagnostics when True.

    Returns:
        GaussianMixture: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> gm = clustor.GaussianMixture(1, max_iter=10, random_state=0)
        >>> gm.fit(X)["weights"].shape
        (1,)
    """

    def fit(self, X):  # noqa: N802
        return super().fit(_as_f64_2d(X))

    def predict(self, X):  # noqa: N802
        return super().predict(_as_f64_2d(X))

    def predict_proba(self, X):  # noqa: N802
        return super().predict_proba(_as_f64_2d(X))

    def score_samples(self, X):  # noqa: N802
        return super().score_samples(_as_f64_2d(X))

    def score(self, X):  # noqa: N802
        return super().score(_as_f64_2d(X))

    def aic(self, X):  # noqa: N802
        return super().aic(_as_f64_2d(X))

    def bic(self, X):  # noqa: N802
        return super().bic(_as_f64_2d(X))

    def sample(self, n_samples: int):
        return super().sample(int(n_samples))


class OPTICS(_OPTICS):
    """OPTICS density-based cluster ordering (reachability + core distances).

    Args:
        min_samples: Minimum samples required to form a core point.
        max_eps: Maximum epsilon neighborhood radius; defaults to no upper bound.
        metric: Distance metric ("euclidean" or "cosine").
        normalize_input: Optional input normalization for cosine distance.

    Returns:
        OPTICS: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.OPTICS(min_samples=1, max_eps=2.0).fit(X)["ordering"].shape
        (2,)
    """

    def fit(self, X):  # noqa: N802
        return super().fit(_as_f64_2d(X))

    def fit_predict(self, X):  # noqa: N802
        return super().fit_predict(_as_f64_2d(X))


class AffinityPropagation(_AffinityPropagation):
    """Affinity Propagation clustering (message passing).

    Args:
        damping: Damping factor in [0.5, 1.0).
        max_iter: Maximum number of iterations.
        convergence_iter: Iterations with no change for convergence.
        preference: Optional preference for exemplar selection.
        metric: Distance metric ("euclidean" or "cosine").
        normalize_input: Optional input normalization for cosine distance.
        verbose: Emit iteration diagnostics when True.

    Returns:
        AffinityPropagation: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.AffinityPropagation(damping=0.6, max_iter=50).fit(X)["labels"].shape
        (2,)
    """

    def fit(self, X):  # noqa: N802
        return super().fit(_as_f64_2d(X))

    def fit_predict(self, X):  # noqa: N802
        return super().fit_predict(_as_f64_2d(X))


class Birch(_Birch):
    """BIRCH streaming clustering using a CF-tree.

    Args:
        threshold: Radius threshold for subcluster creation.
        branching_factor: Maximum number of children per node.
        n_clusters: Optional number of final clusters.

    Returns:
        Birch: A fitted estimator when calling :meth:`fit`.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.Birch(threshold=1.0, n_clusters=1).fit(X)["labels"]
        array([0, 0])
    """

    def fit(self, X):  # noqa: N802
        return super().fit(_as_f64_2d(X))

    def fit_predict(self, X):  # noqa: N802
        return super().fit_predict(_as_f64_2d(X))


def kmeans(X, k: int, **kwargs):
    """Run KMeans via the functional API.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        k: Number of clusters to form.
        **kwargs: Keyword arguments forwarded to ``KMeans``.

    Returns:
        tuple: (centers, labels, inertia, n_iter).

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> centers, labels, inertia, n_iter = clustor.kmeans(X, 1, n_init=1, random_state=0)
        >>> labels.shape
        (2,)
    """
    Xc = _as_f64_2d(X)
    if "sample_weight" in kwargs and kwargs["sample_weight"] is not None:
        kwargs["sample_weight"] = _as_f64_1d(kwargs["sample_weight"], Xc.shape[0])
    return _kmeans(Xc, int(k), **kwargs)


def dbscan(X, **kwargs):
    """Run DBSCAN via the functional API.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        **kwargs: Keyword arguments forwarded to ``DBSCAN``.

    Returns:
        tuple: (labels, core_sample_indices, n_clusters).

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> labels, core_idx, n_clusters = clustor.dbscan(X, eps=0.5, min_samples=1)
        >>> n_clusters
        2
    """
    return _dbscan(_as_f64_2d(X), **kwargs)


def gaussian_mixture(X, n_components: int, **kwargs):
    """Run Gaussian Mixture Model via the functional API.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        n_components: Number of mixture components.
        **kwargs: Keyword arguments forwarded to ``GaussianMixture``.

    Returns:
        tuple: (weights, means, covars, responsibilities, converged, lower_bound, n_iter).

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> weights, means, covars, resp, converged, lower, n_iter = clustor.gaussian_mixture(
        ...     X, 1, max_iter=10, random_state=0
        ... )
        >>> weights.shape
        (1,)
    """
    return _gaussian_mixture(_as_f64_2d(X), int(n_components), **kwargs)


def optics(X, **kwargs):
    """Run OPTICS via the functional API.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        **kwargs: Keyword arguments forwarded to ``OPTICS``.

    Returns:
        dict: Dictionary containing ordering, reachability, core_distances, predecessor.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> result = clustor.optics(X, min_samples=1, max_eps=2.0)
        >>> sorted(result.keys())
        ['core_distances', 'ordering', 'predecessor', 'reachability']
    """
    return _optics(_as_f64_2d(X), **kwargs)


def affinity_propagation(X, **kwargs):
    """Run Affinity Propagation via the functional API.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        **kwargs: Keyword arguments forwarded to ``AffinityPropagation``.

    Returns:
        dict: Dictionary containing exemplars, labels, converged, n_iter.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> result = clustor.affinity_propagation(X, damping=0.6, max_iter=50)
        >>> "labels" in result
        True
    """
    return _affinity_propagation(_as_f64_2d(X), **kwargs)


def birch(X, **kwargs):
    """Run BIRCH via the functional API.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        **kwargs: Keyword arguments forwarded to ``Birch``.

    Returns:
        dict: Dictionary containing centers, labels, n_subclusters.

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> result = clustor.birch(X, threshold=1.0, n_clusters=1)
        >>> result["centers"].shape
        (1, 2)
    """
    return _birch(_as_f64_2d(X), **kwargs)


def hac_dendrogram(X, **kwargs):
    """Run hierarchical agglomerative clustering linkage.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        **kwargs: Keyword arguments forwarded to ``hac_dendrogram``.

    Returns:
        np.ndarray: Linkage matrix with shape (n_samples - 1, 4).

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.hac_dendrogram(X, method="average", metric="euclidean").shape
        (1, 4)
    """
    return _hac_dendrogram(_as_f64_2d(X), **kwargs)


def silhouette_score(X, labels, metric: str = "euclidean") -> float:
    """Compute the mean Silhouette Coefficient for all samples.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        labels: Cluster labels for each sample.
        metric: Distance metric ("euclidean" or "cosine").

    Returns:
        float: Silhouette score in [-1, 1].

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        >>> clustor.silhouette_score(X, [0, 1])
        0.0
    """
    return float(
        _silhouette_score(_as_f64_2d(X), list(map(int, labels)), metric=metric)
    )


def calinski_harabasz_score(X, labels) -> float:
    """Compute the Calinski-Harabasz score.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        labels: Cluster labels for each sample.

    Returns:
        float: Calinski-Harabasz score (higher is better).

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float64)
        >>> clustor.calinski_harabasz_score(X, [0, 0, 1, 1])
        399.99999999999966
    """
    return float(_calinski_harabasz_score(_as_f64_2d(X), list(map(int, labels))))


def davies_bouldin_score(X, labels) -> float:
    """Compute the Davies-Bouldin score.

    Args:
        X: Array-like input of shape (n_samples, n_features).
        labels: Cluster labels for each sample.

    Returns:
        float: Davies-Bouldin score (lower is better).

    Examples:
        >>> import numpy as np
        >>> import clustor
        >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float64)
        >>> clustor.davies_bouldin_score(X, [0, 0, 1, 1])
        0.07071067811865478
    """
    return float(_davies_bouldin_score(_as_f64_2d(X), list(map(int, labels))))


__all__ = [
    "KMeans",
    "MiniBatchKMeans",
    "BisectingKMeans",
    "DBSCAN",
    "GaussianMixture",
    "OPTICS",
    "AffinityPropagation",
    "Birch",
    "kmeans",
    "dbscan",
    "gaussian_mixture",
    "optics",
    "affinity_propagation",
    "birch",
    "hac_dendrogram",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "__version__",
]

# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pytest

import clustor


def _make_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=(0.0, 0.0), scale=0.4, size=(n, 2))
    b = rng.normal(loc=(4.0, 4.0), scale=0.4, size=(n, 2))
    c = rng.normal(loc=(0.0, 5.0), scale=0.4, size=(n, 2))
    return np.vstack([a, b, c])


def test_optics_shapes():
    X = _make_blobs(80, seed=1)
    res = clustor.optics(X, min_samples=10, max_eps=1.5, metric="euclidean")
    assert res["ordering"].shape == (X.shape[0],)
    assert res["reachability"].shape == (X.shape[0],)
    assert res["core_distances"].shape == (X.shape[0],)
    assert res["predecessor"].shape == (X.shape[0],)
    # Some core distances should be finite
    assert np.isfinite(res["core_distances"]).sum() > 0


def test_affinity_propagation_basic():
    X = _make_blobs(60, seed=2)
    ap = clustor.AffinityPropagation(damping=0.7, max_iter=300, convergence_iter=10)
    res = ap.fit(X)
    labels = res["labels"]
    exemplars = res["exemplars"]
    assert labels.shape == (X.shape[0],)
    assert exemplars.ndim == 1
    assert exemplars.shape[0] >= 1
    # labels are contiguous [0, k-1]
    assert labels.min() == 0
    assert labels.max() == exemplars.shape[0] - 1


def test_birch_basic():
    X = _make_blobs(60, seed=3)
    res = clustor.birch(X, threshold=0.8, branching_factor=25, n_clusters=3)
    centers = res["centers"]
    labels = res["labels"]
    assert centers.shape == (3, X.shape[1])
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset(set(range(3)))


def test_hac_dendrogram_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    Z = clustor.hac_dendrogram(X, method="average", metric="euclidean")
    assert Z.shape == (X.shape[0] - 1, 4)
    # last merge should include all samples
    assert int(Z[-1, 3]) == X.shape[0]


def test_optics_invalid_params():
    with pytest.raises(ValueError):
        clustor.OPTICS(min_samples=0)
    with pytest.raises(ValueError):
        clustor.OPTICS(max_eps=-1.0)


def test_affinity_invalid_params():
    with pytest.raises(ValueError):
        clustor.AffinityPropagation(damping=0.4)


def test_birch_invalid_params():
    with pytest.raises(ValueError):
        clustor.Birch(threshold=0.0)
    with pytest.raises(ValueError):
        clustor.Birch(branching_factor=1)
    with pytest.raises(ValueError):
        clustor.Birch(n_clusters=0)


def test_hac_invalid_method():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        clustor.hac_dendrogram(X, method="invalid", metric="euclidean")

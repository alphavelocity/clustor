# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pytest

import clustor


def test_kmeans_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=1.0, size=(200, 2)),
            rng.normal(loc=6.0, scale=1.0, size=(200, 2)),
        ]
    )
    km = clustor.KMeans(n_clusters=2, n_init=5, max_iter=200, random_state=0)
    out = km.fit(X)
    centers = out["centers"]
    assert centers.shape == (2, 2)
    labels = km.predict(X)
    assert labels.shape == (400,)
    assert 50 < (labels == labels[0]).sum() < 350


def test_cosine_metric_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    km = clustor.KMeans(n_clusters=3, metric="cosine", n_init=3, random_state=1)
    out = km.fit(X)
    assert out["centers"].shape == (3, 5)


def test_cosine_metric_zero_vectors():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    km = clustor.KMeans(n_clusters=2, metric="cosine", n_init=2, random_state=0)
    out = km.fit(X)
    assert np.isfinite(out["centers"]).all()
    labels = km.predict(X)
    assert labels.shape == (3,)


def test_minibatch_fit():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 3))
    mb = clustor.MiniBatchKMeans(
        n_clusters=4, batch_size=64, max_steps=200, random_state=2
    )
    out = mb.fit(X)
    assert out["centers"].shape == (4, 3)
    labels = mb.predict(X)
    assert labels.shape == (500,)


def test_kmeans_sample_weight_pulls_center():
    # With k=1, center should be weighted mean.
    X = np.array([[0.0], [10.0]], dtype=np.float64)
    w = np.array([1.0, 9.0], dtype=np.float64)
    km = clustor.KMeans(n_clusters=1, n_init=1, max_iter=50, random_state=0)
    out = km.fit(X, sample_weight=w)
    c = out["centers"].ravel()[0]
    # weighted mean = (0*1 + 10*9) / 10 = 9
    assert abs(c - 9.0) < 1e-6


def test_bisecting_kmeans_three_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(loc=-5.0, scale=0.8, size=(150, 2)),
            rng.normal(loc=0.0, scale=0.8, size=(150, 2)),
            rng.normal(loc=5.0, scale=0.8, size=(150, 2)),
        ]
    )
    bk = clustor.BisectingKMeans(n_clusters=3, n_init=5, max_iter=200, random_state=0)
    out = bk.fit(X)
    centers = out["centers"]
    assert centers.shape == (3, 2)
    labels = bk.predict(X)
    assert labels.shape == (450,)
    assert len(np.unique(labels)) == 3


def test_kmeans_invalid_params():
    with pytest.raises(ValueError):
        clustor.KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        clustor.BisectingKMeans(n_clusters=0)
    with pytest.raises(ValueError):
        clustor.BisectingKMeans(n_clusters=2, n_init=0)
    X = np.array([[0.0], [1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        clustor.kmeans(X, 0)
    with pytest.raises(ValueError):
        clustor.kmeans(X, 1, n_init=0)
    with pytest.raises(ValueError):
        clustor._clustor.bisecting_kmeans(X, 0)
    with pytest.raises(ValueError):
        clustor._clustor.bisecting_kmeans(X, 1, n_init=0)


def test_kmeans_sample_weight_mismatch():
    X = np.array([[0.0], [1.0]], dtype=np.float64)
    km = clustor.KMeans(n_clusters=1, n_init=1, max_iter=10, random_state=0)
    with pytest.raises(ValueError):
        km.fit(X, sample_weight=np.array([1.0], dtype=np.float64))


def test_minibatch_invalid_params():
    with pytest.raises(ValueError):
        clustor.MiniBatchKMeans(n_clusters=0)
    with pytest.raises(ValueError):
        clustor.MiniBatchKMeans(n_clusters=2, batch_size=0)


def test_minibatch_sample_weight_rejected():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 2))
    mb = clustor.MiniBatchKMeans(
        n_clusters=2, batch_size=4, max_steps=5, random_state=3
    )
    with pytest.raises(ValueError):
        mb.fit(X, sample_weight=np.ones(10, dtype=np.float64))

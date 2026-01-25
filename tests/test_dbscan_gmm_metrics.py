# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pytest

import clustor


def test_dbscan_simple():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )
    db = clustor.DBSCAN(eps=0.25, min_samples=2)
    out = db.fit(X)
    labels = out["labels"]
    assert labels.shape == (7,)
    assert out["n_clusters"] == 2
    assert (labels == -1).sum() == 1


def test_gmm_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(loc=-2.0, scale=0.5, size=(200, 2)),
            rng.normal(loc=2.0, scale=0.5, size=(200, 2)),
        ]
    )
    gm = clustor.GaussianMixture(2, max_iter=50, tol=1e-4, random_state=0)
    out = gm.fit(X)
    assert out["means"].shape == (2, 2)
    labels = gm.predict(X)
    assert labels.shape == (400,)
    assert 50 < (labels == 0).sum() < 350


def test_validation_metrics():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.4, size=(100, 2)),
            rng.normal(loc=4.0, scale=0.4, size=(100, 2)),
        ]
    )
    km = clustor.KMeans(n_clusters=2, n_init=3, random_state=1)
    out = km.fit(X)
    labels = out["labels"]
    s = clustor.silhouette_score(X, labels)
    assert -1.0 <= s <= 1.0
    ch = clustor.calinski_harabasz_score(X, labels)
    assert ch > 0
    db = clustor.davies_bouldin_score(X, labels)
    assert db > 0


def test_silhouette_singletons_and_noise():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
    labels = np.array([0, 1, 2], dtype=np.int64)
    s = clustor.silhouette_score(X, labels)
    assert np.isclose(s, 0.0)

    with pytest.raises(ValueError):
        clustor.silhouette_score(X, [-1, -1, -1])


def test_gmm_aic_bic_finite():
    rng = np.random.default_rng(2)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.6, size=(150, 2)),
            rng.normal(loc=1.0, scale=0.6, size=(150, 2)),
        ]
    )
    gm = clustor.GaussianMixture(2, max_iter=60, tol=1e-4, random_state=0)
    gm.fit(X)
    aic = gm.aic(X)
    bic = gm.bic(X)
    score = gm.score(X)
    assert np.isfinite(aic)
    assert np.isfinite(bic)
    assert np.isfinite(score)


def test_dbscan_invalid_params():
    with pytest.raises(ValueError):
        clustor.DBSCAN(eps=0.0)
    with pytest.raises(ValueError):
        clustor.DBSCAN(eps=0.1, min_samples=0)


def test_gmm_invalid_params():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        clustor.GaussianMixture(2, tol=-1.0).fit(X)
    with pytest.raises(ValueError):
        clustor.GaussianMixture(2, reg_covar=-1.0).fit(X)


def test_gmm_sample_weight_validation():
    X = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    gm = clustor.GaussianMixture(2, max_iter=10, tol=1e-6, random_state=0)
    with pytest.raises(ValueError):
        gm.fit(X, sample_weight=np.array([1.0]))
    with pytest.raises(ValueError):
        gm.fit(X, sample_weight=np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        gm.fit(X, sample_weight=np.array([0.0, 0.0]))


def test_gmm_sample_weight_bias():
    X = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    weights = np.array([10.0, 1.0], dtype=np.float64)
    gm = clustor.GaussianMixture(
        2, max_iter=20, tol=1e-6, init="kmeans++", random_state=0
    )
    out = gm.fit(X, sample_weight=weights)
    assert np.isclose(out["weights"].sum(), 1.0)
    assert out["weights"].max() > 0.8

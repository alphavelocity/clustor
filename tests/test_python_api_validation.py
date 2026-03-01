# Copyright (c) 2026 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
import pytest

import clustor

NON_2D_RAW_INPUTS = [
    np.array(1.0, dtype=np.float64),
    np.array([0.0, 1.0, 2.0], dtype=np.float64),
    np.array([[[0.0], [1.0], [2.0]]], dtype=np.float64),
]
NON_2D_INPUTS = [
    pytest.param(NON_2D_RAW_INPUTS[0], id="0d"),
    pytest.param(NON_2D_RAW_INPUTS[1], id="1d"),
    pytest.param(NON_2D_RAW_INPUTS[2], id="3d"),
]

SIMPLE_2D = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
SCORING_DATA = np.array(
    [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]], dtype=np.float64
)
TWO_SAMPLE_X = np.array([[0.0], [1.0]], dtype=np.float64)
BAD_SW_2D = np.array([[1.0], [1.0]], dtype=np.float64)
BAD_SW_LEN = np.array([1.0], dtype=np.float64)

ESTIMATOR_FACTORIES = [
    pytest.param(
        lambda: clustor.KMeans(n_clusters=1, n_init=1, random_state=0),
        id="kmeans",
    ),
    pytest.param(
        lambda: clustor.MiniBatchKMeans(n_clusters=1, batch_size=2, random_state=0),
        id="minibatch_kmeans",
    ),
    pytest.param(
        lambda: clustor.BisectingKMeans(n_clusters=1, n_init=1, random_state=0),
        id="bisecting_kmeans",
    ),
    pytest.param(lambda: clustor.DBSCAN(eps=0.5, min_samples=1), id="dbscan"),
    pytest.param(
        lambda: clustor.GaussianMixture(1, max_iter=5, random_state=0),
        id="gaussian_mixture",
    ),
    pytest.param(lambda: clustor.OPTICS(min_samples=1, max_eps=2.0), id="optics"),
    pytest.param(
        lambda: clustor.AffinityPropagation(damping=0.6, max_iter=10),
        id="affinity_propagation",
    ),
    pytest.param(lambda: clustor.Birch(threshold=1.0, n_clusters=1), id="birch"),
]

FUNCTIONAL_APIS = [
    pytest.param(
        lambda x: clustor.kmeans(x, 1, n_init=1, random_state=0),
        id="kmeans",
    ),
    pytest.param(lambda x: clustor.dbscan(x, eps=0.5, min_samples=1), id="dbscan"),
    pytest.param(
        lambda x: clustor.gaussian_mixture(x, 1, max_iter=5, random_state=0),
        id="gaussian_mixture",
    ),
    pytest.param(lambda x: clustor.optics(x, min_samples=1, max_eps=2.0), id="optics"),
    pytest.param(
        lambda x: clustor.affinity_propagation(x, damping=0.6, max_iter=10),
        id="affinity_propagation",
    ),
    pytest.param(
        lambda x: clustor.birch(x, threshold=1.0, n_clusters=1),
        id="birch",
    ),
    pytest.param(
        lambda x: clustor.hac_dendrogram(x, method="average", metric="euclidean"),
        id="hac_dendrogram",
    ),
]


@pytest.mark.parametrize("bad_input", NON_2D_INPUTS)
@pytest.mark.parametrize("estimator_factory", ESTIMATOR_FACTORIES)
def test_estimators_reject_non_2d_inputs(estimator_factory, bad_input):
    with pytest.raises(ValueError, match="2D"):
        estimator_factory().fit(bad_input)


@pytest.mark.parametrize("bad_input", NON_2D_INPUTS)
@pytest.mark.parametrize("func", FUNCTIONAL_APIS)
def test_functional_apis_reject_non_2d_inputs(func, bad_input):
    with pytest.raises(ValueError, match="2D"):
        func(bad_input)


def test_predict_methods_reject_non_2d_inputs():
    models = [
        clustor.KMeans(n_clusters=1, n_init=1, random_state=0),
        clustor.MiniBatchKMeans(n_clusters=1, batch_size=2, random_state=0),
        clustor.BisectingKMeans(n_clusters=1, n_init=1, random_state=0),
    ]
    for model in models:
        model.fit(SIMPLE_2D)
        for bad_input in NON_2D_RAW_INPUTS:
            with pytest.raises(ValueError, match="2D"):
                model.predict(bad_input)


def test_kmeans_fit_accepts_integer_input_and_coerces_dtype():
    X = np.array([[0, 0], [2, 2], [10, 10], [12, 12]], dtype=np.int32)
    out = clustor.KMeans(n_clusters=2, n_init=3, random_state=0).fit(X)

    assert out["centers"].dtype == np.float64
    assert out["labels"].shape == (4,)


def test_kmeans_fit_accepts_non_contiguous_input():
    X_base = np.array(
        [[0.0, 100.0], [1.0, 101.0], [10.0, 110.0], [11.0, 111.0]],
        dtype=np.float64,
    )
    X = X_base[:, ::2]
    assert not X.flags.c_contiguous

    out = clustor.KMeans(n_clusters=2, n_init=2, random_state=0).fit(X)
    assert out["centers"].shape == (2, 1)


def test_kmeans_fit_rejects_non_1d_sample_weight():
    km = clustor.KMeans(n_clusters=1, n_init=1, random_state=0)
    with pytest.raises(ValueError, match="1D"):
        km.fit(TWO_SAMPLE_X, sample_weight=BAD_SW_2D)


def test_gaussian_mixture_fit_rejects_non_1d_sample_weight():
    gm = clustor.GaussianMixture(1, max_iter=10, random_state=0)
    with pytest.raises(ValueError, match="1D"):
        gm.fit(TWO_SAMPLE_X, sample_weight=BAD_SW_2D)


def test_kmeans_functional_api_rejects_non_1d_sample_weight():
    with pytest.raises(ValueError, match="1D"):
        clustor.kmeans(
            TWO_SAMPLE_X,
            1,
            sample_weight=BAD_SW_2D,
            n_init=1,
            random_state=0,
        )


def test_gaussian_mixture_functional_api_rejects_non_1d_sample_weight():
    with pytest.raises(ValueError, match="1D"):
        clustor.gaussian_mixture(
            TWO_SAMPLE_X,
            1,
            sample_weight=BAD_SW_2D,
        )


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (clustor.kmeans, {"k": 1, "n_init": 1, "random_state": 0}),
        (clustor.gaussian_mixture, {"n_components": 1}),
    ],
)
def test_functional_apis_reject_sample_weight_length_mismatch(fn, kwargs):
    with pytest.raises(ValueError, match="length"):
        fn(TWO_SAMPLE_X, sample_weight=BAD_SW_LEN, **kwargs)


@pytest.mark.parametrize(
    ("score_fn", "kwargs"),
    [
        (clustor.silhouette_score, {"metric": "euclidean"}),
        (clustor.calinski_harabasz_score, {}),
        (clustor.davies_bouldin_score, {}),
    ],
)
def test_metric_functions_reject_label_length_mismatch(score_fn, kwargs):
    with pytest.raises(ValueError, match="labels"):
        score_fn(SCORING_DATA, [0, 0, 1], **kwargs)


def test_metric_functions_accept_integer_labels():
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    assert np.isfinite(clustor.silhouette_score(SCORING_DATA, labels))
    assert np.isfinite(clustor.calinski_harabasz_score(SCORING_DATA, labels))
    assert np.isfinite(clustor.davies_bouldin_score(SCORING_DATA, labels))

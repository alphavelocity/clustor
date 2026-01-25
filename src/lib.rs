// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Clustor: fast clustering in Rust + Python (PyO3).

mod affinity;
mod birch;
mod bisecting;
mod dbscan;
mod errors;
mod gmm;
mod hac;
mod kmeans;
mod metrics;
mod optics;
mod utils;
mod validation;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModuleMethods};

use crate::affinity::{AffinityParams, fit_affinity_propagation};
use crate::birch::{BirchParams, fit_birch};
use crate::bisecting::{BisectingParams, fit_bisecting_kmeans};
use crate::dbscan::{DbscanParams, fit_dbscan};
use crate::errors::ClustorError;
use crate::gmm::{
    GmmParams, fit_gmm_diag, gmm_log_likelihoods_diag, gmm_log_resp_diag, sample_gmm_diag,
};
use crate::hac::{Linkage, hac_linkage};
use crate::kmeans::{
    KMeansParams, MiniBatchParams, MiniBatchState, fit_kmeans, minibatch_fit, minibatch_partial_fit,
};
use crate::metrics::Metric;
use crate::optics::{OpticsParams, fit_optics};
use crate::validation::{
    calinski_harabasz_score as ch_score, davies_bouldin_score as db_score,
    silhouette_score as sil_score,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn map_err(e: ClustorError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn as_metric(metric: &str) -> Result<Metric, ClustorError> {
    Metric::parse(metric).ok_or_else(|| {
        ClustorError::InvalidArg(format!(
            "unknown metric: {metric} (expected 'euclidean' or 'cosine')"
        ))
    })
}

fn ensure_i64_bounds(value: usize, context: &str) -> PyResult<()> {
    if i64::try_from(value).is_err() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{context} exceeds i64::MAX"
        )));
    }
    Ok(())
}

fn usize_iter_to_i64<I>(values: I, context: &str) -> PyResult<Vec<i64>>
where
    I: IntoIterator<Item = usize>,
{
    let iter = values.into_iter();
    let (lower, _) = iter.size_hint();
    let mut out = Vec::with_capacity(lower);
    for value in iter {
        let value_i64 = i64::try_from(value).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{context} value {value} exceeds i64::MAX"
            ))
        })?;
        out.push(value_i64);
    }
    Ok(out)
}

fn vec_to_pyarray2<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), data).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to build array: {e}"))
    })?;
    Ok(PyArray2::from_owned_array(py, array))
}

type PyKMeansResult<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    f64,
    u32,
);
type PyDbscanResult<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, usize);
type PyGaussianMixtureResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    bool,
    f64,
    u32,
);

fn to_py_fit_result<'py>(
    py: Python<'py>,
    centers: Vec<f64>,
    labels: Vec<usize>,
    inertia: f64,
    n_iter: u32,
    n_clusters: usize,
    n_features: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let centers_arr = vec_to_pyarray2(py, n_clusters, n_features, centers)?;
    let labels_i64 = usize_iter_to_i64(labels, "labels")?;
    let labels_arr = PyArray1::from_vec(py, labels_i64);
    d.set_item("centers", centers_arr)?;
    d.set_item("labels", labels_arr)?;
    d.set_item("inertia", inertia)?;
    d.set_item("n_iter", n_iter)?;
    Ok(d)
}

// ---------------- KMeans ----------------

#[pyclass(subclass, module = "clustor._clustor")]
struct KMeans {
    n_clusters: usize,
    n_init: usize,
    max_iter: u32,
    tol: f64,
    metric: String,
    normalize_input: Option<bool>,
    normalize_centers: Option<bool>,
    random_state: Option<u64>,
    verbose: bool,

    centers_: Option<Vec<f64>>,
    n_features_: Option<usize>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters, *, n_init=10, max_iter=300, tol=1e-4, metric="euclidean", normalize_input=None, normalize_centers=None, random_state=None, verbose=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        n_init: usize,
        max_iter: u32,
        tol: f64,
        metric: &str,
        normalize_input: Option<bool>,
        normalize_centers: Option<bool>,
        random_state: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        if n_clusters == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_clusters must be > 0",
            ));
        }
        if n_init == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_init must be > 0",
            ));
        }
        Ok(Self {
            n_clusters,
            n_init,
            max_iter,
            tol,
            metric: metric.to_string(),
            normalize_input,
            normalize_centers,
            random_state,
            verbose,
            centers_: None,
            n_features_: None,
        })
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data: Vec<f64> = x.iter().copied().collect();
        let sw: Option<Vec<f64>> = sample_weight.map(|sw| sw.as_array().to_vec());
        if let Some(ref w) = sw
            && w.len() != n_samples
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sample_weight length mismatch",
            ));
        }

        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);
        let normalize_centers = self.normalize_centers.unwrap_or(metric == Metric::Cosine);

        let params = KMeansParams {
            n_clusters: self.n_clusters,
            n_init: self.n_init,
            max_iter: self.max_iter,
            tol: self.tol,
            metric,
            normalize_input,
            normalize_centers,
            random_state: self.random_state,
            verbose: self.verbose,
        };

        let out = py
            .detach(|| fit_kmeans(&data, n_samples, n_features, sw.as_deref(), &params))
            .map_err(map_err)?;
        self.centers_ = Some(out.centers.clone());
        self.n_features_ = Some(n_features);

        to_py_fit_result(
            py,
            out.centers,
            out.labels,
            out.inertia,
            out.n_iter,
            self.n_clusters,
            n_features,
        )
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = self.fit(py, x, sample_weight)?;
        let labels = d.get_item("labels")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("labels missing from result")
        })?;
        Ok(labels.cast_into()?)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        ensure_i64_bounds(self.n_clusters, "n_clusters")?;
        let centers = self.centers_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let n_features = self.n_features_.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;

        let x = x.as_array();
        let (n_samples, nf) = x.dim();
        if nf != n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature dimension mismatch",
            ));
        }
        let data: Vec<f64> = x.iter().copied().collect();

        let metric = as_metric(&self.metric).map_err(map_err)?;

        let labels = py.detach(|| {
            let mut labels = vec![0i64; n_samples];
            let data_norms = if metric == Metric::Cosine {
                Some(crate::utils::compute_row_norms(
                    &data, n_samples, n_features,
                ))
            } else {
                None
            };
            let center_norms = if metric == Metric::Cosine {
                let mut cn = Vec::with_capacity(self.n_clusters);
                for c in 0..self.n_clusters {
                    cn.push(crate::metrics::l2_norm(
                        &centers[c * n_features..(c + 1) * n_features],
                    ));
                }
                Some(cn)
            } else {
                None
            };

            for i in 0..n_samples {
                let xrow = &data[i * n_features..(i + 1) * n_features];
                let mut best_k = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..self.n_clusters {
                    let cent = &centers[c * n_features..(c + 1) * n_features];
                    let d = match metric {
                        Metric::Euclidean => crate::metrics::euclidean_sq(xrow, cent),
                        Metric::Cosine => {
                            let xn = data_norms.as_ref().unwrap()[i];
                            let cn = center_norms.as_ref().unwrap()[c];
                            let cd = crate::metrics::cosine_distance(xrow, xn, cent, cn);
                            cd * cd
                        }
                    };
                    if d < best_d {
                        best_d = d;
                        best_k = c;
                    }
                }
                labels[i] = best_k as i64;
            }
            labels
        });

        Ok(PyArray1::from_vec(py, labels))
    }

    #[getter]
    fn centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let centers = self.centers_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let nf = self.n_features_.unwrap();
        vec_to_pyarray2(py, self.n_clusters, nf, centers.clone())
    }
}

// ---------------- MiniBatchKMeans ----------------

#[pyclass(subclass, module = "clustor._clustor")]
struct MiniBatchKMeans {
    n_clusters: usize,
    batch_size: usize,
    max_steps: u32,
    metric: String,
    normalize_input: Option<bool>,
    normalize_centers: Option<bool>,
    random_state: Option<u64>,
    verbose: bool,

    state: Option<MiniBatchState>,
}

#[pymethods]
impl MiniBatchKMeans {
    #[new]
    #[pyo3(signature = (n_clusters, *, batch_size=256, max_steps=10_000, metric="euclidean", normalize_input=None, normalize_centers=None, random_state=None, verbose=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        batch_size: usize,
        max_steps: u32,
        metric: &str,
        normalize_input: Option<bool>,
        normalize_centers: Option<bool>,
        random_state: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        if n_clusters == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_clusters must be > 0",
            ));
        }
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be > 0",
            ));
        }
        Ok(Self {
            n_clusters,
            batch_size,
            max_steps,
            metric: metric.to_string(),
            normalize_input,
            normalize_centers,
            random_state,
            verbose,
            state: None,
        })
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _ = sample_weight;
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data: Vec<f64> = x.iter().copied().collect();

        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);
        let normalize_centers = self.normalize_centers.unwrap_or(metric == Metric::Cosine);

        let params = MiniBatchParams {
            n_clusters: self.n_clusters,
            batch_size: self.batch_size,
            max_steps: self.max_steps,
            metric,
            normalize_input,
            normalize_centers,
            random_state: self.random_state,
            verbose: self.verbose,
        };

        let (state, out) = py
            .detach(|| minibatch_fit(&data, n_samples, n_features, &params))
            .map_err(map_err)?;
        self.state = Some(state);
        to_py_fit_result(
            py,
            out.centers,
            out.labels,
            out.inertia,
            out.n_iter,
            self.n_clusters,
            n_features,
        )
    }

    fn partial_fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data: Vec<f64> = x.iter().copied().collect();

        let state = self.state.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not initialized; call fit() first")
        })?;
        py.detach(|| minibatch_partial_fit(state, &data, n_samples, n_features, self.verbose))
            .map_err(map_err)?;

        let d = PyDict::new(py);
        let centers_arr = vec_to_pyarray2(py, self.n_clusters, n_features, state.centers.clone())?;
        d.set_item("centers", centers_arr)?;
        d.set_item("n_steps", 1)?;
        Ok(d)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        ensure_i64_bounds(self.n_clusters, "n_clusters")?;
        let state = self.state.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let centers = &state.centers;
        let n_features = state.n_features;

        let x = x.as_array();
        let (n_samples, nf) = x.dim();
        if nf != n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature dimension mismatch",
            ));
        }
        let data: Vec<f64> = x.iter().copied().collect();
        let metric = as_metric(&self.metric).map_err(map_err)?;

        let labels = py.detach(|| {
            let mut labels = vec![0i64; n_samples];
            let data_norms = if metric == Metric::Cosine {
                Some(crate::utils::compute_row_norms(
                    &data, n_samples, n_features,
                ))
            } else {
                None
            };
            let center_norms = if metric == Metric::Cosine {
                let mut cn = Vec::with_capacity(self.n_clusters);
                for c in 0..self.n_clusters {
                    cn.push(crate::metrics::l2_norm(
                        &centers[c * n_features..(c + 1) * n_features],
                    ));
                }
                Some(cn)
            } else {
                None
            };
            for i in 0..n_samples {
                let xrow = &data[i * n_features..(i + 1) * n_features];
                let mut best_k = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..self.n_clusters {
                    let cent = &centers[c * n_features..(c + 1) * n_features];
                    let d = match metric {
                        Metric::Euclidean => crate::metrics::euclidean_sq(xrow, cent),
                        Metric::Cosine => {
                            let xn = data_norms.as_ref().unwrap()[i];
                            let cn = center_norms.as_ref().unwrap()[c];
                            let cd = crate::metrics::cosine_distance(xrow, xn, cent, cn);
                            cd * cd
                        }
                    };
                    if d < best_d {
                        best_d = d;
                        best_k = c;
                    }
                }
                labels[i] = best_k as i64;
            }
            labels
        });
        Ok(PyArray1::from_vec(py, labels))
    }

    #[getter]
    fn centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let state = self.state.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        vec_to_pyarray2(py, self.n_clusters, state.n_features, state.centers.clone())
    }
}

// ---------------- BisectingKMeans ----------------

#[pyclass(subclass, module = "clustor._clustor")]
struct BisectingKMeans {
    n_clusters: usize,
    n_init: usize,
    max_iter: u32,
    tol: f64,
    metric: String,
    normalize_input: Option<bool>,
    normalize_centers: Option<bool>,
    random_state: Option<u64>,
    verbose: bool,

    centers_: Option<Vec<f64>>,
    n_features_: Option<usize>,
}

#[pymethods]
impl BisectingKMeans {
    #[new]
    #[pyo3(signature = (n_clusters, *, n_init=10, max_iter=300, tol=1e-4, metric="euclidean", normalize_input=None, normalize_centers=None, random_state=None, verbose=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_clusters: usize,
        n_init: usize,
        max_iter: u32,
        tol: f64,
        metric: &str,
        normalize_input: Option<bool>,
        normalize_centers: Option<bool>,
        random_state: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        if n_clusters < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_clusters must be >= 1",
            ));
        }
        Ok(Self {
            n_clusters,
            n_init,
            max_iter,
            tol,
            metric: metric.to_string(),
            normalize_input,
            normalize_centers,
            random_state,
            verbose,
            centers_: None,
            n_features_: None,
        })
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data: Vec<f64> = x.iter().copied().collect();
        let sw: Option<Vec<f64>> = sample_weight.map(|sw| sw.as_array().to_vec());
        if let Some(ref w) = sw
            && w.len() != n_samples
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sample_weight length mismatch",
            ));
        }

        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);
        let normalize_centers = self.normalize_centers.unwrap_or(metric == Metric::Cosine);

        let params = BisectingParams {
            n_clusters: self.n_clusters,
            n_init: self.n_init,
            max_iter: self.max_iter,
            tol: self.tol,
            metric,
            normalize_input,
            normalize_centers,
            random_state: self.random_state,
            verbose: self.verbose,
        };

        let out = py
            .detach(|| fit_bisecting_kmeans(&data, n_samples, n_features, sw.as_deref(), &params))
            .map_err(map_err)?;
        self.centers_ = Some(out.centers.clone());
        self.n_features_ = Some(n_features);

        // Return a dict similar to KMeans
        let dct = PyDict::new(py);
        let centers_arr = vec_to_pyarray2(py, self.n_clusters, n_features, out.centers)?;
        let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
        dct.set_item("centers", centers_arr)?;
        dct.set_item("labels", PyArray1::from_vec(py, labels_i64))?;
        dct.set_item("inertia", out.inertia)?;
        dct.set_item("n_splits", out.n_splits)?;
        Ok(dct)
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = self.fit(py, x, sample_weight)?;
        let labels = d.get_item("labels")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("labels missing from result")
        })?;
        Ok(labels.cast_into()?)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        ensure_i64_bounds(self.n_clusters, "n_clusters")?;
        let centers = self.centers_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let n_features = self.n_features_.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;

        let x = x.as_array();
        let (n_samples, nf) = x.dim();
        if nf != n_features {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature dimension mismatch",
            ));
        }
        let data: Vec<f64> = x.iter().copied().collect();

        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);
        let normalize_centers = self.normalize_centers.unwrap_or(metric == Metric::Cosine);

        let labels = py.detach(|| {
            let mut labels = vec![0i64; n_samples];
            let data_norms = if metric == Metric::Cosine && normalize_input {
                Some(crate::utils::compute_row_norms(
                    &data, n_samples, n_features,
                ))
            } else {
                None
            };
            let center_norms = if metric == Metric::Cosine && normalize_centers {
                Some(crate::utils::compute_row_norms(
                    centers,
                    self.n_clusters,
                    n_features,
                ))
            } else {
                None
            };

            for i in 0..n_samples {
                let xrow = &data[i * n_features..(i + 1) * n_features];
                let mut best_k = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..self.n_clusters {
                    let cent = &centers[c * n_features..(c + 1) * n_features];
                    let d = match metric {
                        Metric::Euclidean => crate::metrics::euclidean_sq(xrow, cent),
                        Metric::Cosine => {
                            let xn = data_norms.as_ref().unwrap()[i];
                            let cn = center_norms.as_ref().unwrap()[c];
                            let cd = crate::metrics::cosine_distance(xrow, xn, cent, cn);
                            cd * cd
                        }
                    };
                    if d < best_d {
                        best_d = d;
                        best_k = c;
                    }
                }
                labels[i] = best_k as i64;
            }
            labels
        });
        Ok(PyArray1::from_vec(py, labels))
    }

    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let centers = self
            .centers_
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let d = self.n_features_.unwrap();
        vec_to_pyarray2(py, self.n_clusters, d, centers.clone())
    }
}

#[allow(clippy::upper_case_acronyms)]
#[pyclass(subclass, module = "clustor._clustor")]
struct DBSCAN {
    eps: f64,
    min_samples: usize,
    metric: String,
    normalize_input: Option<bool>,
    verbose: bool,
    labels_: Option<Vec<i64>>,
    core_sample_indices_: Option<Vec<usize>>,
    n_clusters_: Option<usize>,
}

#[pymethods]
impl DBSCAN {
    #[new]
    #[pyo3(signature = (*, eps=0.5, min_samples=5, metric="euclidean", normalize_input=None, verbose=false))]
    fn new(
        eps: f64,
        min_samples: usize,
        metric: &str,
        normalize_input: Option<bool>,
        verbose: bool,
    ) -> PyResult<Self> {
        if eps <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("eps must be > 0"));
        }
        if min_samples == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_samples must be > 0",
            ));
        }
        Ok(Self {
            eps,
            min_samples,
            metric: metric.to_string(),
            normalize_input,
            verbose,
            labels_: None,
            core_sample_indices_: None,
            n_clusters_: None,
        })
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _ = sample_weight;
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data: Vec<f64> = x.iter().copied().collect();

        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);

        let params = DbscanParams {
            eps: self.eps,
            min_samples: self.min_samples,
            metric,
            normalize_input,
            verbose: self.verbose,
        };

        let out = py
            .detach(|| fit_dbscan(&data, n_samples, n_features, &params))
            .map_err(map_err)?;
        self.labels_ = Some(out.labels.clone());
        self.core_sample_indices_ = Some(out.core_sample_indices.clone());
        self.n_clusters_ = Some(out.n_clusters);

        let d = PyDict::new(py);
        d.set_item("labels", PyArray1::from_vec(py, out.labels))?;
        let core_sample_indices =
            usize_iter_to_i64(out.core_sample_indices, "core_sample_indices")?;
        d.set_item(
            "core_sample_indices",
            PyArray1::from_vec(py, core_sample_indices),
        )?;
        d.set_item("n_clusters", out.n_clusters)?;
        Ok(d)
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = self.fit(py, x, sample_weight)?;
        let labels = d.get_item("labels")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("labels missing from result")
        })?;
        Ok(labels.cast_into()?)
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = self.labels_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        Ok(PyArray1::from_vec(py, labels.clone()))
    }

    #[getter]
    fn core_sample_indices_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let idx = self.core_sample_indices_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let idx_i64 = usize_iter_to_i64(idx.iter().copied(), "core_sample_indices")?;
        Ok(PyArray1::from_vec(py, idx_i64))
    }

    #[getter]
    fn n_clusters_(&self) -> PyResult<usize> {
        self.n_clusters_.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })
    }
}

// ---------------- GaussianMixture (diag) ----------------

#[pyclass(subclass, module = "clustor._clustor")]
struct GaussianMixture {
    n_components: usize,
    max_iter: u32,
    tol: f64,
    reg_covar: f64,
    init: String,
    random_state: Option<u64>,
    verbose: bool,

    weights_: Option<Vec<f64>>,
    means_: Option<Vec<f64>>,
    covars_: Option<Vec<f64>>,
    n_features_: Option<usize>,
    converged_: Option<bool>,
    lower_bound_: Option<f64>,
    n_iter_: Option<u32>,
}

#[pymethods]
impl GaussianMixture {
    #[new]
    #[pyo3(signature = (n_components, *, max_iter=100, tol=1e-3, reg_covar=1e-6, init="kmeans++", random_state=None, verbose=false))]
    fn new(
        n_components: usize,
        max_iter: u32,
        tol: f64,
        reg_covar: f64,
        init: &str,
        random_state: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        if n_components == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_components must be > 0",
            ));
        }
        Ok(Self {
            n_components,
            max_iter,
            tol,
            reg_covar,
            init: init.to_string(),
            random_state,
            verbose,
            weights_: None,
            means_: None,
            covars_: None,
            n_features_: None,
            converged_: None,
            lower_bound_: None,
            n_iter_: None,
        })
    }

    #[pyo3(signature = (x, *, sample_weight=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data: Vec<f64> = x.iter().copied().collect();
        let sample_weight_vec =
            sample_weight.map(|weights| weights.as_array().iter().copied().collect::<Vec<f64>>());

        let params = GmmParams {
            n_components: self.n_components,
            max_iter: self.max_iter,
            tol: self.tol,
            reg_covar: self.reg_covar,
            init: self.init.clone(),
            random_state: self.random_state,
            verbose: self.verbose,
        };

        let out = py
            .detach(move || {
                fit_gmm_diag(
                    &data,
                    n_samples,
                    n_features,
                    &params,
                    sample_weight_vec.as_deref(),
                )
            })
            .map_err(map_err)?;
        self.weights_ = Some(out.weights.clone());
        self.means_ = Some(out.means.clone());
        self.covars_ = Some(out.covars.clone());
        self.n_features_ = Some(n_features);
        self.converged_ = Some(out.converged);
        self.lower_bound_ = Some(out.lower_bound);
        self.n_iter_ = Some(out.n_iter);

        let d = PyDict::new(py);
        d.set_item("weights", PyArray1::from_vec(py, out.weights))?;
        let means_arr = vec_to_pyarray2(py, self.n_components, n_features, out.means)?;
        let cov_arr = vec_to_pyarray2(py, self.n_components, n_features, out.covars)?;
        d.set_item("means", means_arr)?;
        d.set_item("covars", cov_arr)?;
        d.set_item("converged", out.converged)?;
        d.set_item("lower_bound", out.lower_bound)?;
        d.set_item("n_iter", out.n_iter)?;
        Ok(d)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        ensure_i64_bounds(self.n_components, "n_components")?;
        let probs = self.predict_proba(py, x)?;
        let probs = probs.readonly();
        let probs = probs.as_array();
        let (n, k) = probs.dim();
        let mut labels = vec![0i64; n];
        for i in 0..n {
            let mut best = 0usize;
            let mut bestv = f64::NEG_INFINITY;
            for c in 0..k {
                let v = probs[[i, c]];
                if v > bestv {
                    bestv = v;
                    best = c;
                }
            }
            labels[i] = best as i64;
        }
        Ok(PyArray1::from_vec(py, labels))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let weights = self.weights_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let means = self.means_.as_ref().unwrap();
        let covars = self.covars_.as_ref().unwrap();
        let d = self.n_features_.unwrap();
        let k = self.n_components;

        let x = x.as_array();
        let (n_samples, nf) = x.dim();
        if nf != d {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature dimension mismatch",
            ));
        }
        let data: Vec<f64> = x.iter().copied().collect();

        let resp = py.detach(|| {
            let (resp, _) = gmm_log_resp_diag(weights, means, covars, &data, n_samples, d, k);
            resp
        });

        vec_to_pyarray2(py, n_samples, k, resp)
    }

    fn score_samples<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let weights = self.weights_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let means = self.means_.as_ref().unwrap();
        let covars = self.covars_.as_ref().unwrap();
        let d = self.n_features_.unwrap();
        let k = self.n_components;

        let x = x.as_array();
        let (n_samples, nf) = x.dim();
        if nf != d {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature dimension mismatch",
            ));
        }
        let data: Vec<f64> = x.iter().copied().collect();

        let scores = py.detach(|| {
            gmm_log_likelihoods_diag(weights, means, covars, &data, n_samples, d, k, None)
        });

        Ok(PyArray1::from_vec(py, scores))
    }

    fn score(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let weights = self.weights_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let means = self.means_.as_ref().unwrap();
        let covars = self.covars_.as_ref().unwrap();
        let d = self.n_features_.unwrap();
        let k = self.n_components;

        let x = x.as_array();
        let (n_samples, nf) = x.dim();
        if nf != d {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "feature dimension mismatch",
            ));
        }
        let data: Vec<f64> = x.iter().copied().collect();

        let mean_ll = py.detach(|| {
            let scores =
                gmm_log_likelihoods_diag(weights, means, covars, &data, n_samples, d, k, None);
            scores.iter().sum::<f64>() / (n_samples as f64)
        });
        Ok(mean_ll)
    }

    fn aic(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let d = self.n_features_.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let k = self.n_components;
        let scores = self.score_samples(py, x)?;
        let scores = scores.readonly();
        let scores = scores.as_array();
        let log_l = scores.iter().sum::<f64>(); // total log-likelihood
        let p = (k as f64 - 1.0) + 2.0 * (k as f64) * (d as f64); // diag covar
        Ok(-2.0 * log_l + 2.0 * p)
    }

    fn bic(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let d = self.n_features_.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let k = self.n_components;
        let scores = self.score_samples(py, x)?;
        let scores = scores.readonly();
        let scores = scores.as_array();
        let n = scores.len();
        let log_l = scores.iter().sum::<f64>(); // total log-likelihood
        let p = (k as f64 - 1.0) + 2.0 * (k as f64) * (d as f64); // diag covar
        Ok(-2.0 * log_l + p * (n as f64).ln())
    }

    fn sample<'py>(
        &self,
        py: Python<'py>,
        n_samples: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let weights = self.weights_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let means = self.means_.as_ref().unwrap();
        let covars = self.covars_.as_ref().unwrap();
        let d = self.n_features_.unwrap();
        let k = self.n_components;
        let out = py
            .detach(|| sample_gmm_diag(weights, means, covars, k, d, n_samples, self.random_state));
        vec_to_pyarray2(py, n_samples, d, out)
    }

    #[getter]
    fn weights_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = self
            .weights_
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        Ok(PyArray1::from_vec(py, w.clone()))
    }
    #[getter]
    fn means_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self
            .means_
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let d = self.n_features_.unwrap();
        vec_to_pyarray2(py, self.n_components, d, m.clone())
    }
    #[getter]
    fn covars_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let c = self
            .covars_
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))?;
        let d = self.n_features_.unwrap();
        vec_to_pyarray2(py, self.n_components, d, c.clone())
    }
    #[getter]
    fn converged_(&self) -> PyResult<bool> {
        self.converged_
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))
    }
    #[getter]
    fn lower_bound_(&self) -> PyResult<f64> {
        self.lower_bound_
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))
    }
    #[getter]
    fn n_iter_(&self) -> PyResult<u32> {
        self.n_iter_
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Model not fitted"))
    }
}

// ---------------- Validation functions ----------------

#[pyfunction]
#[pyo3(signature = (x, labels, *, metric="euclidean"))]
fn silhouette_score(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    labels: Vec<i64>,
    metric: &str,
) -> PyResult<f64> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    let metric = as_metric(metric).map_err(map_err)?;
    py.detach(|| sil_score(&data, n_samples, n_features, &labels, metric))
        .map_err(map_err)
}

#[pyfunction]
fn calinski_harabasz_score(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    labels: Vec<i64>,
) -> PyResult<f64> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    py.detach(|| ch_score(&data, n_samples, n_features, &labels))
        .map_err(map_err)
}

#[pyfunction]
fn davies_bouldin_score(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    labels: Vec<i64>,
) -> PyResult<f64> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    py.detach(|| db_score(&data, n_samples, n_features, &labels))
        .map_err(map_err)
}

// ---------------- Convenience functional APIs ----------------

#[pyfunction(name = "kmeans")]
#[pyo3(signature = (x, k, *, n_init=10, max_iter=300, tol=1e-4, metric="euclidean", normalize_input=None, normalize_centers=None, sample_weight=None, random_state=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn kmeans_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    k: usize,
    n_init: usize,
    max_iter: u32,
    tol: f64,
    metric: &str,
    normalize_input: Option<bool>,
    normalize_centers: Option<bool>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    random_state: Option<u64>,
    verbose: bool,
) -> PyResult<PyKMeansResult<'py>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    let sw: Option<Vec<f64>> = sample_weight.map(|sw| sw.as_array().to_vec());
    if let Some(ref w) = sw
        && w.len() != n_samples
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample_weight length mismatch",
        ));
    }

    let metric = as_metric(metric).map_err(map_err)?;
    let normalize_input = normalize_input.unwrap_or(metric == Metric::Cosine);
    let normalize_centers = normalize_centers.unwrap_or(metric == Metric::Cosine);

    let params = KMeansParams {
        n_clusters: k,
        n_init,
        max_iter,
        tol,
        metric,
        normalize_input,
        normalize_centers,
        random_state,
        verbose,
    };

    let out = py
        .detach(|| fit_kmeans(&data, n_samples, n_features, sw.as_deref(), &params))
        .map_err(map_err)?;
    let centers_arr = vec_to_pyarray2(py, k, n_features, out.centers)?;
    let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
    let labels_arr = PyArray1::from_vec(py, labels_i64);
    Ok((centers_arr, labels_arr, out.inertia, out.n_iter))
}

#[pyfunction]
#[pyo3(signature = (x, k, *, n_init=10, max_iter=300, tol=1e-4, metric="euclidean", normalize_input=None, normalize_centers=None, sample_weight=None, random_state=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn bisecting_kmeans<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    k: usize,
    n_init: usize,
    max_iter: u32,
    tol: f64,
    metric: &str,
    normalize_input: Option<bool>,
    normalize_centers: Option<bool>,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    random_state: Option<u64>,
    verbose: bool,
) -> PyResult<PyKMeansResult<'py>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    let sw: Option<Vec<f64>> = sample_weight.map(|sw| sw.as_array().to_vec());
    if let Some(ref w) = sw
        && w.len() != n_samples
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sample_weight length mismatch",
        ));
    }

    let metric = as_metric(metric).map_err(map_err)?;
    let normalize_input = normalize_input.unwrap_or(metric == Metric::Cosine);
    let normalize_centers = normalize_centers.unwrap_or(metric == Metric::Cosine);

    let params = BisectingParams {
        n_clusters: k,
        n_init,
        max_iter,
        tol,
        metric,
        normalize_input,
        normalize_centers,
        random_state,
        verbose,
    };

    let out = py
        .detach(|| fit_bisecting_kmeans(&data, n_samples, n_features, sw.as_deref(), &params))
        .map_err(map_err)?;
    let centers = vec_to_pyarray2(py, k, n_features, out.centers)?;
    let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
    let labels = PyArray1::from_vec(py, labels_i64);
    Ok((centers, labels, out.inertia, out.n_splits))
}

#[pyfunction(name = "dbscan")]
#[pyo3(signature = (x, *, eps=0.5, min_samples=5, metric="euclidean", normalize_input=None, verbose=false))]
fn dbscan_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    eps: f64,
    min_samples: usize,
    metric: &str,
    normalize_input: Option<bool>,
    verbose: bool,
) -> PyResult<PyDbscanResult<'py>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    let metric = as_metric(metric).map_err(map_err)?;
    let normalize_input = normalize_input.unwrap_or(metric == Metric::Cosine);
    let params = DbscanParams {
        eps,
        min_samples,
        metric,
        normalize_input,
        verbose,
    };
    let out = py
        .detach(|| fit_dbscan(&data, n_samples, n_features, &params))
        .map_err(map_err)?;
    let labels = PyArray1::from_vec(py, out.labels);
    let core_i64 = usize_iter_to_i64(out.core_sample_indices, "core_sample_indices")?;
    let core = PyArray1::from_vec(py, core_i64);
    Ok((labels, core, out.n_clusters))
}

#[pyfunction]
#[pyo3(signature = (x, n_components, *, max_iter=100, tol=1e-3, reg_covar=1e-6, init="kmeans++", sample_weight=None, random_state=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn gaussian_mixture<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    n_components: usize,
    max_iter: u32,
    tol: f64,
    reg_covar: f64,
    init: &str,
    sample_weight: Option<PyReadonlyArray1<f64>>,
    random_state: Option<u64>,
    verbose: bool,
) -> PyResult<PyGaussianMixtureResult<'py>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data: Vec<f64> = x.iter().copied().collect();
    let sample_weight_vec =
        sample_weight.map(|weights| weights.as_array().iter().copied().collect::<Vec<f64>>());
    let params = GmmParams {
        n_components,
        max_iter,
        tol,
        reg_covar,
        init: init.to_string(),
        random_state,
        verbose,
    };
    let out = py
        .detach(move || {
            fit_gmm_diag(
                &data,
                n_samples,
                n_features,
                &params,
                sample_weight_vec.as_deref(),
            )
        })
        .map_err(map_err)?;
    let w = PyArray1::from_vec(py, out.weights);
    let means = vec_to_pyarray2(py, n_components, n_features, out.means)?;
    let covars = vec_to_pyarray2(py, n_components, n_features, out.covars)?;
    let resp = vec_to_pyarray2(py, n_samples, n_components, out.resp)?;
    Ok((
        w,
        means,
        covars,
        resp,
        out.converged,
        out.lower_bound,
        out.n_iter,
    ))
}

// ========================= OPTICS =========================

#[allow(clippy::upper_case_acronyms)]
#[pyclass(subclass, module = "clustor._clustor")]
struct OPTICS {
    min_samples: usize,
    max_eps: f64,
    metric: String,
    normalize_input: Option<bool>,

    ordering_: Option<Vec<usize>>,
    reachability_: Option<Vec<f64>>,
    core_distances_: Option<Vec<f64>>,
    predecessor_: Option<Vec<i32>>,
    n_features_: Option<usize>,
}

#[pymethods]
impl OPTICS {
    #[new]
    #[pyo3(signature = (*, min_samples=5, max_eps=f64::INFINITY, metric="euclidean", normalize_input=None))]
    fn new(
        min_samples: usize,
        max_eps: f64,
        metric: &str,
        normalize_input: Option<bool>,
    ) -> PyResult<Self> {
        if min_samples == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_samples must be > 0",
            ));
        }
        if max_eps < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_eps must be >= 0",
            ));
        }
        Ok(Self {
            min_samples,
            max_eps,
            metric: metric.to_string(),
            normalize_input,
            ordering_: None,
            reachability_: None,
            core_distances_: None,
            predecessor_: None,
            n_features_: None,
        })
    }

    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data = x.to_owned().into_raw_vec_and_offset().0;
        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);
        let params = OpticsParams {
            min_samples: self.min_samples,
            max_eps: self.max_eps,
            metric,
            normalize_input,
        };
        let out = py
            .detach(|| fit_optics(&data, n_samples, n_features, &params))
            .map_err(map_err)?;

        self.ordering_ = Some(out.ordering.clone());
        self.reachability_ = Some(out.reachability.clone());
        self.core_distances_ = Some(out.core_distances.clone());
        self.predecessor_ = Some(out.predecessor.clone());
        self.n_features_ = Some(n_features);

        let d = PyDict::new(py);
        let ord_i64 = usize_iter_to_i64(out.ordering, "ordering")?;
        let pred_i64: Vec<i64> = out.predecessor.into_iter().map(i64::from).collect();
        d.set_item("ordering", PyArray1::from_vec(py, ord_i64))?;
        d.set_item("reachability", PyArray1::from_vec(py, out.reachability))?;
        d.set_item("core_distances", PyArray1::from_vec(py, out.core_distances))?;
        d.set_item("predecessor", PyArray1::from_vec(py, pred_i64))?;
        Ok(d)
    }

    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.fit(py, x)
    }

    #[getter]
    fn ordering_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ord = self.ordering_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let ord_i64 = usize_iter_to_i64(ord.iter().copied(), "ordering")?;
        Ok(PyArray1::from_vec(py, ord_i64))
    }

    #[getter]
    fn reachability_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self.reachability_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        Ok(PyArray1::from_vec(py, v.clone()))
    }

    #[getter]
    fn core_distances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self.core_distances_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        Ok(PyArray1::from_vec(py, v.clone()))
    }

    #[getter]
    fn predecessor_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let v = self.predecessor_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let v_i64: Vec<i64> = v.iter().map(|&x| i64::from(x)).collect();
        Ok(PyArray1::from_vec(py, v_i64))
    }
}

// ===================== Affinity Propagation =====================

#[pyclass(subclass, module = "clustor._clustor")]
struct AffinityPropagation {
    damping: f64,
    max_iter: u32,
    convergence_iter: u32,
    preference: Option<f64>,
    metric: String,
    normalize_input: Option<bool>,
    verbose: bool,

    exemplars_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

#[pymethods]
impl AffinityPropagation {
    #[new]
    #[pyo3(signature = (*, damping=0.5, max_iter=200, convergence_iter=15, preference=None, metric="euclidean", normalize_input=None, verbose=false))]
    fn new(
        damping: f64,
        max_iter: u32,
        convergence_iter: u32,
        preference: Option<f64>,
        metric: &str,
        normalize_input: Option<bool>,
        verbose: bool,
    ) -> PyResult<Self> {
        if !(0.5..1.0).contains(&damping) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "damping must be in [0.5, 1.0)",
            ));
        }
        Ok(Self {
            damping,
            max_iter,
            convergence_iter,
            preference,
            metric: metric.to_string(),
            normalize_input,
            verbose,
            exemplars_: None,
            n_features_: None,
        })
    }

    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data = x.to_owned().into_raw_vec_and_offset().0;
        let metric = as_metric(&self.metric).map_err(map_err)?;
        let normalize_input = self.normalize_input.unwrap_or(metric == Metric::Cosine);
        let params = AffinityParams {
            damping: self.damping,
            max_iter: self.max_iter,
            convergence_iter: self.convergence_iter,
            preference: self.preference,
            metric,
            normalize_input,
            verbose: self.verbose,
        };
        let out = py
            .detach(|| fit_affinity_propagation(&data, n_samples, n_features, &params))
            .map_err(map_err)?;
        self.exemplars_ = Some(out.exemplars.clone());
        self.n_features_ = Some(n_features);

        let d = PyDict::new(py);
        let ex_i64 = usize_iter_to_i64(out.exemplars, "exemplars")?;
        let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
        d.set_item("exemplars", PyArray1::from_vec(py, ex_i64))?;
        d.set_item("labels", PyArray1::from_vec(py, labels_i64))?;
        d.set_item("converged", out.converged)?;
        d.set_item("n_iter", out.n_iter)?;
        Ok(d)
    }

    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = self.fit(py, x)?;
        let labels = d.get_item("labels")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("labels missing from result")
        })?;
        Ok(labels.cast_into()?)
    }

    #[getter]
    fn exemplars_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ex = self.exemplars_.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Model not fitted; call fit()")
        })?;
        let ex_i64 = usize_iter_to_i64(ex.iter().copied(), "exemplars")?;
        Ok(PyArray1::from_vec(py, ex_i64))
    }
}

// ========================= BIRCH =========================

#[pyclass(subclass, module = "clustor._clustor")]
struct Birch {
    threshold: f64,
    branching_factor: usize,
    n_clusters: Option<usize>,
    centers_: Option<Vec<f64>>,
    n_features_: Option<usize>,
    n_subclusters_: Option<usize>,
}

#[pymethods]
impl Birch {
    #[new]
    #[pyo3(signature = (*, threshold=0.5, branching_factor=50, n_clusters=None))]
    fn new(threshold: f64, branching_factor: usize, n_clusters: Option<usize>) -> PyResult<Self> {
        if threshold <= 0.0 || !threshold.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threshold must be finite and > 0",
            ));
        }
        if branching_factor < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "branching_factor must be >= 2",
            ));
        }
        if matches!(n_clusters, Some(0)) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_clusters must be >= 1",
            ));
        }
        Ok(Self {
            threshold,
            branching_factor,
            n_clusters,
            centers_: None,
            n_features_: None,
            n_subclusters_: None,
        })
    }

    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x = x.as_array();
        let (n_samples, n_features) = x.dim();
        let data = x.to_owned().into_raw_vec_and_offset().0;

        let params = BirchParams {
            threshold: self.threshold,
            branching_factor: self.branching_factor,
            n_clusters: self.n_clusters,
        };

        let out = py
            .detach(|| fit_birch(&data, n_samples, n_features, &params))
            .map_err(map_err)?;

        self.centers_ = Some(out.centers.clone());
        self.n_features_ = Some(n_features);
        self.n_subclusters_ = Some(out.n_subclusters);

        let d = PyDict::new(py);
        let n_centers = if let Some(k) = self.n_clusters {
            k
        } else {
            out.n_subclusters
        };
        let centers_arr = vec_to_pyarray2(py, n_centers, n_features, out.centers)?;
        let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
        d.set_item("centers", centers_arr)?;
        d.set_item("labels", PyArray1::from_vec(py, labels_i64))?;
        d.set_item("n_subclusters", out.n_subclusters)?;
        Ok(d)
    }

    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let d = self.fit(py, x)?;
        let labels = d.get_item("labels")?.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("labels missing from result")
        })?;
        Ok(labels.cast_into()?)
    }

    #[getter]
    fn n_subclusters(&self) -> PyResult<Option<usize>> {
        Ok(self.n_subclusters_)
    }
}

// ========================= Functional APIs =========================

#[pyfunction(name = "optics")]
#[pyo3(signature = (x, *, min_samples=5, max_eps=f64::INFINITY, metric="euclidean", normalize_input=None))]
fn optics_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    min_samples: usize,
    max_eps: f64,
    metric: &str,
    normalize_input: Option<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data = x.to_owned().into_raw_vec_and_offset().0;
    let metric = as_metric(metric).map_err(map_err)?;
    let normalize_input = normalize_input.unwrap_or(metric == Metric::Cosine);
    let params = OpticsParams {
        min_samples,
        max_eps,
        metric,
        normalize_input,
    };
    let out = py
        .detach(|| fit_optics(&data, n_samples, n_features, &params))
        .map_err(map_err)?;

    let d = PyDict::new(py);
    let ord_i64 = usize_iter_to_i64(out.ordering, "ordering")?;
    let pred_i64: Vec<i64> = out.predecessor.into_iter().map(i64::from).collect();
    d.set_item("ordering", PyArray1::from_vec(py, ord_i64))?;
    d.set_item("reachability", PyArray1::from_vec(py, out.reachability))?;
    d.set_item("core_distances", PyArray1::from_vec(py, out.core_distances))?;
    d.set_item("predecessor", PyArray1::from_vec(py, pred_i64))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x, *, damping=0.5, max_iter=200, convergence_iter=15, preference=None, metric="euclidean", normalize_input=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn affinity_propagation<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    damping: f64,
    max_iter: u32,
    convergence_iter: u32,
    preference: Option<f64>,
    metric: &str,
    normalize_input: Option<bool>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data = x.to_owned().into_raw_vec_and_offset().0;
    let metric = as_metric(metric).map_err(map_err)?;
    let normalize_input = normalize_input.unwrap_or(metric == Metric::Cosine);
    let params = AffinityParams {
        damping,
        max_iter,
        convergence_iter,
        preference,
        metric,
        normalize_input,
        verbose,
    };
    let out = py
        .detach(|| fit_affinity_propagation(&data, n_samples, n_features, &params))
        .map_err(map_err)?;

    let d = PyDict::new(py);
    let ex_i64 = usize_iter_to_i64(out.exemplars, "exemplars")?;
    let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
    d.set_item("exemplars", PyArray1::from_vec(py, ex_i64))?;
    d.set_item("labels", PyArray1::from_vec(py, labels_i64))?;
    d.set_item("converged", out.converged)?;
    d.set_item("n_iter", out.n_iter)?;
    Ok(d)
}

#[pyfunction(name = "birch")]
#[pyo3(signature = (x, *, threshold=0.5, branching_factor=50, n_clusters=None))]
fn birch_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    threshold: f64,
    branching_factor: usize,
    n_clusters: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data = x.to_owned().into_raw_vec_and_offset().0;
    let params = BirchParams {
        threshold,
        branching_factor,
        n_clusters,
    };
    let out = py
        .detach(|| fit_birch(&data, n_samples, n_features, &params))
        .map_err(map_err)?;

    let d = PyDict::new(py);
    let n_centers = if let Some(k) = n_clusters {
        k
    } else {
        out.n_subclusters
    };
    let centers_arr = vec_to_pyarray2(py, n_centers, n_features, out.centers)?;
    let labels_i64 = usize_iter_to_i64(out.labels, "labels")?;
    d.set_item("centers", centers_arr)?;
    d.set_item("labels", PyArray1::from_vec(py, labels_i64))?;
    d.set_item("n_subclusters", out.n_subclusters)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (x, *, method="average", metric="euclidean"))]
fn hac_dendrogram<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    method: &str,
    metric: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = x.as_array();
    let (n_samples, n_features) = x.dim();
    let data = x.to_owned().into_raw_vec_and_offset().0;
    let link = Linkage::parse(method).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unknown linkage method: {method}"))
    })?;
    let metric = as_metric(metric).map_err(map_err)?;
    let z = py
        .detach(|| hac_linkage(&data, n_samples, n_features, link, metric))
        .map_err(map_err)?;
    vec_to_pyarray2(py, n_samples - 1, 4, z)
}

#[pyfunction]
fn version() -> &'static str {
    VERSION
}

#[pymodule]
fn _clustor(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(kmeans_py, m)?)?;
    m.add_function(wrap_pyfunction!(bisecting_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(dbscan_py, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_mixture, m)?)?;
    m.add_function(wrap_pyfunction!(optics_py, m)?)?;
    m.add_function(wrap_pyfunction!(affinity_propagation, m)?)?;
    m.add_function(wrap_pyfunction!(birch_py, m)?)?;
    m.add_function(wrap_pyfunction!(hac_dendrogram, m)?)?;
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<KMeans>()?;
    m.add_class::<MiniBatchKMeans>()?;
    m.add_class::<BisectingKMeans>()?;
    m.add_class::<DBSCAN>()?;
    m.add_class::<GaussianMixture>()?;
    m.add_class::<OPTICS>()?;
    m.add_class::<AffinityPropagation>()?;
    m.add_class::<Birch>()?;
    Ok(())
}

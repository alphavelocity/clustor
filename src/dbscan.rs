// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance, euclidean_sq, normalize_in_place};
use crate::utils::{compute_row_norms, validate_data_shape};

#[derive(Clone, Debug)]
pub struct DbscanOutput {
    pub labels: Vec<i64>, // -1 noise, 0..n_clusters-1 clusters
    pub core_sample_indices: Vec<usize>,
    pub n_clusters: usize,
}

#[derive(Clone, Debug)]
pub struct DbscanParams {
    pub eps: f64,
    pub min_samples: usize,
    pub metric: Metric,
    pub normalize_input: bool,
    pub verbose: bool,
}

fn validate_inputs(data: &[f64], n_samples: usize, n_features: usize) -> ClustorResult<()> {
    if n_samples == 0 || n_features == 0 {
        return Err(ClustorError::InvalidArg(
            "X must be non-empty 2D array".into(),
        ));
    }
    validate_data_shape(
        data.len(),
        n_samples,
        n_features,
        "X data length does not match shape",
        "X shape product overflows usize",
    )?;
    Ok(())
}

fn maybe_normalize_input(data: &mut [f64], n_samples: usize, n_features: usize) {
    for i in 0..n_samples {
        let row = &mut data[i * n_features..(i + 1) * n_features];
        normalize_in_place(row);
    }
}

fn region_query_euclidean(
    i: usize,
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    eps_sq: f64,
    out: &mut Vec<usize>,
) {
    out.clear();
    let xi = &data[i * n_features..(i + 1) * n_features];
    for j in 0..n_samples {
        let xj = &data[j * n_features..(j + 1) * n_features];
        if euclidean_sq(xi, xj) <= eps_sq {
            out.push(j);
        }
    }
}

fn region_query_cosine(
    i: usize,
    data: &[f64],
    norms: &[f64],
    n_samples: usize,
    n_features: usize,
    eps: f64,
    out: &mut Vec<usize>,
) {
    out.clear();
    let xi = &data[i * n_features..(i + 1) * n_features];
    let ni = norms[i];
    for j in 0..n_samples {
        let xj = &data[j * n_features..(j + 1) * n_features];
        if cosine_distance(xi, ni, xj, norms[j]) <= eps {
            out.push(j);
        }
    }
}

fn fit_dbscan_with_region_query<F>(
    n_samples: usize,
    params: &DbscanParams,
    mut region_query: F,
) -> DbscanOutput
where
    F: FnMut(usize, &mut Vec<usize>),
{
    // labels: -99 = unvisited, -1 = noise, >=0 cluster id
    let mut labels = vec![-99i64; n_samples];
    let mut is_core = vec![false; n_samples];

    // Deduplicate queued seeds per cluster using an epoch marker vector.
    let mut seed_marks = vec![0u32; n_samples];
    let mut current_mark = 1u32;

    let mut neighbors = Vec::with_capacity(n_samples);
    let mut seeds = Vec::with_capacity(n_samples);

    let mut cluster_id: i64 = 0;
    for i in 0..n_samples {
        if labels[i] != -99 {
            continue;
        }

        region_query(i, &mut neighbors);
        if neighbors.len() < params.min_samples {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;
        is_core[i] = true;

        seeds.clear();
        seeds.extend_from_slice(&neighbors);

        for &p in &seeds {
            seed_marks[p] = current_mark;
        }

        let mut idx = 0usize;
        while idx < seeds.len() {
            let p = seeds[idx];
            idx += 1;

            if labels[p] == -1 {
                labels[p] = cluster_id;
            }
            if labels[p] != -99 {
                continue;
            }
            labels[p] = cluster_id;

            region_query(p, &mut neighbors);
            if neighbors.len() >= params.min_samples {
                is_core[p] = true;
                for &q in &neighbors {
                    if labels[q] == -1 {
                        labels[q] = cluster_id;
                    } else if labels[q] == -99 && seed_marks[q] != current_mark {
                        seed_marks[q] = current_mark;
                        seeds.push(q);
                    }
                }
            }
        }

        current_mark = current_mark.wrapping_add(1);
        if current_mark == 0 {
            seed_marks.fill(0);
            current_mark = 1;
        }

        if params.verbose {
            eprintln!("[Clustor][DBSCAN] formed cluster {}", cluster_id);
        }
        cluster_id += 1;
    }

    let n_clusters = cluster_id.max(0) as usize;
    let core_sample_indices = is_core
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b && labels[i] >= 0 { Some(i) } else { None })
        .collect();

    DbscanOutput {
        labels,
        core_sample_indices,
        n_clusters,
    }
}

pub fn fit_dbscan(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &DbscanParams,
) -> ClustorResult<DbscanOutput> {
    validate_inputs(data_in, n_samples, n_features)?;
    if params.eps <= 0.0 || !params.eps.is_finite() {
        return Err(ClustorError::InvalidArg(
            "eps must be finite and > 0".into(),
        ));
    }
    if params.min_samples == 0 {
        return Err(ClustorError::InvalidArg("min_samples must be > 0".into()));
    }

    let mut data = data_in.to_vec();
    if params.metric == Metric::Cosine && params.normalize_input {
        maybe_normalize_input(&mut data, n_samples, n_features);
    }

    let out = match params.metric {
        Metric::Euclidean => {
            let eps_sq = params.eps * params.eps;
            fit_dbscan_with_region_query(n_samples, params, |i, out| {
                region_query_euclidean(i, &data, n_samples, n_features, eps_sq, out)
            })
        }
        Metric::Cosine => {
            let data_norms = compute_row_norms(&data, n_samples, n_features);
            fit_dbscan_with_region_query(n_samples, params, |i, out| {
                region_query_cosine(
                    i,
                    &data,
                    &data_norms,
                    n_samples,
                    n_features,
                    params.eps,
                    out,
                )
            })
        }
    };

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dbscan_rejects_overflowing_shape_product() {
        let params = DbscanParams {
            eps: 0.5,
            min_samples: 2,
            metric: Metric::Euclidean,
            normalize_input: false,
            verbose: false,
        };

        let err = fit_dbscan(&[], usize::MAX, 2, &params).unwrap_err();
        let ClustorError::InvalidArg(msg) = err;
        assert!(msg.contains("overflows"));
    }

    #[test]
    fn dbscan_relabels_noise_when_density_reachable() {
        // point 0 is visited first and is not a core sample on its own,
        // but must be relabeled when expanded from point 1.
        let data = vec![-0.14, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0];
        let params = DbscanParams {
            eps: 0.15,
            min_samples: 4,
            metric: Metric::Euclidean,
            normalize_input: false,
            verbose: false,
        };

        let out = fit_dbscan(&data, 4, 2, &params).expect("dbscan should succeed");
        assert_eq!(out.n_clusters, 1);
        assert!(out.labels.iter().all(|&v| v == out.labels[0]));
        assert_ne!(out.labels[0], -1);
    }
}

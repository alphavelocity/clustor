// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance, euclidean_sq, normalize_in_place};
use crate::utils::compute_row_norms;

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
    if data.len() != n_samples * n_features {
        return Err(ClustorError::InvalidArg(
            "X data length does not match shape".into(),
        ));
    }
    Ok(())
}

fn maybe_normalize_input(data: &mut [f64], n_samples: usize, n_features: usize) {
    for i in 0..n_samples {
        let row = &mut data[i * n_features..(i + 1) * n_features];
        normalize_in_place(row);
    }
}

#[allow(clippy::too_many_arguments)]
fn region_query(
    i: usize,
    data: &[f64],
    data_norms: Option<&[f64]>,
    n_samples: usize,
    n_features: usize,
    metric: Metric,
    eps: f64,
    eps_sq: f64,
) -> Vec<usize> {
    let xi = &data[i * n_features..(i + 1) * n_features];
    let mut neigh = Vec::new();
    for j in 0..n_samples {
        let xj = &data[j * n_features..(j + 1) * n_features];
        let ok = match metric {
            Metric::Euclidean => euclidean_sq(xi, xj) <= eps_sq,
            Metric::Cosine => {
                let ni = data_norms.unwrap()[i];
                let nj = data_norms.unwrap()[j];
                cosine_distance(xi, ni, xj, nj) <= eps
            }
        };
        if ok {
            neigh.push(j);
        }
    }
    neigh
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
    let data_norms = if params.metric == Metric::Cosine {
        Some(compute_row_norms(&data, n_samples, n_features))
    } else {
        None
    };

    let eps_sq = params.eps * params.eps;
    // labels: -99 = unvisited, -1 = noise, >=0 cluster id
    let mut labels = vec![-99i64; n_samples];
    let mut is_core = vec![false; n_samples];

    let mut cluster_id: i64 = 0;
    for i in 0..n_samples {
        if labels[i] != -99 {
            continue;
        }

        let neigh = region_query(
            i,
            &data,
            data_norms.as_deref(),
            n_samples,
            n_features,
            params.metric,
            params.eps,
            eps_sq,
        );
        if neigh.len() < params.min_samples {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;
        is_core[i] = true;
        let mut seeds = neigh;
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

            let neigh_p = region_query(
                p,
                &data,
                data_norms.as_deref(),
                n_samples,
                n_features,
                params.metric,
                params.eps,
                eps_sq,
            );
            if neigh_p.len() >= params.min_samples {
                is_core[p] = true;
                seeds.extend(neigh_p);
            }
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

    Ok(DbscanOutput {
        labels,
        core_sample_indices,
        n_clusters,
    })
}

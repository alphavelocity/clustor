// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ordered_float::OrderedFloat;

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance, euclidean_sq, normalize_in_place};
use crate::utils::compute_row_norms;

#[derive(Clone, Debug)]
pub struct OpticsParams {
    pub min_samples: usize,
    pub max_eps: f64, // use INFINITY for "no cap"
    pub metric: Metric,
    pub normalize_input: bool, // only relevant for cosine
}

#[derive(Clone, Debug)]
pub struct OpticsOutput {
    pub ordering: Vec<usize>,
    pub reachability: Vec<f64>,
    pub core_distances: Vec<f64>,
    pub predecessor: Vec<i32>,
}

fn validate_inputs(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &OpticsParams,
) -> ClustorResult<()> {
    if n_samples == 0 {
        return Err(ClustorError::InvalidArg("n_samples must be > 0".into()));
    }
    if n_features == 0 {
        return Err(ClustorError::InvalidArg("n_features must be > 0".into()));
    }
    if data.len() != n_samples * n_features {
        return Err(ClustorError::InvalidArg("data length mismatch".into()));
    }
    if params.min_samples == 0 {
        return Err(ClustorError::InvalidArg("min_samples must be > 0".into()));
    }
    if !params.max_eps.is_finite() && params.max_eps != f64::INFINITY {
        return Err(ClustorError::InvalidArg(
            "max_eps must be finite or +inf".into(),
        ));
    }
    if params.max_eps < 0.0 {
        return Err(ClustorError::InvalidArg("max_eps must be >= 0".into()));
    }
    Ok(())
}

#[inline]
fn dist(
    data: &[f64],
    norms: Option<&[f64]>,
    i: usize,
    j: usize,
    n_features: usize,
    metric: Metric,
) -> f64 {
    let a = &data[i * n_features..(i + 1) * n_features];
    let b = &data[j * n_features..(j + 1) * n_features];
    match metric {
        Metric::Euclidean => euclidean_sq(a, b).sqrt(),
        Metric::Cosine => {
            let n = norms.expect("norms required for cosine");
            cosine_distance(a, n[i], b, n[j]).max(0.0)
        }
    }
}

/// Returns neighbor indices and distances (including `p` itself at distance 0.0).
fn neighbors_within_eps(
    data: &[f64],
    norms: Option<&[f64]>,
    p: usize,
    n_samples: usize,
    n_features: usize,
    metric: Metric,
    max_eps: f64,
) -> (Vec<usize>, Vec<f64>) {
    let mut idxs = Vec::new();
    let mut ds = Vec::new();

    // Always include self
    idxs.push(p);
    ds.push(0.0);

    let cap = max_eps;
    for j in 0..n_samples {
        if j == p {
            continue;
        }
        let d = dist(data, norms, p, j, n_features, metric);
        if cap == f64::INFINITY || d <= cap {
            idxs.push(j);
            ds.push(d);
        }
    }
    (idxs, ds)
}

fn core_distance_from_dists(mut dists: Vec<f64>, min_samples: usize) -> f64 {
    // min_samples includes the point itself => core distance is distance to min_samples-th nearest neighbor
    if dists.len() < min_samples {
        return f64::INFINITY;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists[min_samples - 1]
}

pub fn fit_optics(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &OpticsParams,
) -> ClustorResult<OpticsOutput> {
    validate_inputs(data_in, n_samples, n_features, params)?;

    let mut data: Vec<f64>;
    let data_ref: &[f64] = if params.metric == Metric::Cosine && params.normalize_input {
        data = data_in.to_vec();
        for i in 0..n_samples {
            let row = &mut data[i * n_features..(i + 1) * n_features];
            normalize_in_place(row);
        }
        &data
    } else {
        data_in
    };

    let norms = if params.metric == Metric::Cosine && !params.normalize_input {
        Some(compute_row_norms(data_ref, n_samples, n_features))
    } else if params.metric == Metric::Cosine {
        Some(vec![1.0; n_samples])
    } else {
        None
    };
    let norms_ref = norms.as_deref();

    let mut processed = vec![false; n_samples];
    let mut ordering: Vec<usize> = Vec::with_capacity(n_samples);
    let mut reachability = vec![f64::INFINITY; n_samples];
    let mut core_distances = vec![f64::INFINITY; n_samples];
    let mut predecessor = vec![-1i32; n_samples];

    // Min-heap via Reverse
    let mut seeds: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)> = BinaryHeap::new();

    for start in 0..n_samples {
        if processed[start] {
            continue;
        }

        // Expand cluster order from this start point
        let (nbrs, dists) = neighbors_within_eps(
            data_ref,
            norms_ref,
            start,
            n_samples,
            n_features,
            params.metric,
            params.max_eps,
        );
        processed[start] = true;
        ordering.push(start);

        let core = core_distance_from_dists(dists.clone(), params.min_samples);
        core_distances[start] = core;

        if core.is_finite() {
            // Update seeds
            for (&o, &dpo) in nbrs.iter().zip(dists.iter()) {
                if processed[o] {
                    continue;
                }
                let new_reach = core.max(dpo);
                if new_reach < reachability[o] {
                    reachability[o] = new_reach;
                    predecessor[o] = start as i32;
                    seeds.push((Reverse(OrderedFloat(new_reach)), o));
                }
            }

            while let Some((Reverse(rq), q)) = seeds.pop() {
                let rq = rq.0;
                if processed[q] {
                    continue;
                }
                if rq > reachability[q] {
                    continue;
                } // stale entry
                let (nbrs_q, dists_q) = neighbors_within_eps(
                    data_ref,
                    norms_ref,
                    q,
                    n_samples,
                    n_features,
                    params.metric,
                    params.max_eps,
                );
                processed[q] = true;
                ordering.push(q);

                let core_q = core_distance_from_dists(dists_q.clone(), params.min_samples);
                core_distances[q] = core_q;
                if core_q.is_finite() {
                    for (&o, &dqo) in nbrs_q.iter().zip(dists_q.iter()) {
                        if processed[o] {
                            continue;
                        }
                        let new_reach = core_q.max(dqo);
                        if new_reach < reachability[o] {
                            reachability[o] = new_reach;
                            predecessor[o] = q as i32;
                            seeds.push((Reverse(OrderedFloat(new_reach)), o));
                        }
                    }
                }
            }
        }
    }

    Ok(OpticsOutput {
        ordering,
        reachability,
        core_distances,
        predecessor,
    })
}

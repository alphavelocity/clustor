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
use crate::utils::{compute_row_norms, validate_data_shape};

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
    if n_samples > i32::MAX as usize {
        return Err(ClustorError::InvalidArg(
            "n_samples must be <= i32::MAX".into(),
        ));
    }
    validate_data_shape(
        data.len(),
        n_samples,
        n_features,
        "data length mismatch",
        "shape product overflows usize",
    )?;
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
fn sanitize_distance(d: f64) -> f64 {
    if d.is_finite() { d } else { f64::INFINITY }
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
    let d = match metric {
        Metric::Euclidean => euclidean_sq(a, b).sqrt(),
        Metric::Cosine => {
            let n = norms.expect("norms required for cosine");
            cosine_distance(a, n[i], b, n[j]).max(0.0)
        }
    };
    sanitize_distance(d)
}

/// Returns neighbor `(index, distance)` pairs (including `p` itself at distance 0.0).
fn neighbors_within_eps(
    data: &[f64],
    norms: Option<&[f64]>,
    p: usize,
    n_samples: usize,
    n_features: usize,
    metric: Metric,
    max_eps: f64,
) -> Vec<(usize, f64)> {
    let mut neighbors = Vec::with_capacity(n_samples);

    // Always include self.
    neighbors.push((p, 0.0));

    if max_eps == f64::INFINITY {
        for j in 0..p {
            let d = dist(data, norms, p, j, n_features, metric);
            neighbors.push((j, d));
        }
        for j in (p + 1)..n_samples {
            let d = dist(data, norms, p, j, n_features, metric);
            neighbors.push((j, d));
        }
        return neighbors;
    }

    for j in 0..p {
        let d = dist(data, norms, p, j, n_features, metric);
        if d <= max_eps {
            neighbors.push((j, d));
        }
    }
    for j in (p + 1)..n_samples {
        let d = dist(data, norms, p, j, n_features, metric);
        if d <= max_eps {
            neighbors.push((j, d));
        }
    }
    neighbors
}

fn core_distance_from_neighbors(neighbors: &[(usize, f64)], min_samples: usize) -> f64 {
    // min_samples includes the point itself => core distance is distance to min_samples-th nearest neighbor
    let n = neighbors.len();
    if n < min_samples {
        return f64::INFINITY;
    }

    // Fast path: min_samples includes self at distance 0.0.
    if min_samples == 1 {
        return 0.0;
    }

    // Fast path: k == n means the largest neighbor distance.
    if min_samples == n {
        let mut max_d = f64::NEG_INFINITY;
        for &(_, d) in neighbors {
            if d > max_d {
                max_d = d;
            }
        }
        return max_d;
    }

    // Track the min_samples smallest distances in a bounded max-heap.
    // This avoids copying all distances and keeps the computation non-mutating.
    let mut smallest: BinaryHeap<OrderedFloat<f64>> = BinaryHeap::with_capacity(min_samples);
    for &(_, d) in neighbors {
        let od = OrderedFloat(d);
        if smallest.len() < min_samples {
            smallest.push(od);
            continue;
        }
        if let Some(&top) = smallest.peek()
            && od < top
        {
            smallest.pop();
            smallest.push(od);
        }
    }
    smallest.peek().map_or(f64::INFINITY, |v| v.0)
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

    // Min-heap via Reverse.
    let mut seeds: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)> = BinaryHeap::new();

    for start in 0..n_samples {
        if processed[start] {
            continue;
        }

        let neighbors = neighbors_within_eps(
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
        // Safety: validated by `validate_inputs` (n_samples <= i32::MAX).
        debug_assert!(start <= i32::MAX as usize);
        let start_i32 = start as i32;

        let core = core_distance_from_neighbors(&neighbors, params.min_samples);
        core_distances[start] = core;

        if core.is_finite() {
            for &(o, dpo) in &neighbors {
                if processed[o] {
                    continue;
                }
                let new_reach = core.max(dpo);
                if new_reach < reachability[o] {
                    reachability[o] = new_reach;
                    predecessor[o] = start_i32;
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

                let neighbors_q = neighbors_within_eps(
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
                debug_assert!(q <= i32::MAX as usize);
                let q_i32 = q as i32;

                let core_q = core_distance_from_neighbors(&neighbors_q, params.min_samples);
                core_distances[q] = core_q;
                if core_q.is_finite() {
                    for &(o, dqo) in &neighbors_q {
                        if processed[o] {
                            continue;
                        }
                        let new_reach = core_q.max(dqo);
                        if new_reach < reachability[o] {
                            reachability[o] = new_reach;
                            predecessor[o] = q_i32;
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

#[cfg(test)]
mod tests {
    use super::{
        Metric, OpticsParams, core_distance_from_neighbors, fit_optics, neighbors_within_eps,
    };
    #[test]
    fn fit_optics_rejects_sample_count_larger_than_i32_max() {
        let params = OpticsParams {
            min_samples: 2,
            max_eps: f64::INFINITY,
            metric: Metric::Euclidean,
            normalize_input: false,
        };
        let err = fit_optics(&[], i32::MAX as usize + 1, 1, &params).expect_err("must reject");
        assert!(
            matches!(err, crate::errors::ClustorError::InvalidArg(msg) if msg == "n_samples must be <= i32::MAX")
        );
    }

    #[test]
    fn core_distance_selects_min_samples_neighbor_without_full_sort() {
        let neighbors = vec![(0, 0.8), (1, 0.1), (2, 0.0), (3, 0.6), (4, 0.4), (5, 0.2)];
        let core = core_distance_from_neighbors(&neighbors, 4);
        assert!((core - 0.4).abs() < 1e-12);
    }

    #[test]
    fn core_distance_is_infinite_when_not_enough_neighbors() {
        let neighbors = vec![(0, 0.0), (1, 0.2)];
        let core = core_distance_from_neighbors(&neighbors, 3);
        assert!(core.is_infinite());
    }

    #[test]
    fn core_distance_with_single_required_sample_is_zero_for_self_inclusion() {
        let neighbors = vec![(0, 0.0), (1, 2.0), (2, 3.0)];
        let core = core_distance_from_neighbors(&neighbors, 1);
        assert_eq!(core, 0.0);
    }

    #[test]
    fn core_distance_single_sample_zero_even_with_nonfinite_others() {
        let neighbors = vec![(0, 0.0), (1, f64::INFINITY), (2, 9.0)];
        let core = core_distance_from_neighbors(&neighbors, 1);
        assert_eq!(core, 0.0);
    }

    #[test]
    fn core_distance_handles_infinite_neighbors() {
        let neighbors = vec![(0, 0.0), (1, f64::INFINITY), (2, 5.0)];
        let core = core_distance_from_neighbors(&neighbors, 2);
        assert_eq!(core, 5.0);
    }

    #[test]
    fn core_distance_with_k_equal_to_neighbor_count_returns_max() {
        let neighbors = vec![(0, 0.0), (1, 1.5), (2, 4.0), (3, 2.0)];
        let core = core_distance_from_neighbors(&neighbors, neighbors.len());
        assert_eq!(core, 4.0);
    }

    #[test]
    fn core_distance_does_not_reorder_neighbors() {
        let neighbors = vec![(10, 0.3), (11, 0.2), (12, 0.1)];
        let _ = core_distance_from_neighbors(&neighbors, 2);
        assert_eq!(neighbors, vec![(10, 0.3), (11, 0.2), (12, 0.1)]);
    }

    #[test]
    fn neighbors_with_zero_eps_include_only_self() {
        let data = vec![0.0, 1.0, 2.0];
        let neighbors = neighbors_within_eps(&data, None, 1, 3, 1, Metric::Euclidean, 0.0);
        assert_eq!(neighbors, vec![(1, 0.0)]);
    }

    #[test]
    fn neighbors_with_infinite_eps_include_all_except_self_once() {
        let data = vec![0.0, 1.0, 2.0, 4.0];
        let neighbors =
            neighbors_within_eps(&data, None, 1, 4, 1, Metric::Euclidean, f64::INFINITY);
        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0], (1, 0.0));
        let idxs: Vec<usize> = neighbors.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(idxs, vec![1, 0, 2, 3]);
    }

    #[test]
    fn euclidean_distance_nan_from_infinities_is_treated_as_infinite() {
        let data = vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0];
        let neighbors =
            neighbors_within_eps(&data, None, 0, 2, 2, Metric::Euclidean, f64::INFINITY);
        assert_eq!(neighbors[0], (0, 0.0));
        assert_eq!(neighbors[1], (1, f64::INFINITY));
    }

    #[test]
    fn nan_euclidean_distances_are_safely_mapped_to_infinity() {
        let data = vec![f64::NAN, 0.0, 1.0, 0.0];
        let neighbors =
            neighbors_within_eps(&data, None, 0, 2, 2, Metric::Euclidean, f64::INFINITY);
        assert_eq!(neighbors, vec![(0, 0.0), (1, f64::INFINITY)]);
    }

    #[test]
    fn optics_reachability_is_stable_with_core_distance_selection() {
        // 1D chain where each point should be density-reachable from immediate neighbors.
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let params = OpticsParams {
            min_samples: 2,
            max_eps: f64::INFINITY,
            metric: Metric::Euclidean,
            normalize_input: false,
        };

        let out = fit_optics(&data, 4, 1, &params).expect("optics should succeed");

        let start = out.ordering[0];
        for i in 0..4 {
            if i == start {
                continue;
            }
            assert!(out.reachability[i].is_finite());
            assert!(out.predecessor[i] >= 0);
            assert!(out.predecessor[i] < i32::MAX);
        }
    }
}

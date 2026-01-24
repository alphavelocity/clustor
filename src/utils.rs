// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::metrics::{Metric, cosine_distance, euclidean_sq, l2_norm};
use rand::Rng;

pub fn compute_row_norms(data: &[f64], n_samples: usize, n_features: usize) -> Vec<f64> {
    let mut norms = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row = &data[i * n_features..(i + 1) * n_features];
        norms.push(l2_norm(row));
    }
    norms
}

#[inline]
pub fn pick_random_index<R: Rng>(rng: &mut R, n: usize) -> usize {
    rng.random_range(0..n)
}

/// KMeans++ initialization (Arthur & Vassilvitskii, 2007).
/// Returns centers as a flat Vec (k * n_features).
#[allow(clippy::too_many_arguments)]
pub fn kmeans_plus_plus<R: Rng>(
    rng: &mut R,
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    k: usize,
    metric: Metric,
    data_norms: Option<&[f64]>,
    normalize_centers_cosine: bool,
) -> Vec<f64> {
    let mut centers = vec![0.0; k * n_features];

    // 1) pick first center randomly
    let first = pick_random_index(rng, n_samples);
    centers[0..n_features].copy_from_slice(&data[first * n_features..(first + 1) * n_features]);
    if metric == Metric::Cosine && normalize_centers_cosine {
        crate::metrics::normalize_in_place(&mut centers[0..n_features]);
    }

    // precompute center norms if needed (cosine)
    let mut center_norms: Vec<f64> = Vec::new();
    if metric == Metric::Cosine {
        center_norms = vec![crate::metrics::l2_norm(&centers[0..n_features])];
    }

    // 2) pick remaining centers
    let mut dist = vec![0.0; n_samples];
    for c in 1..k {
        for i in 0..n_samples {
            let x = &data[i * n_features..(i + 1) * n_features];
            let mut best = f64::INFINITY;
            for j in 0..c {
                let cent = &centers[j * n_features..(j + 1) * n_features];
                let d = match metric {
                    Metric::Euclidean => euclidean_sq(x, cent), // squared
                    Metric::Cosine => {
                        let xn = data_norms.unwrap()[i];
                        let cn = center_norms[j];
                        let cd = cosine_distance(x, xn, cent, cn);
                        cd * cd
                    }
                };
                if d < best {
                    best = d;
                }
            }
            dist[i] = best;
        }

        let sum: f64 = dist.iter().sum();
        let next_idx = if sum <= 0.0 || !sum.is_finite() {
            pick_random_index(rng, n_samples)
        } else {
            let mut r = rng.random_range(0.0..sum);
            let mut chosen = 0usize;
            for (i, d) in dist.iter().enumerate() {
                if *d >= r {
                    chosen = i;
                    break;
                }
                r -= *d;
            }
            chosen
        };

        let start = c * n_features;
        let end = start + n_features;
        centers[start..end]
            .copy_from_slice(&data[next_idx * n_features..(next_idx + 1) * n_features]);
        if metric == Metric::Cosine && normalize_centers_cosine {
            crate::metrics::normalize_in_place(&mut centers[start..end]);
            center_norms.push(1.0);
        } else if metric == Metric::Cosine {
            center_norms.push(crate::metrics::l2_norm(&centers[start..end]));
        }
    }

    centers
}

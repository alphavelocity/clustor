// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::metrics::{Metric, cosine_distance, euclidean_sq, l2_norm};
use crate::{errors::ClustorError, errors::ClustorResult};
use rand::RngExt;

#[inline]
pub fn validate_data_shape(
    data_len: usize,
    n_samples: usize,
    n_features: usize,
    mismatch_msg: &'static str,
    overflow_msg: &'static str,
) -> ClustorResult<()> {
    let expected = n_samples
        .checked_mul(n_features)
        .ok_or_else(|| ClustorError::InvalidArg(overflow_msg.into()))?;
    if data_len != expected {
        return Err(ClustorError::InvalidArg(mismatch_msg.into()));
    }
    Ok(())
}

pub fn compute_row_norms(data: &[f64], n_samples: usize, n_features: usize) -> Vec<f64> {
    let mut norms = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let row = &data[i * n_features..(i + 1) * n_features];
        norms.push(l2_norm(row));
    }
    norms
}

#[inline]
pub fn pick_random_index<R: RngExt>(rng: &mut R, n: usize) -> usize {
    rng.random_range(0..n)
}

#[inline]
fn sample_weighted_index<R: RngExt>(rng: &mut R, weights: &[f64]) -> usize {
    debug_assert!(!weights.is_empty(), "weights must not be empty");
    if weights.is_empty() {
        return 0;
    }

    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return pick_random_index(rng, weights.len());
    }

    let mut r = rng.random_range(0.0..sum);
    for (i, &w) in weights.iter().enumerate() {
        if r < w {
            return i;
        }
        r -= w;
    }

    // Floating-point accumulation can leave a tiny remainder.
    weights.len() - 1
}

/// KMeans++ initialization (Arthur & Vassilvitskii, 2007).
/// Returns centers as a flat Vec (k * n_features).
#[allow(clippy::too_many_arguments)]
pub fn kmeans_plus_plus<R: RngExt>(
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

    // 2) pick remaining centers
    let mut dist = vec![0.0; n_samples];
    match metric {
        Metric::Euclidean => {
            for c in 1..k {
                for i in 0..n_samples {
                    let x = &data[i * n_features..(i + 1) * n_features];
                    let mut best = f64::INFINITY;
                    for j in 0..c {
                        let cent = &centers[j * n_features..(j + 1) * n_features];
                        let d = euclidean_sq(x, cent); // squared
                        if d < best {
                            best = d;
                        }
                    }
                    dist[i] = best;
                }

                let next_idx = sample_weighted_index(rng, &dist);
                let start = c * n_features;
                let end = start + n_features;
                centers[start..end]
                    .copy_from_slice(&data[next_idx * n_features..(next_idx + 1) * n_features]);
            }
        }
        Metric::Cosine => {
            let data_norms = data_norms.expect("norms required for cosine");
            let mut center_norms = vec![crate::metrics::l2_norm(&centers[0..n_features])];

            for c in 1..k {
                for i in 0..n_samples {
                    let x = &data[i * n_features..(i + 1) * n_features];
                    let xn = data_norms[i];
                    let mut best = f64::INFINITY;
                    for j in 0..c {
                        let cent = &centers[j * n_features..(j + 1) * n_features];
                        let cn = center_norms[j];
                        let cd = cosine_distance(x, xn, cent, cn);
                        let d = cd * cd;
                        if d < best {
                            best = d;
                        }
                    }
                    dist[i] = best;
                }

                let next_idx = sample_weighted_index(rng, &dist);
                let start = c * n_features;
                let end = start + n_features;
                centers[start..end]
                    .copy_from_slice(&data[next_idx * n_features..(next_idx + 1) * n_features]);
                if normalize_centers_cosine {
                    crate::metrics::normalize_in_place(&mut centers[start..end]);
                    center_norms.push(1.0);
                } else {
                    center_norms.push(crate::metrics::l2_norm(&centers[start..end]));
                }
            }
        }
    }

    centers
}

#[cfg(test)]
mod tests {
    use super::{sample_weighted_index, validate_data_shape};
    use crate::errors::ClustorError;
    use rand::SeedableRng;

    #[test]
    fn weighted_sampling_picks_only_positive_mass_index() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..32 {
            let idx = sample_weighted_index(&mut rng, &[0.0, 0.0, 2.0, 0.0]);
            assert_eq!(idx, 2);
        }
    }

    #[test]
    fn weighted_sampling_returns_in_range_when_sum_invalid() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for _ in 0..64 {
            let idx = sample_weighted_index(&mut rng, &[f64::NAN, 1.0, 2.0]);
            assert!(idx < 3);
        }
    }

    #[test]
    fn validate_data_shape_rejects_overflowing_shape_product() {
        let err = validate_data_shape(0, usize::MAX, 2, "mismatch", "overflow")
            .expect_err("overflow should be rejected");
        assert!(matches!(err, ClustorError::InvalidArg(msg) if msg == "overflow"));
    }

    #[test]
    fn validate_data_shape_rejects_mismatched_length() {
        let err = validate_data_shape(11, 3, 4, "mismatch", "overflow")
            .expect_err("mismatch should be rejected");
        assert!(matches!(err, ClustorError::InvalidArg(msg) if msg == "mismatch"));
    }

    #[test]
    fn validate_data_shape_accepts_matching_product() {
        let out = validate_data_shape(12, 3, 4, "mismatch", "overflow");
        assert!(out.is_ok());
    }
}

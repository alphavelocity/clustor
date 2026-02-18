// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance};
use crate::utils::compute_row_norms;

fn validate(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
) -> ClustorResult<()> {
    if n_samples == 0 || n_features == 0 {
        return Err(ClustorError::InvalidArg("X must be non-empty".into()));
    }
    let expected_len = n_samples
        .checked_mul(n_features)
        .ok_or_else(|| ClustorError::InvalidArg("Input too large".into()))?;
    if data.len() != expected_len {
        return Err(ClustorError::InvalidArg("X length mismatch".into()));
    }
    if labels.len() != n_samples {
        return Err(ClustorError::InvalidArg("labels length mismatch".into()));
    }
    for (i, &lab) in labels.iter().enumerate() {
        if lab < -1 {
            return Err(ClustorError::InvalidArg(
                "labels must be >= -1, where -1 denotes noise".into(),
            ));
        }
        if lab >= 0 && !row_is_finite(data, i, n_features) {
            return Err(ClustorError::InvalidArg(
                "X must contain only finite values for non-noise samples".into(),
            ));
        }
    }
    Ok(())
}

#[inline]
fn row_is_finite(data: &[f64], i: usize, n_features: usize) -> bool {
    data[i * n_features..(i + 1) * n_features]
        .iter()
        .all(|v| v.is_finite())
}

#[inline]
fn dist(
    a: &[f64],
    an: Option<f64>,
    b: &[f64],
    bn: Option<f64>,
    metric: Metric,
    euclidean_inv_scale: f64,
) -> f64 {
    match metric {
        Metric::Euclidean => euclidean_distance_scaled(a, b, euclidean_inv_scale),
        Metric::Cosine => cosine_distance(a, an.unwrap(), b, bn.unwrap()),
    }
}

#[inline]
fn euclidean_distance_scaled(a: &[f64], b: &[f64], inv_scale: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] * inv_scale - b[i] * inv_scale;
        s += d * d;
    }
    s.sqrt()
}

#[inline]
fn non_noise_inv_scale_from_labels(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
) -> f64 {
    let mut max_abs = 0.0;
    for (i, &lab) in labels.iter().take(n_samples).enumerate() {
        if lab < 0 {
            continue;
        }
        let row = &data[i * n_features..(i + 1) * n_features];
        for &v in row {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    if max_abs > 0.0 { 1.0 / max_abs } else { 1.0 }
}

/// Silhouette score (mean over samples). Noise label -1 is ignored.
/// O(n^2) implementation intended for moderate n.
pub fn silhouette_score(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
    metric: Metric,
) -> ClustorResult<f64> {
    validate(data, n_samples, n_features, labels)?;

    use std::collections::HashMap;
    let mut label_to_cluster: HashMap<i64, usize> = HashMap::new();
    let mut cluster_sizes: Vec<usize> = Vec::new();
    let mut sample_cluster = vec![usize::MAX; n_samples];
    let mut non_noise_indices: Vec<usize> = Vec::new();
    let mut euclidean_max_abs = 0.0;
    for (i, &lab) in labels.iter().enumerate() {
        if lab < 0 {
            continue;
        }
        let idx = match label_to_cluster.get(&lab) {
            Some(&idx) => idx,
            None => {
                let idx = cluster_sizes.len();
                label_to_cluster.insert(lab, idx);
                cluster_sizes.push(0);
                idx
            }
        };
        cluster_sizes[idx] += 1;
        sample_cluster[i] = idx;
        non_noise_indices.push(i);
        if metric == Metric::Euclidean {
            let row = &data[i * n_features..(i + 1) * n_features];
            for &v in row {
                let a = v.abs();
                if a > euclidean_max_abs {
                    euclidean_max_abs = a;
                }
            }
        }
    }
    let k = cluster_sizes.len();
    if k <= 1 {
        return Err(ClustorError::InvalidArg(
            "Need at least 2 clusters (excluding noise) for silhouette_score".into(),
        ));
    }
    let norms = if metric == Metric::Cosine {
        Some(compute_row_norms(data, n_samples, n_features))
    } else {
        None
    };
    let euclidean_inv_scale = if metric == Metric::Euclidean && euclidean_max_abs > 0.0 {
        1.0 / euclidean_max_abs
    } else {
        1.0
    };

    let m = non_noise_indices.len();
    let sum_len = m
        .checked_mul(k)
        .ok_or_else(|| ClustorError::InvalidArg("Input too large".into()))?;
    let mut sums = vec![0.0; sum_len];

    for pos_i in 0..m {
        let i = non_noise_indices[pos_i];
        let ci = sample_cluster[i];
        debug_assert_ne!(ci, usize::MAX);
        let xi = &data[i * n_features..(i + 1) * n_features];
        let ni = norms.as_ref().map(|v| v[i]);
        for pos_j in (pos_i + 1)..m {
            let j = non_noise_indices[pos_j];
            let cj = sample_cluster[j];
            debug_assert_ne!(cj, usize::MAX);
            let xj = &data[j * n_features..(j + 1) * n_features];
            let nj = norms.as_ref().map(|v| v[j]);
            let d = dist(xi, ni, xj, nj, metric, euclidean_inv_scale);
            sums[pos_i * k + cj] += d;
            sums[pos_j * k + ci] += d;
        }
    }

    let mut total = 0.0;

    for (pos, &i) in non_noise_indices.iter().enumerate() {
        let ci = sample_cluster[i];
        debug_assert_ne!(ci, usize::MAX);
        let size = cluster_sizes[ci];
        if size <= 1 {
            continue;
        }

        let a = sums[pos * k + ci] / ((size - 1) as f64);
        let mut b = f64::INFINITY;
        for c in 0..k {
            if c == ci {
                continue;
            }
            let mean = sums[pos * k + c] / (cluster_sizes[c] as f64);
            if mean < b {
                b = mean;
            }
        }
        let denom = a.max(b).max(1e-12);
        total += (b - a) / denom;
    }

    Ok(total / (m as f64))
}

/// Calinski-Harabasz index (higher is better). Noise (-1) ignored.
pub fn calinski_harabasz_score(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
) -> ClustorResult<f64> {
    validate(data, n_samples, n_features, labels)?;
    use std::collections::BTreeMap;

    let mut clusters: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &lab) in labels.iter().enumerate() {
        if lab >= 0 {
            clusters.entry(lab).or_default().push(i);
        }
    }
    let k = clusters.len();
    if k <= 1 {
        return Err(ClustorError::InvalidArg(
            "Need at least 2 clusters for calinski_harabasz_score".into(),
        ));
    }

    let n = clusters.values().map(|v| v.len()).sum::<usize>();
    if n <= k {
        return Err(ClustorError::InvalidArg("Not enough samples".into()));
    }

    let inv_scale = non_noise_inv_scale_from_labels(data, n_samples, n_features, labels);

    let mut overall = vec![0.0; n_features];
    for i in 0..n_samples {
        if labels[i] < 0 {
            continue;
        }
        let x = &data[i * n_features..(i + 1) * n_features];
        for (overall_j, x_j) in overall.iter_mut().zip(x.iter()) {
            *overall_j += x_j;
        }
    }
    for overall_j in overall.iter_mut() {
        *overall_j /= n as f64;
    }

    let mut b = 0.0;
    let mut w = 0.0;
    for (_lab, idxs) in clusters.iter() {
        let mut mu = vec![0.0; n_features];
        for &i in idxs.iter() {
            let x = &data[i * n_features..(i + 1) * n_features];
            for (mu_j, x_j) in mu.iter_mut().zip(x.iter()) {
                *mu_j += x_j;
            }
        }
        for mu_j in mu.iter_mut() {
            *mu_j /= idxs.len() as f64;
        }
        let mut dist2 = 0.0;
        for j in 0..n_features {
            let d = (mu[j] - overall[j]) * inv_scale;
            dist2 += d * d;
        }
        b += (idxs.len() as f64) * dist2;
        for &i in idxs.iter() {
            let x = &data[i * n_features..(i + 1) * n_features];
            let mut s = 0.0;
            for j in 0..n_features {
                let d = (x[j] - mu[j]) * inv_scale;
                s += d * d;
            }
            w += s;
        }
    }
    if w == 0.0 {
        if b == 0.0 {
            return Err(ClustorError::InvalidArg(
                "Calinski-Harabasz score undefined for zero dispersion".into(),
            ));
        }
        return Ok(f64::INFINITY);
    }
    let kf = k as f64;
    let nf = n as f64;
    Ok((b / (kf - 1.0)) / (w / (nf - kf)))
}

/// Davies-Bouldin index (lower is better). Noise (-1) ignored.
pub fn davies_bouldin_score(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    labels: &[i64],
) -> ClustorResult<f64> {
    validate(data, n_samples, n_features, labels)?;
    use std::collections::BTreeMap;

    let mut clusters: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, &lab) in labels.iter().enumerate() {
        if lab >= 0 {
            clusters.entry(lab).or_default().push(i);
        }
    }
    let k = clusters.len();
    if k <= 1 {
        return Err(ClustorError::InvalidArg(
            "Need at least 2 clusters for davies_bouldin_score".into(),
        ));
    }

    let inv_scale = non_noise_inv_scale_from_labels(data, n_samples, n_features, labels);

    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut scatters: Vec<f64> = Vec::with_capacity(k);

    for (_lab, idxs) in clusters.iter() {
        let mut mu = vec![0.0; n_features];
        for &i in idxs.iter() {
            let x = &data[i * n_features..(i + 1) * n_features];
            for (mu_j, x_j) in mu.iter_mut().zip(x.iter()) {
                *mu_j += x_j;
            }
        }
        for mu_j in mu.iter_mut() {
            *mu_j /= idxs.len() as f64;
        }
        let mut s = 0.0;
        for &i in idxs.iter() {
            let x = &data[i * n_features..(i + 1) * n_features];
            let mut d2 = 0.0;
            for j in 0..n_features {
                let d = (x[j] - mu[j]) * inv_scale;
                d2 += d * d;
            }
            s += d2.sqrt();
        }
        s /= idxs.len() as f64;
        centroids.push(mu);
        scatters.push(s);
    }

    let mut sum_r = 0.0;
    for i in 0..k {
        let mut max_r = f64::NEG_INFINITY;
        for j in 0..k {
            if i == j {
                continue;
            }
            let mut d2 = 0.0;
            for (&ci, &cj) in centroids[i].iter().zip(centroids[j].iter()) {
                let d = (ci - cj) * inv_scale;
                d2 += d * d;
            }
            let d = d2.sqrt().max(1e-12);
            let r = (scatters[i] + scatters[j]) / d;
            if r > max_r {
                max_r = r;
            }
        }
        sum_r += max_r;
    }
    Ok(sum_r / (k as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_with_non_finite_noise() -> (Vec<f64>, Vec<i64>) {
        (
            vec![
                f64::NAN,
                f64::INFINITY,
                0.0,
                0.0,
                0.0,
                1.0,
                5.0,
                0.0,
                5.0,
                1.0,
            ],
            vec![-1, 0, 0, 1, 1],
        )
    }

    fn assert_invalid_arg(res: ClustorResult<f64>, expected_substr: &str) {
        match res {
            Err(ClustorError::InvalidArg(msg)) => assert!(msg.contains(expected_substr)),
            other => panic!("expected InvalidArg, got {other:?}"),
        }
    }

    #[test]
    fn validation_metrics_ignore_non_finite_noise_rows() {
        let (data, labels) = data_with_non_finite_noise();

        let sil = silhouette_score(&data, 5, 2, &labels, Metric::Euclidean).unwrap();
        let ch = calinski_harabasz_score(&data, 5, 2, &labels).unwrap();
        let db = davies_bouldin_score(&data, 5, 2, &labels).unwrap();

        assert!(sil.is_finite());
        assert!((-1.0..=1.0).contains(&sil));
        assert!(ch.is_finite());
        assert!(db.is_finite());
    }

    #[test]
    fn validation_metrics_reject_non_finite_non_noise_rows() {
        let data = vec![0.0, 0.0, f64::NAN, 1.0, 2.0, f64::INFINITY];
        let labels = vec![0, 0, 1];

        assert_invalid_arg(
            silhouette_score(&data, 3, 2, &labels, Metric::Euclidean),
            "finite",
        );
        assert_invalid_arg(calinski_harabasz_score(&data, 3, 2, &labels), "finite");
        assert_invalid_arg(davies_bouldin_score(&data, 3, 2, &labels), "finite");
    }

    #[test]
    fn validation_rejects_invalid_negative_labels() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![-2, 0];

        assert_invalid_arg(
            silhouette_score(&data, 2, 2, &labels, Metric::Euclidean),
            "labels must be >= -1",
        );
        assert_invalid_arg(
            calinski_harabasz_score(&data, 2, 2, &labels),
            "labels must be >= -1",
        );
        assert_invalid_arg(
            davies_bouldin_score(&data, 2, 2, &labels),
            "labels must be >= -1",
        );
    }

    #[test]
    fn validation_rejects_overflowing_shape_product() {
        assert_invalid_arg(
            silhouette_score(&[], usize::MAX, 2, &[], Metric::Euclidean),
            "Input too large",
        );
        assert_invalid_arg(
            calinski_harabasz_score(&[], usize::MAX, 2, &[]),
            "Input too large",
        );
        assert_invalid_arg(
            davies_bouldin_score(&[], usize::MAX, 2, &[]),
            "Input too large",
        );
    }

    #[test]
    fn validation_metrics_handle_large_finite_values_without_overflow() {
        let data = vec![0.0, 0.0, 1e308, 1e308, -1e308, -1e308];
        let labels = vec![0, 1, 1];

        let sil = silhouette_score(&data, 3, 2, &labels, Metric::Euclidean).unwrap();
        let db = davies_bouldin_score(&data, 3, 2, &labels).unwrap();
        let ch = calinski_harabasz_score(&data, 3, 2, &labels).unwrap();

        assert!(sil.is_finite());
        assert!((-1.0..=1.0).contains(&sil));
        assert!(db.is_finite());
        assert!(ch.is_finite());
    }
}

// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::errors::{ClustorError, ClustorResult};
use crate::kmeans::{KMeansParams, fit_kmeans};
use crate::metrics::{Metric, cosine_distance, euclidean_sq, l2_norm, normalize_in_place};
use crate::utils::compute_row_norms;

#[derive(Clone, Debug)]
pub struct BisectingParams {
    pub n_clusters: usize,
    pub n_init: usize,
    pub max_iter: u32,
    pub tol: f64,
    pub metric: Metric,
    pub normalize_input: bool,
    pub normalize_centers: bool,
    pub random_state: Option<u64>,
    pub verbose: bool,
}

#[derive(Clone, Debug)]
pub struct BisectingOutput {
    pub centers: Vec<f64>,  // k * n_features
    pub labels: Vec<usize>, // n_samples
    pub inertia: f64,
    pub n_splits: u32,
}

fn validate_inputs(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    k: usize,
) -> ClustorResult<()> {
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
    if k == 0 {
        return Err(ClustorError::InvalidArg("n_clusters must be > 0".into()));
    }
    if k > n_samples {
        return Err(ClustorError::InvalidArg(
            "n_clusters cannot exceed n_samples".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_sse_for_subset(
    data: &[f64],
    data_norms: Option<&[f64]>,
    centers: &[f64], // 2 * n_features
    center_norms: Option<&[f64]>,
    labels: &[usize],
    sample_weight: Option<&[f64]>,
    n_samples: usize,
    n_features: usize,
    metric: Metric,
) -> (f64, f64) {
    let mut sse0 = 0.0;
    let mut sse1 = 0.0;
    for i in 0..n_samples {
        let x = &data[i * n_features..(i + 1) * n_features];
        let lab = labels[i];
        let cent = &centers[lab * n_features..(lab + 1) * n_features];
        let d2 = match metric {
            Metric::Euclidean => euclidean_sq(x, cent),
            Metric::Cosine => {
                let xn = data_norms.unwrap()[i];
                let cn = center_norms.unwrap()[lab];
                let cd = cosine_distance(x, xn, cent, cn);
                cd * cd
            }
        };
        let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
        if lab == 0 {
            sse0 += d2 * w;
        } else {
            sse1 += d2 * w;
        }
    }
    (sse0, sse1)
}

pub fn fit_bisecting_kmeans(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    sample_weight: Option<&[f64]>,
    params: &BisectingParams,
) -> ClustorResult<BisectingOutput> {
    validate_inputs(data_in, n_samples, n_features, params.n_clusters)?;
    if params.tol < 0.0 {
        return Err(ClustorError::InvalidArg("tol must be >= 0".into()));
    }
    if params.n_init == 0 {
        return Err(ClustorError::InvalidArg("n_init must be >= 1".into()));
    }
    if let Some(w) = sample_weight {
        if w.len() != n_samples {
            return Err(ClustorError::InvalidArg(
                "sample_weight length must equal n_samples".into(),
            ));
        }
        for &wi in w.iter() {
            if !wi.is_finite() || wi < 0.0 {
                return Err(ClustorError::InvalidArg(
                    "sample_weight must be finite and >= 0".into(),
                ));
            }
        }
    }

    // Copy + optional normalization
    let mut data = data_in.to_vec();
    let mut data_norms: Option<Vec<f64>> = None;

    if params.metric == Metric::Cosine {
        if params.normalize_input {
            for i in 0..n_samples {
                let row = &mut data[i * n_features..(i + 1) * n_features];
                normalize_in_place(row);
            }
        }
        data_norms = Some(compute_row_norms(&data, n_samples, n_features));
    }

    let seed = params
        .random_state
        .unwrap_or(0xC0FFEE_u64 ^ (n_samples as u64) ^ ((n_features as u64) << 32));
    let mut rng = StdRng::seed_from_u64(seed);

    // Cluster containers
    let mut labels = vec![0usize; n_samples];
    let mut clusters: Vec<Vec<usize>> = vec![(0..n_samples).collect()];
    let mut centers: Vec<Vec<f64>> = vec![vec![0.0; n_features]];
    let mut inertias: Vec<f64> = vec![0.0];

    // Initialize center/inertia for the full cluster via a 1-cluster mean
    {
        let idxs = &clusters[0];
        let mut c = vec![0.0; n_features];
        let mut wsum = 0.0;
        for &i in idxs.iter() {
            let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
            wsum += w;
            let x = &data[i * n_features..(i + 1) * n_features];
            for j in 0..n_features {
                c[j] += w * x[j];
            }
        }
        if wsum > 0.0 {
            for c_j in c.iter_mut() {
                *c_j /= wsum;
            }
        }
        if params.metric == Metric::Cosine && params.normalize_centers {
            normalize_in_place(&mut c);
        }
        centers[0] = c;

        // SSE
        let mut sse = 0.0;
        for &i in idxs.iter() {
            let x = &data[i * n_features..(i + 1) * n_features];
            let d2 = match params.metric {
                Metric::Euclidean => euclidean_sq(x, &centers[0]),
                Metric::Cosine => {
                    let xn = data_norms.as_ref().unwrap()[i];
                    let cn = l2_norm(&centers[0]);
                    let cd = cosine_distance(x, xn, &centers[0], cn);
                    cd * cd
                }
            };
            let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
            sse += d2 * w;
        }
        inertias[0] = sse;
    }

    let mut splits: u32 = 0;

    while clusters.len() < params.n_clusters {
        // choose cluster with largest inertia (and at least 2 points)
        let mut split_idx: Option<usize> = None;
        let mut best_inertia = -1.0;
        for (ci, idxs) in clusters.iter().enumerate() {
            if idxs.len() < 2 {
                continue;
            }
            let sse = inertias[ci];
            if sse > best_inertia {
                best_inertia = sse;
                split_idx = Some(ci);
            }
        }
        let split_idx = match split_idx {
            Some(v) => v,
            None => break,
        };

        let idxs = clusters[split_idx].clone();
        let m = idxs.len();
        if m < 2 {
            break;
        }

        // Build subset
        let mut sub = vec![0.0; m * n_features];
        let mut sub_w: Option<Vec<f64>> = sample_weight.map(|_| vec![0.0; m]);
        for (r, &i) in idxs.iter().enumerate() {
            sub[r * n_features..(r + 1) * n_features]
                .copy_from_slice(&data[i * n_features..(i + 1) * n_features]);
            if let Some(ref mut wv) = sub_w {
                wv[r] = sample_weight.unwrap()[i];
            }
        }

        let kparams = KMeansParams {
            n_clusters: 2,
            n_init: params.n_init,
            max_iter: params.max_iter,
            tol: params.tol,
            metric: params.metric,
            normalize_input: false, // already normalized above if needed
            normalize_centers: params.normalize_centers,
            random_state: Some(rng.random::<u64>()),
            verbose: params.verbose,
        };

        let out = fit_kmeans(&sub, m, n_features, sub_w.as_deref(), &kparams)?;

        // split indices
        let mut left: Vec<usize> = Vec::with_capacity(m);
        let mut right: Vec<usize> = Vec::with_capacity(m);
        for (r, &i) in idxs.iter().enumerate() {
            if out.labels[r] == 0 {
                left.push(i);
            } else {
                right.push(i);
            }
        }

        // Guard against degeneracy: fallback to random half split
        if left.is_empty() || right.is_empty() {
            let mut shuffled = idxs.clone();
            shuffled.shuffle(&mut rng);
            let mid = shuffled.len() / 2;
            left = shuffled[..mid].to_vec();
            right = shuffled[mid..].to_vec();
        }

        // Prepare per-split SSEs computed on subset with out.centers
        let mut sub_norms: Option<Vec<f64>> = None;
        let mut center_norms: Option<Vec<f64>> = None;
        if params.metric == Metric::Cosine {
            sub_norms = Some(compute_row_norms(&sub, m, n_features));
            let c0 = &out.centers[0..n_features];
            let c1 = &out.centers[n_features..2 * n_features];
            center_norms = Some(vec![l2_norm(c0), l2_norm(c1)]);
        }
        let (sse0, sse1) = compute_sse_for_subset(
            &sub,
            sub_norms.as_deref(),
            &out.centers,
            center_norms.as_deref(),
            &out.labels,
            sub_w.as_deref(),
            m,
            n_features,
            params.metric,
        );

        // Update global labels: old id for left, new id for right
        let new_id = clusters.len();
        for &i in left.iter() {
            labels[i] = split_idx;
        }
        for &i in right.iter() {
            labels[i] = new_id;
        }

        clusters[split_idx] = left;
        centers[split_idx] = out.centers[0..n_features].to_vec();
        inertias[split_idx] = sse0;

        clusters.push(right);
        centers.push(out.centers[n_features..2 * n_features].to_vec());
        inertias.push(sse1);

        splits += 1;

        if params.verbose {
            eprintln!("Split {} -> now k={}", split_idx, clusters.len());
        }
    }

    let total_inertia: f64 = inertias.iter().sum();
    let mut flat_centers = Vec::with_capacity(centers.len() * n_features);
    for c in centers.iter() {
        flat_centers.extend_from_slice(c);
    }

    Ok(BisectingOutput {
        centers: flat_centers,
        labels,
        inertia: total_inertia,
        n_splits: splits,
    })
}

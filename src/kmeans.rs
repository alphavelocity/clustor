// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance, euclidean_sq, l2_norm, normalize_in_place};
use crate::utils::{compute_row_norms, kmeans_plus_plus, pick_random_index};

#[derive(Clone, Debug)]
pub struct FitOutput {
    pub centers: Vec<f64>,  // k * n_features
    pub labels: Vec<usize>, // n_samples
    pub inertia: f64,
    pub n_iter: u32,
}

#[derive(Clone, Debug)]
pub struct KMeansParams {
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

fn maybe_normalize_input(data: &mut [f64], n_samples: usize, n_features: usize) {
    for i in 0..n_samples {
        let row = &mut data[i * n_features..(i + 1) * n_features];
        normalize_in_place(row);
    }
}

#[allow(clippy::too_many_arguments)]
fn assign_labels(
    data: &[f64],
    data_norms: Option<&[f64]>,
    centers: &[f64],
    center_norms: Option<&[f64]>,
    sample_weight: Option<&[f64]>,
    n_samples: usize,
    n_features: usize,
    k: usize,
    metric: Metric,
    labels: &mut [usize],
) -> f64 {
    let mut inertia = 0.0;
    for i in 0..n_samples {
        let x = &data[i * n_features..(i + 1) * n_features];
        let mut best_k = 0usize;
        let mut best_d = f64::INFINITY;
        for c in 0..k {
            let cent = &centers[c * n_features..(c + 1) * n_features];
            let d = match metric {
                Metric::Euclidean => euclidean_sq(x, cent),
                Metric::Cosine => {
                    let xn = data_norms.unwrap()[i];
                    let cn = center_norms.unwrap()[c];
                    let cd = cosine_distance(x, xn, cent, cn);
                    cd * cd
                }
            };
            if d < best_d {
                best_d = d;
                best_k = c;
            }
        }
        labels[i] = best_k;
        let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
        inertia += best_d * w;
    }
    inertia
}

#[allow(clippy::too_many_arguments)]
fn recompute_centers(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    k: usize,
    labels: &[usize],
    sample_weight: Option<&[f64]>,
    rng: &mut StdRng,
    metric: Metric,
    normalize_centers_cosine: bool,
) -> Vec<f64> {
    let mut centers = vec![0.0; k * n_features];
    let mut weight_sums = vec![0.0; k];

    for i in 0..n_samples {
        let c = labels[i];
        let w = sample_weight.map(|sw| sw[i]).unwrap_or(1.0);
        weight_sums[c] += w;
        let x = &data[i * n_features..(i + 1) * n_features];
        let cent = &mut centers[c * n_features..(c + 1) * n_features];
        for (j, cent_j) in cent.iter_mut().enumerate().take(n_features) {
            *cent_j += w * x[j];
        }
    }

    for c in 0..k {
        if weight_sums[c] <= 0.0 {
            // empty or zero-weight cluster: re-seed randomly
            let idx = pick_random_index(rng, n_samples);
            centers[c * n_features..(c + 1) * n_features]
                .copy_from_slice(&data[idx * n_features..(idx + 1) * n_features]);
        } else {
            let inv = 1.0 / weight_sums[c];
            let cent = &mut centers[c * n_features..(c + 1) * n_features];
            for cent_j in cent.iter_mut() {
                *cent_j *= inv;
            }
        }

        if metric == Metric::Cosine && normalize_centers_cosine {
            normalize_in_place(&mut centers[c * n_features..(c + 1) * n_features]);
        }
    }
    centers
}

fn max_center_shift(old: &[f64], new: &[f64], k: usize, n_features: usize) -> f64 {
    let mut best = 0.0;
    for c in 0..k {
        let a = &old[c * n_features..(c + 1) * n_features];
        let b = &new[c * n_features..(c + 1) * n_features];
        let mut s = 0.0;
        for j in 0..n_features {
            let d = a[j] - b[j];
            s += d * d;
        }
        let shift = s.sqrt();
        if shift > best {
            best = shift;
        }
    }
    best
}

pub fn fit_kmeans(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    sample_weight: Option<&[f64]>,
    params: &KMeansParams,
) -> ClustorResult<FitOutput> {
    validate_inputs(data_in, n_samples, n_features, params.n_clusters)?;
    if params.n_init == 0 {
        return Err(ClustorError::InvalidArg("n_init must be > 0".into()));
    }
    if params.tol < 0.0 {
        return Err(ClustorError::InvalidArg("tol must be >= 0".into()));
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

    let mut data = data_in.to_vec();
    if params.metric == Metric::Cosine && params.normalize_input {
        maybe_normalize_input(&mut data, n_samples, n_features);
    }

    let data_norms = if params.metric == Metric::Cosine {
        Some(compute_row_norms(&data, n_samples, n_features))
    } else {
        None
    };

    let mut best_out: Option<FitOutput> = None;

    for init_idx in 0..params.n_init {
        let seed = params
            .random_state
            .unwrap_or_else(|| rand::rng().random::<u64>())
            .wrapping_add(init_idx as u64 * 977);
        let mut rng = StdRng::seed_from_u64(seed);

        let centers0 = kmeans_plus_plus(
            &mut rng,
            &data,
            n_samples,
            n_features,
            params.n_clusters,
            params.metric,
            data_norms.as_deref(),
            params.metric == Metric::Cosine && params.normalize_centers,
        );

        let mut centers = centers0;
        let mut labels = vec![0usize; n_samples];

        let mut center_norms = if params.metric == Metric::Cosine {
            let mut cn = Vec::with_capacity(params.n_clusters);
            for c in 0..params.n_clusters {
                cn.push(l2_norm(&centers[c * n_features..(c + 1) * n_features]));
            }
            Some(cn)
        } else {
            None
        };

        let mut inertia = assign_labels(
            &data,
            data_norms.as_deref(),
            &centers,
            center_norms.as_deref(),
            sample_weight,
            n_samples,
            n_features,
            params.n_clusters,
            params.metric,
            &mut labels,
        );

        let mut n_iter = 0u32;
        for it in 0..params.max_iter {
            n_iter = it + 1;
            let old_centers = centers.clone();
            centers = recompute_centers(
                &data,
                n_samples,
                n_features,
                params.n_clusters,
                &labels,
                sample_weight,
                &mut rng,
                params.metric,
                params.metric == Metric::Cosine && params.normalize_centers,
            );

            if params.metric == Metric::Cosine {
                let mut cn = vec![0.0; params.n_clusters];
                for c in 0..params.n_clusters {
                    cn[c] = l2_norm(&centers[c * n_features..(c + 1) * n_features]);
                }
                center_norms = Some(cn);
            }

            inertia = assign_labels(
                &data,
                data_norms.as_deref(),
                &centers,
                center_norms.as_deref(),
                sample_weight,
                n_samples,
                n_features,
                params.n_clusters,
                params.metric,
                &mut labels,
            );

            let shift = max_center_shift(&old_centers, &centers, params.n_clusters, n_features);
            if params.verbose {
                eprintln!(
                    "[Clustor][KMeans] init={} iter={} inertia={} shift={}",
                    init_idx, n_iter, inertia, shift
                );
            }
            if shift <= params.tol {
                break;
            }
        }

        let out = FitOutput {
            centers,
            labels,
            inertia,
            n_iter,
        };
        let take = match &best_out {
            None => true,
            Some(b) => out.inertia < b.inertia,
        };
        if take {
            best_out = Some(out);
        }
    }

    Ok(best_out.unwrap())
}

// ---------------- MiniBatch KMeans ----------------

#[derive(Clone, Debug)]
pub struct MiniBatchParams {
    pub n_clusters: usize,
    pub batch_size: usize,
    pub max_steps: u32,
    pub metric: Metric,
    pub normalize_input: bool,
    pub normalize_centers: bool,
    pub random_state: Option<u64>,
    pub verbose: bool,
}

#[derive(Clone, Debug)]
pub struct MiniBatchState {
    pub centers: Vec<f64>,
    pub counts: Vec<u64>,
    pub n_features: usize,
    pub metric: Metric,
    pub normalize_centers: bool,
}

fn minibatch_validate(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    k: usize,
    batch_size: usize,
) -> ClustorResult<()> {
    validate_inputs(data, n_samples, n_features, k)?;
    if batch_size == 0 {
        return Err(ClustorError::InvalidArg("batch_size must be > 0".into()));
    }
    Ok(())
}

pub fn minibatch_init(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &MiniBatchParams,
) -> ClustorResult<MiniBatchState> {
    minibatch_validate(
        data_in,
        n_samples,
        n_features,
        params.n_clusters,
        params.batch_size,
    )?;
    let mut data = data_in.to_vec();
    if params.metric == Metric::Cosine && params.normalize_input {
        maybe_normalize_input(&mut data, n_samples, n_features);
    }
    let data_norms = if params.metric == Metric::Cosine {
        Some(compute_row_norms(&data, n_samples, n_features))
    } else {
        None
    };
    let seed = params
        .random_state
        .unwrap_or_else(|| rand::rng().random::<u64>());
    let mut rng = StdRng::seed_from_u64(seed);
    let centers = kmeans_plus_plus(
        &mut rng,
        &data,
        n_samples,
        n_features,
        params.n_clusters,
        params.metric,
        data_norms.as_deref(),
        params.metric == Metric::Cosine && params.normalize_centers,
    );
    Ok(MiniBatchState {
        centers,
        counts: vec![0u64; params.n_clusters],
        n_features,
        metric: params.metric,
        normalize_centers: params.metric == Metric::Cosine && params.normalize_centers,
    })
}

pub fn minibatch_partial_fit(
    state: &mut MiniBatchState,
    batch_in: &[f64],
    n_samples: usize,
    n_features: usize,
    verbose: bool,
) -> ClustorResult<()> {
    if state.centers.is_empty() {
        return Err(ClustorError::InvalidArg(
            "MiniBatchState not initialized; call fit() first".into(),
        ));
    }
    if n_features != state.n_features {
        return Err(ClustorError::InvalidArg(
            "feature dimension mismatch".into(),
        ));
    }
    if batch_in.len() != n_samples * n_features {
        return Err(ClustorError::InvalidArg("batch length mismatch".into()));
    }

    let mut center_norms: Option<Vec<f64>> = None;
    if state.metric == Metric::Cosine {
        let mut cn = Vec::with_capacity(state.counts.len());
        for c in 0..state.counts.len() {
            cn.push(l2_norm(
                &state.centers[c * n_features..(c + 1) * n_features],
            ));
        }
        center_norms = Some(cn);
    }

    let x_norms = if state.metric == Metric::Cosine {
        Some(compute_row_norms(batch_in, n_samples, n_features))
    } else {
        None
    };

    for i in 0..n_samples {
        let x = &batch_in[i * n_features..(i + 1) * n_features];
        let mut best_k = 0usize;
        let mut best_d = f64::INFINITY;
        for c in 0..state.counts.len() {
            let cent = &state.centers[c * n_features..(c + 1) * n_features];
            let d = match state.metric {
                Metric::Euclidean => euclidean_sq(x, cent),
                Metric::Cosine => {
                    let xn = x_norms.as_ref().unwrap()[i];
                    let cn = center_norms.as_ref().unwrap()[c];
                    let cd = cosine_distance(x, xn, cent, cn);
                    cd * cd
                }
            };
            if d < best_d {
                best_d = d;
                best_k = c;
            }
        }

        state.counts[best_k] += 1;
        let eta = 1.0 / (state.counts[best_k] as f64);
        let cent = &mut state.centers[best_k * n_features..(best_k + 1) * n_features];
        for j in 0..n_features {
            cent[j] = (1.0 - eta) * cent[j] + eta * x[j];
        }
        if state.metric == Metric::Cosine && state.normalize_centers {
            normalize_in_place(cent);
        }
    }

    if verbose {
        eprintln!("[Clustor][MiniBatch] partial_fit batch_n={}", n_samples);
    }

    Ok(())
}

pub fn minibatch_fit(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &MiniBatchParams,
) -> ClustorResult<(MiniBatchState, FitOutput)> {
    minibatch_validate(
        data_in,
        n_samples,
        n_features,
        params.n_clusters,
        params.batch_size,
    )?;
    let mut state = minibatch_init(data_in, n_samples, n_features, params)?;

    let seed = params
        .random_state
        .unwrap_or_else(|| rand::rng().random::<u64>());
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(13));

    let mut data = data_in.to_vec();
    if params.metric == Metric::Cosine && params.normalize_input {
        maybe_normalize_input(&mut data, n_samples, n_features);
    }

    for step in 1..=params.max_steps {
        let bs = params.batch_size.min(n_samples);
        let mut idxs: Vec<usize> = (0..n_samples).collect();
        idxs.shuffle(&mut rng);
        idxs.truncate(bs);

        let mut batch = vec![0.0; bs * n_features];
        for (bi, &si) in idxs.iter().enumerate() {
            batch[bi * n_features..(bi + 1) * n_features]
                .copy_from_slice(&data[si * n_features..(si + 1) * n_features]);
        }

        minibatch_partial_fit(&mut state, &batch, bs, n_features, params.verbose)?;

        if params.verbose && step % 100 == 0 {
            eprintln!("[Clustor][MiniBatch] step={}", step);
        }
    }

    // Final labels/inertia on full data
    let data_norms = if params.metric == Metric::Cosine {
        Some(compute_row_norms(&data, n_samples, n_features))
    } else {
        None
    };
    let center_norms = if params.metric == Metric::Cosine {
        let mut cn = Vec::with_capacity(params.n_clusters);
        for c in 0..params.n_clusters {
            cn.push(l2_norm(
                &state.centers[c * n_features..(c + 1) * n_features],
            ));
        }
        Some(cn)
    } else {
        None
    };

    let mut labels = vec![0usize; n_samples];
    let inertia = assign_labels(
        &data,
        data_norms.as_deref(),
        &state.centers,
        center_norms.as_deref(),
        None,
        n_samples,
        n_features,
        params.n_clusters,
        params.metric,
        &mut labels,
    );
    let out = FitOutput {
        centers: state.centers.clone(),
        labels,
        inertia,
        n_iter: params.max_steps,
    };
    Ok((state, out))
}

// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance, euclidean_sq, normalize_in_place};
use crate::utils::compute_row_norms;

#[derive(Clone, Debug)]
pub struct AffinityParams {
    pub damping: f64,
    pub max_iter: u32,
    pub convergence_iter: u32,
    pub preference: Option<f64>,
    pub metric: Metric,
    pub normalize_input: bool, // for cosine
    pub verbose: bool,
}

#[derive(Clone, Debug)]
pub struct AffinityOutput {
    pub exemplars: Vec<usize>,
    pub labels: Vec<usize>,
    pub converged: bool,
    pub n_iter: u32,
}

fn validate_inputs(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &AffinityParams,
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
    if !(0.5..1.0).contains(&params.damping) {
        return Err(ClustorError::InvalidArg(
            "damping must be in [0.5, 1.0)".into(),
        ));
    }
    if params.max_iter == 0 {
        return Err(ClustorError::InvalidArg("max_iter must be > 0".into()));
    }
    if params.convergence_iter == 0 {
        return Err(ClustorError::InvalidArg(
            "convergence_iter must be > 0".into(),
        ));
    }
    Ok(())
}

#[inline]
fn dist_sq(
    data: &[f64],
    norms: Option<&[f64]>,
    i: usize,
    k: usize,
    n_features: usize,
    metric: Metric,
) -> f64 {
    let a = &data[i * n_features..(i + 1) * n_features];
    let b = &data[k * n_features..(k + 1) * n_features];
    match metric {
        Metric::Euclidean => euclidean_sq(a, b),
        Metric::Cosine => {
            let n = norms.expect("norms required for cosine");
            let d = cosine_distance(a, n[i], b, n[k]).max(0.0);
            d * d
        }
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

pub fn fit_affinity_propagation(
    data_in: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &AffinityParams,
) -> ClustorResult<AffinityOutput> {
    validate_inputs(data_in, n_samples, n_features, params)?;
    if params.verbose {
        eprintln!(
            "[Clustor][Affinity] n_samples={}, n_features={}",
            n_samples, n_features
        );
    }

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

    // Similarities S(i,k) = -dist^2
    let mut s = vec![0.0f64; n_samples * n_samples];
    let mut s_offdiag = Vec::with_capacity(n_samples * (n_samples - 1));
    for i in 0..n_samples {
        for k in 0..n_samples {
            let val = if i == k {
                0.0
            } else {
                -dist_sq(data_ref, norms_ref, i, k, n_features, params.metric)
            };
            s[i * n_samples + k] = val;
            if i != k {
                s_offdiag.push(val);
            }
        }
    }

    let pref = params.preference.unwrap_or_else(|| median(s_offdiag));
    for k in 0..n_samples {
        s[k * n_samples + k] = pref;
    }

    let mut a = vec![0.0f64; n_samples * n_samples];
    let mut r = vec![0.0f64; n_samples * n_samples];

    let damp = params.damping;
    let max_iter = params.max_iter as usize;
    let conv_iter = params.convergence_iter as usize;

    let mut e_prev = vec![0usize; n_samples];
    let mut stable = 0usize;

    let mut converged = false;
    let mut it_done = 0u32;

    for it in 0..max_iter {
        // Update responsibilities
        for i in 0..n_samples {
            // find largest and second largest of a+s
            let mut max1 = f64::NEG_INFINITY;
            let mut max2 = f64::NEG_INFINITY;
            let mut idx1 = 0usize;
            for k in 0..n_samples {
                let v = a[i * n_samples + k] + s[i * n_samples + k];
                if v > max1 {
                    max2 = max1;
                    max1 = v;
                    idx1 = k;
                } else if v > max2 {
                    max2 = v;
                }
            }
            for k in 0..n_samples {
                let v = s[i * n_samples + k] - if k == idx1 { max2 } else { max1 };
                let old = r[i * n_samples + k];
                r[i * n_samples + k] = damp * old + (1.0 - damp) * v;
            }
        }

        // Update availabilities
        for k in 0..n_samples {
            let mut sum_pos = 0.0;
            for i in 0..n_samples {
                if i == k {
                    continue;
                }
                let val = r[i * n_samples + k];
                if val > 0.0 {
                    sum_pos += val;
                }
            }

            for i in 0..n_samples {
                let v = if i == k {
                    sum_pos
                } else {
                    let rik = r[i * n_samples + k].max(0.0);
                    (r[k * n_samples + k] + (sum_pos - rik)).min(0.0)
                };
                let old = a[i * n_samples + k];
                a[i * n_samples + k] = damp * old + (1.0 - damp) * v;
            }
        }

        // Check convergence
        let mut e = vec![0usize; n_samples];
        for i in 0..n_samples {
            let mut best = f64::NEG_INFINITY;
            let mut best_k = 0usize;
            for k in 0..n_samples {
                let v = a[i * n_samples + k] + r[i * n_samples + k];
                if v > best {
                    best = v;
                    best_k = k;
                }
            }
            e[i] = best_k;
        }

        if e == e_prev {
            stable += 1;
        } else {
            stable = 0;
            e_prev.clone_from(&e);
        }

        if stable >= conv_iter {
            converged = true;
            it_done = (it + 1) as u32;
            break;
        }
        it_done = (it + 1) as u32;
    }

    // Determine exemplars: k where a(k,k)+r(k,k)>0
    let mut exemplars = Vec::new();
    for k in 0..n_samples {
        if (a[k * n_samples + k] + r[k * n_samples + k]) > 0.0 {
            exemplars.push(k);
        }
    }
    if exemplars.is_empty() {
        // pick the best diagonal
        let mut best = f64::NEG_INFINITY;
        let mut best_k = 0usize;
        for k in 0..n_samples {
            let v = a[k * n_samples + k] + r[k * n_samples + k];
            if v > best {
                best = v;
                best_k = k;
            }
        }
        exemplars.push(best_k);
    }

    // Assign labels to nearest exemplar based on a+r
    let mut labels = vec![0usize; n_samples];
    for i in 0..n_samples {
        let mut best = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for (ci, &k) in exemplars.iter().enumerate() {
            let v = a[i * n_samples + k] + r[i * n_samples + k];
            if v > best {
                best = v;
                best_idx = ci;
            }
        }
        labels[i] = best_idx;
    }

    Ok(AffinityOutput {
        exemplars,
        labels,
        converged,
        n_iter: it_done,
    })
}

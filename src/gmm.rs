// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::Metric;
use crate::utils::kmeans_plus_plus;

const LOG_2PI: f64 = 1.8378770664093453; // ln(2*pi)

#[derive(Clone, Debug)]
pub struct GmmOutput {
    pub weights: Vec<f64>, // k
    pub means: Vec<f64>,   // k * d
    pub covars: Vec<f64>,  // k * d (diag)
    pub resp: Vec<f64>,    // n * k
    pub n_iter: u32,
    pub lower_bound: f64, // avg log-likelihood per sample
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct GmmParams {
    pub n_components: usize,
    pub max_iter: u32,
    pub tol: f64,
    pub reg_covar: f64,
    pub init: String, // "kmeans++" or "random"
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
        return Err(ClustorError::InvalidArg("n_components must be > 0".into()));
    }
    if k > n_samples {
        return Err(ClustorError::InvalidArg(
            "n_components cannot exceed n_samples".into(),
        ));
    }
    Ok(())
}

#[inline]
fn logsumexp(v: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in v {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &x in v {
        s += (x - m).exp();
    }
    m + s.ln()
}

#[inline]
fn log_gaussian_diag(x: &[f64], mean: &[f64], var: &[f64], d: usize) -> f64 {
    let mut log_det = 0.0;
    let mut quad = 0.0;
    for j in 0..d {
        let v = var[j].max(1e-12);
        log_det += v.ln();
        let diff = x[j] - mean[j];
        quad += diff * diff / v;
    }
    -0.5 * ((d as f64) * LOG_2PI + log_det + quad)
}

fn init_params(
    rng: &mut StdRng,
    data: &[f64],
    n_samples: usize,
    d: usize,
    k: usize,
    init: &str,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; k * d];
    match init {
        "kmeans++" => {
            let centers =
                kmeans_plus_plus(rng, data, n_samples, d, k, Metric::Euclidean, None, false);
            means.copy_from_slice(&centers);
        }
        _ => {
            for c in 0..k {
                let idx = rng.random_range(0..n_samples);
                means[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
            }
        }
    }
    // global variance
    let mut global_mean = vec![0.0; d];
    for i in 0..n_samples {
        let x = &data[i * d..(i + 1) * d];
        for (mean_j, x_j) in global_mean.iter_mut().zip(x.iter()) {
            *mean_j += x_j;
        }
    }
    for mean_j in global_mean.iter_mut() {
        *mean_j /= n_samples as f64;
    }

    let mut global_var = vec![0.0; d];
    for i in 0..n_samples {
        let x = &data[i * d..(i + 1) * d];
        for ((var_j, mean_j), x_j) in global_var.iter_mut().zip(global_mean.iter()).zip(x.iter()) {
            let diff = x_j - mean_j;
            *var_j += diff * diff;
        }
    }
    for var_j in global_var.iter_mut() {
        *var_j = (*var_j / n_samples as f64).max(1e-6);
    }
    let mut covars = vec![0.0; k * d];
    for c in 0..k {
        covars[c * d..(c + 1) * d].copy_from_slice(&global_var);
    }
    let weights = vec![1.0 / (k as f64); k];
    (weights, means, covars)
}

pub fn fit_gmm_diag(
    data: &[f64],
    n_samples: usize,
    d: usize,
    params: &GmmParams,
) -> ClustorResult<GmmOutput> {
    validate_inputs(data, n_samples, d, params.n_components)?;
    if params.tol < 0.0 {
        return Err(ClustorError::InvalidArg("tol must be >= 0".into()));
    }
    if params.reg_covar < 0.0 {
        return Err(ClustorError::InvalidArg("reg_covar must be >= 0".into()));
    }

    let seed = params
        .random_state
        .unwrap_or_else(|| rand::rng().random::<u64>());
    let mut rng = StdRng::seed_from_u64(seed);

    let k = params.n_components;
    let (mut weights, mut means, mut covars) =
        init_params(&mut rng, data, n_samples, d, k, params.init.as_str());
    let mut resp = vec![0.0; n_samples * k];

    let mut prev_lb = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0u32;

    let mut log_prob_k = vec![0.0; k];

    for it in 0..params.max_iter {
        n_iter = it + 1;
        let mut ll_sum = 0.0;

        // E step
        for i in 0..n_samples {
            let x = &data[i * d..(i + 1) * d];
            for c in 0..k {
                let w = weights[c].max(1e-300);
                let mu = &means[c * d..(c + 1) * d];
                let var = &covars[c * d..(c + 1) * d];
                log_prob_k[c] = w.ln() + log_gaussian_diag(x, mu, var, d);
            }
            let lse = logsumexp(&log_prob_k);
            ll_sum += lse;
            for c in 0..k {
                resp[i * k + c] = (log_prob_k[c] - lse).exp();
            }
        }
        let lb = ll_sum / (n_samples as f64);
        if params.verbose {
            eprintln!("[Clustor][GMM] iter={} lower_bound={}", n_iter, lb);
        }
        if (lb - prev_lb).abs() <= params.tol {
            converged = true;
            prev_lb = lb;
            break;
        }
        prev_lb = lb;

        // M step
        let mut nk = vec![0.0; k];
        for i in 0..n_samples {
            for c in 0..k {
                nk[c] += resp[i * k + c];
            }
        }
        for c in 0..k {
            nk[c] = nk[c].max(1e-12);
            weights[c] = nk[c] / (n_samples as f64);
        }
        // means
        means.fill(0.0);
        for i in 0..n_samples {
            let x = &data[i * d..(i + 1) * d];
            for c in 0..k {
                let r = resp[i * k + c];
                let mu = &mut means[c * d..(c + 1) * d];
                for (mu_j, x_j) in mu.iter_mut().zip(x.iter()) {
                    *mu_j += r * x_j;
                }
            }
        }
        for c in 0..k {
            let inv = 1.0 / nk[c];
            let mu = &mut means[c * d..(c + 1) * d];
            for mu_j in mu.iter_mut() {
                *mu_j *= inv;
            }
        }
        // covars
        covars.fill(0.0);
        for i in 0..n_samples {
            let x = &data[i * d..(i + 1) * d];
            for c in 0..k {
                let r = resp[i * k + c];
                let mu = &means[c * d..(c + 1) * d];
                let var = &mut covars[c * d..(c + 1) * d];
                for ((var_j, mu_j), x_j) in var.iter_mut().zip(mu.iter()).zip(x.iter()) {
                    let diff = x_j - mu_j;
                    *var_j += r * diff * diff;
                }
            }
        }
        for c in 0..k {
            let inv = 1.0 / nk[c];
            let var = &mut covars[c * d..(c + 1) * d];
            for var_j in var.iter_mut() {
                *var_j = (*var_j * inv) + params.reg_covar;
                *var_j = var_j.max(1e-12);
            }
        }
    }

    Ok(GmmOutput {
        weights,
        means,
        covars,
        resp,
        n_iter,
        lower_bound: prev_lb,
        converged,
    })
}

// Sampling from standard normal via Box-Muller (no extra deps).
#[inline]
fn std_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub fn sample_gmm_diag(
    weights: &[f64],
    means: &[f64],
    covars: &[f64],
    k: usize,
    d: usize,
    n_samples: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::rng().random::<u64>()));
    let mut cum = vec![0.0; k];
    let mut s = 0.0;
    for (i, cum_i) in cum.iter_mut().enumerate() {
        s += weights[i];
        *cum_i = s;
    }
    if s <= 0.0 {
        for (i, cum_i) in cum.iter_mut().enumerate() {
            *cum_i = (i + 1) as f64;
        }
    }
    let total = cum[k - 1].max(1e-12);

    let mut out = vec![0.0; n_samples * d];
    for i in 0..n_samples {
        let r: f64 = rng.random_range(0.0..total);
        let mut c = 0usize;
        while c + 1 < k && r > cum[c] {
            c += 1;
        }
        let mu = &means[c * d..(c + 1) * d];
        let var = &covars[c * d..(c + 1) * d];
        for j in 0..d {
            out[i * d + j] = mu[j] + std_normal(&mut rng) * var[j].sqrt();
        }
    }
    out
}

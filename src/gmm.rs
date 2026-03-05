// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::Metric;
use crate::utils::{kmeans_plus_plus, validate_data_shape};

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
    validate_data_shape(
        data.len(),
        n_samples,
        n_features,
        "X data length does not match shape",
        "X shape product overflows usize",
    )?;
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
pub(crate) fn logsumexp(v: &[f64]) -> f64 {
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
pub(crate) fn log_gaussian_diag(x: &[f64], mean: &[f64], var: &[f64], d: usize) -> f64 {
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

fn validate_sample_weight(sample_weight: &[f64], n_samples: usize) -> ClustorResult<f64> {
    if sample_weight.len() != n_samples {
        return Err(ClustorError::InvalidArg(
            "sample_weight length does not match n_samples".into(),
        ));
    }
    let mut total = 0.0;
    for &w in sample_weight {
        if !w.is_finite() {
            return Err(ClustorError::InvalidArg(
                "sample_weight values must be finite".into(),
            ));
        }
        if w < 0.0 {
            return Err(ClustorError::InvalidArg(
                "sample_weight values must be >= 0".into(),
            ));
        }
        total += w;
    }
    if total <= 0.0 {
        return Err(ClustorError::InvalidArg(
            "sample_weight must sum to a positive value".into(),
        ));
    }
    Ok(total)
}

#[allow(clippy::too_many_arguments)]
pub fn gmm_log_likelihoods_diag(
    weights: &[f64],
    means: &[f64],
    covars: &[f64],
    data: &[f64],
    n_samples: usize,
    d: usize,
    k: usize,
    mut resp: Option<&mut [f64]>,
) -> Vec<f64> {
    let mut log_likelihood = vec![0.0; n_samples];
    let mut log_prob = vec![0.0; k];
    for i in 0..n_samples {
        let xi = &data[i * d..(i + 1) * d];
        for c in 0..k {
            let w = weights[c].max(1e-300);
            let mu = &means[c * d..(c + 1) * d];
            let var = &covars[c * d..(c + 1) * d];
            log_prob[c] = w.ln() + log_gaussian_diag(xi, mu, var, d);
        }
        let lse = logsumexp(&log_prob);
        log_likelihood[i] = lse;
        if let Some(resp_buf) = resp.as_deref_mut() {
            for c in 0..k {
                resp_buf[i * k + c] = (log_prob[c] - lse).exp();
            }
        }
    }
    log_likelihood
}

pub fn gmm_log_resp_diag(
    weights: &[f64],
    means: &[f64],
    covars: &[f64],
    data: &[f64],
    n_samples: usize,
    d: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut resp = vec![0.0; n_samples * k];
    let log_likelihood = gmm_log_likelihoods_diag(
        weights,
        means,
        covars,
        data,
        n_samples,
        d,
        k,
        Some(&mut resp),
    );
    (resp, log_likelihood)
}

pub fn fit_gmm_diag(
    data: &[f64],
    n_samples: usize,
    d: usize,
    params: &GmmParams,
    sample_weight: Option<&[f64]>,
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
    let total_weight = match sample_weight {
        Some(weights) => validate_sample_weight(weights, n_samples)?,
        None => n_samples as f64,
    };
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
            let weight = sample_weight.map_or(1.0, |w| w[i]);
            ll_sum += weight * lse;
            for c in 0..k {
                resp[i * k + c] = (log_prob_k[c] - lse).exp();
            }
        }
        let lb = ll_sum / total_weight;
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
            let weight = sample_weight.map_or(1.0, |w| w[i]);
            for c in 0..k {
                nk[c] += resp[i * k + c] * weight;
            }
        }
        for c in 0..k {
            nk[c] = nk[c].max(1e-12);
            weights[c] = nk[c] / total_weight;
        }
        // means
        means.fill(0.0);
        for i in 0..n_samples {
            let x = &data[i * d..(i + 1) * d];
            let weight = sample_weight.map_or(1.0, |w| w[i]);
            for c in 0..k {
                let r = resp[i * k + c] * weight;
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
            let weight = sample_weight.map_or(1.0, |w| w[i]);
            for c in 0..k {
                let r = resp[i * k + c] * weight;
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
fn std_normal<R: RngExt>(rng: &mut R) -> f64 {
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
    if k == 0 || d == 0 || n_samples == 0 {
        return Vec::new();
    }
    let kd = match k.checked_mul(d) {
        Some(v) => v,
        None => return Vec::new(),
    };
    if weights.len() < k || means.len() < kd || covars.len() < kd {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(|| rand::rng().random::<u64>()));
    let mut cum = vec![0.0; k];
    let mut s = 0.0;
    for (i, cum_i) in cum.iter_mut().enumerate() {
        let w = weights[i];
        s += if w.is_finite() && w > 0.0 { w } else { 0.0 };
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
        let c = cum.partition_point(|&v| v < r).min(k - 1);
        let mu = &means[c * d..(c + 1) * d];
        let var = &covars[c * d..(c + 1) * d];
        for j in 0..d {
            let sigma = var[j].max(0.0).sqrt();
            out[i * d + j] = mu[j] + std_normal(&mut rng) * sigma;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GmmParams {
        GmmParams {
            n_components: 2,
            max_iter: 20,
            tol: 1e-4,
            reg_covar: 1e-6,
            init: "kmeans++".to_string(),
            random_state: Some(0),
            verbose: false,
        }
    }

    #[test]
    fn gmm_rejects_invalid_component_count() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let mut p = params();
        p.n_components = 0;
        let err = fit_gmm_diag(&data, 2, 2, &p, None).unwrap_err();
        assert!(matches!(err, ClustorError::InvalidArg(msg) if msg.contains("n_components")));
    }

    #[test]
    fn sample_gmm_diag_handles_invalid_shapes_and_zero_sizes() {
        assert!(sample_gmm_diag(&[], &[], &[], 0, 2, 5, Some(1)).is_empty());
        assert!(sample_gmm_diag(&[1.0], &[0.0], &[1.0], 1, 1, 0, Some(1)).is_empty());
        assert!(sample_gmm_diag(&[1.0], &[0.0], &[1.0], 1, 2, 3, Some(1)).is_empty());
    }

    #[test]
    fn sample_gmm_diag_can_select_last_component() {
        let out = sample_gmm_diag(
            &[0.0, 0.0, 1.0],
            &[0.0, 0.0, 9.0],
            &[1.0, 1.0, 0.0],
            3,
            1,
            8,
            Some(123),
        );
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|&v| (v - 9.0).abs() < 1e-12));
    }

    #[test]
    fn sample_gmm_diag_sanitizes_non_positive_weights_and_negative_variance() {
        let out = sample_gmm_diag(
            &[f64::NAN, -1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[-1.0, -4.0, -9.0],
            3,
            1,
            4,
            Some(42),
        );
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gmm_fit_and_helpers_smoke() {
        let data = vec![0.0, 0.0, 0.2, 0.1, 5.0, 5.0, 5.2, 5.1];
        let out = fit_gmm_diag(&data, 4, 2, &params(), None).unwrap();
        assert_eq!(out.weights.len(), 2);
        assert_eq!(out.means.len(), 4);
        assert_eq!(out.covars.len(), 4);
        assert_eq!(out.resp.len(), 8);

        let (resp, ll) = gmm_log_resp_diag(&out.weights, &out.means, &out.covars, &data, 4, 2, 2);
        assert_eq!(resp.len(), 8);
        assert_eq!(ll.len(), 4);

        let samples = sample_gmm_diag(&out.weights, &out.means, &out.covars, 2, 2, 3, Some(7));
        assert_eq!(samples.len(), 6);
    }
}

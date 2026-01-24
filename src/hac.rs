// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::errors::{ClustorError, ClustorResult};
use crate::metrics::{Metric, cosine_distance, euclidean_sq};
use crate::utils::compute_row_norms;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

impl Linkage {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "single" => Some(Linkage::Single),
            "complete" => Some(Linkage::Complete),
            "average" => Some(Linkage::Average),
            "ward" => Some(Linkage::Ward),
            _ => None,
        }
    }
}

fn validate_inputs(data: &[f64], n_samples: usize, n_features: usize) -> ClustorResult<()> {
    if n_samples < 2 {
        return Err(ClustorError::InvalidArg("need at least 2 samples".into()));
    }
    if n_features == 0 {
        return Err(ClustorError::InvalidArg("n_features must be > 0".into()));
    }
    if data.len() != n_samples * n_features {
        return Err(ClustorError::InvalidArg("data length mismatch".into()));
    }
    Ok(())
}

#[inline]
fn point_dist(
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

/// Returns linkage matrix Z with shape (n-1, 4) following SciPy convention:
/// [idx1, idx2, dist, sample_count].
pub fn hac_linkage(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    method: Linkage,
    metric: Metric,
) -> ClustorResult<Vec<f64>> {
    validate_inputs(data, n_samples, n_features)?;
    if method == Linkage::Ward && metric != Metric::Euclidean {
        return Err(ClustorError::InvalidArg(
            "ward linkage requires euclidean metric".into(),
        ));
    }

    let norms = if metric == Metric::Cosine {
        Some(compute_row_norms(data, n_samples, n_features))
    } else {
        None
    };
    let norms_ref = norms.as_deref();

    let n = n_samples;
    let mut active = vec![true; n];
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut cluster_size: Vec<usize> = vec![1; n];

    // Dist matrix (full, symmetric), store in row-major n*n
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dij = point_dist(data, norms_ref, i, j, n_features, metric);
            d[i * n + j] = dij;
            d[j * n + i] = dij;
        }
    }

    let mut z = Vec::with_capacity((n - 1) * 4);

    for step in 0..(n - 1) {
        // find closest pair among active clusters
        let mut best = f64::INFINITY;
        let mut bi = 0usize;
        let mut bj = 1usize;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let dij = d[i * n + j];
                if dij < best {
                    best = dij;
                    bi = i;
                    bj = j;
                }
            }
        }

        let id_i = cluster_id[bi] as f64;
        let id_j = cluster_id[bj] as f64;
        let new_count = (cluster_size[bi] + cluster_size[bj]) as f64;

        z.push(id_i);
        z.push(id_j);
        z.push(best);
        z.push(new_count);

        // new cluster id follows SciPy: n + step
        let new_id = n + step;
        cluster_id[bi] = new_id;
        cluster_size[bi] += cluster_size[bj];
        active[bj] = false;

        // Update distances between merged cluster bi and all other active clusters
        for k in 0..n {
            if !active[k] || k == bi {
                continue;
            }
            let dik = d[bi * n + k];
            let djk = d[bj * n + k];
            let new_d = match method {
                Linkage::Single => dik.min(djk),
                Linkage::Complete => dik.max(djk),
                Linkage::Average => {
                    let si = cluster_size[bi] - cluster_size[bj];
                    let sj = cluster_size[bj];
                    let denom = (si + sj) as f64;
                    (dik * (si as f64) + djk * (sj as f64)) / denom
                }
                Linkage::Ward => {
                    // Lance-Williams update on squared distances
                    let ni = (cluster_size[bi] - cluster_size[bj]) as f64;
                    let nj = cluster_size[bj] as f64;
                    let nk = cluster_size[k] as f64;
                    let dij = best;
                    let dik2 = dik * dik;
                    let djk2 = djk * djk;
                    let dij2 = dij * dij;
                    let num = (ni + nk) * dik2 + (nj + nk) * djk2 - nk * dij2;
                    let den = ni + nj + nk;
                    (num / den).max(0.0).sqrt()
                }
            };
            d[bi * n + k] = new_d;
            d[k * n + bi] = new_d;
        }

        // distances to inactive bj are irrelevant
    }

    Ok(z)
}

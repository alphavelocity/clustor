// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    Cosine,
}

impl Metric {
    pub fn parse(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.eq_ignore_ascii_case(b"l2") || bytes.eq_ignore_ascii_case(b"euclidean") {
            Some(Metric::Euclidean)
        } else if bytes.eq_ignore_ascii_case(b"cosine") {
            Some(Metric::Cosine)
        } else {
            None
        }
    }
}

#[inline]
pub fn l2_norm(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in x {
        s += v * v;
    }
    s.sqrt()
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Cosine distance in [0,2]: (1 - cosine similarity).
/// Zero-vector handling matches common cosine implementations:
/// - both norms zero => distance 0
/// - either norm zero => distance 1
#[inline]
pub fn cosine_distance(a: &[f64], a_norm: f64, b: &[f64], b_norm: f64) -> f64 {
    if !a_norm.is_finite() || !b_norm.is_finite() {
        return 1.0;
    }
    if a_norm == 0.0 && b_norm == 0.0 {
        return 0.0;
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        return 1.0;
    }
    let denom = a_norm * b_norm;
    if !denom.is_finite() || denom <= 0.0 {
        return 1.0;
    }
    let dot = dot(a, b);
    if !dot.is_finite() {
        return 1.0;
    }
    let mut sim = dot / denom;
    if !sim.is_finite() {
        return 1.0;
    }
    sim = sim.clamp(-1.0, 1.0);
    1.0 - sim
}

#[inline]
pub fn normalize_in_place(v: &mut [f64]) {
    let n = l2_norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Metric, cosine_distance};

    #[test]
    fn metric_parse_accepts_aliases_case_insensitively() {
        assert_eq!(Metric::parse("EUCLIDEAN"), Some(Metric::Euclidean));
        assert_eq!(Metric::parse("L2"), Some(Metric::Euclidean));
        assert_eq!(Metric::parse("CoSiNe"), Some(Metric::Cosine));
    }

    #[test]
    fn metric_parse_rejects_unknown_values() {
        assert_eq!(Metric::parse("manhattan"), None);
        assert_eq!(Metric::parse(""), None);
        assert_eq!(Metric::parse(" euclidean"), None);
    }

    #[test]
    fn metric_parse_rejects_non_ascii_confusables() {
        assert_eq!(Metric::parse("cоsine"), None); // Cyrillic small o
    }

    #[test]
    fn cosine_distance_handles_zero_vectors() {
        let a = [0.0, 0.0];
        let b = [0.0, 0.0];
        assert_eq!(cosine_distance(&a, 0.0, &b, 0.0), 0.0);
    }

    #[test]
    fn cosine_distance_zero_vs_nonzero() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        assert_eq!(cosine_distance(&a, 0.0, &b, 1.0), 1.0);
    }

    #[test]
    fn cosine_distance_parallel_vectors() {
        let a = [1.0, 0.0];
        let b = [2.0, 0.0];
        assert_eq!(cosine_distance(&a, 1.0, &b, 2.0), 0.0);
    }

    #[test]
    fn cosine_distance_opposite_vectors() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert_eq!(cosine_distance(&a, 1.0, &b, 1.0), 2.0);
    }
}

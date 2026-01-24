// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::errors::{ClustorError, ClustorResult};
use crate::kmeans::{KMeansParams, fit_kmeans};
use crate::metrics::Metric;
use crate::metrics::euclidean_sq;

#[derive(Clone, Debug)]
struct CfEntry {
    n: f64,
    ls: Vec<f64>,
    ss: Vec<f64>,
}

impl CfEntry {
    fn new_from_point(x: &[f64]) -> Self {
        let ls = x.to_vec();
        let mut ss = Vec::with_capacity(x.len());
        for &v in x {
            ss.push(v * v);
        }
        Self { n: 1.0, ls, ss }
    }

    fn add_point(&mut self, x: &[f64]) {
        self.n += 1.0;
        for (j, &v) in x.iter().enumerate() {
            self.ls[j] += v;
            self.ss[j] += v * v;
        }
    }

    fn merge(&mut self, other: &CfEntry) {
        self.n += other.n;
        for j in 0..self.ls.len() {
            self.ls[j] += other.ls[j];
            self.ss[j] += other.ss[j];
        }
    }

    fn centroid(&self) -> Vec<f64> {
        let inv = 1.0 / self.n.max(1e-12);
        self.ls.iter().map(|v| v * inv).collect()
    }

    fn radius_after_adding(&self, x: &[f64]) -> f64 {
        // radius = sqrt(mean squared distance from centroid)
        // Use CF formula: E[||X||^2] - ||E[X]||^2
        let n2 = self.n + 1.0;
        let mut ls = self.ls.clone();
        let mut ss = self.ss.clone();
        for (j, &v) in x.iter().enumerate() {
            ls[j] += v;
            ss[j] += v * v;
        }
        let inv = 1.0 / n2;
        let mut ex2 = 0.0;
        let mut ex = 0.0;
        for j in 0..ls.len() {
            ex2 += ss[j] * inv;
            let m = ls[j] * inv;
            ex += m * m;
        }
        (ex2 - ex).max(0.0).sqrt()
    }
}

#[derive(Clone, Debug)]
#[allow(clippy::vec_box)]
enum Node {
    Leaf {
        entries: Vec<CfEntry>,
    },
    Internal {
        entries: Vec<CfEntry>,
        children: Vec<Box<Node>>,
    },
}

impl Node {
    fn summary(&self) -> CfEntry {
        match self {
            Node::Leaf { entries } => {
                let mut it = entries.iter();
                let first = it.next().expect("leaf has at least one entry").clone();
                let mut sum = first;
                for e in it {
                    sum.merge(e);
                }
                sum
            }
            Node::Internal { entries, .. } => {
                let mut it = entries.iter();
                let first = it.next().expect("internal has at least one entry").clone();
                let mut sum = first;
                for e in it {
                    sum.merge(e);
                }
                sum
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BirchParams {
    pub threshold: f64,
    pub branching_factor: usize,
    pub n_clusters: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct BirchOutput {
    pub centers: Vec<f64>,
    pub labels: Vec<usize>,
    pub n_subclusters: usize,
}

fn validate_inputs(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &BirchParams,
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
    if params.threshold <= 0.0 || !params.threshold.is_finite() {
        return Err(ClustorError::InvalidArg(
            "threshold must be finite and > 0".into(),
        ));
    }
    if params.branching_factor < 2 {
        return Err(ClustorError::InvalidArg(
            "branching_factor must be >= 2".into(),
        ));
    }
    if matches!(params.n_clusters, Some(0)) {
        return Err(ClustorError::InvalidArg("n_clusters must be > 0".into()));
    }
    Ok(())
}

fn centroid_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    euclidean_sq(a, b)
}

fn choose_closest_entry(entries: &[CfEntry], x: &[f64]) -> usize {
    let mut best = f64::INFINITY;
    let mut best_i = 0usize;
    for (i, e) in entries.iter().enumerate() {
        let c = e.centroid();
        let d = centroid_distance_sq(&c, x);
        if d < best {
            best = d;
            best_i = i;
        }
    }
    best_i
}

fn farthest_pair(entries: &[CfEntry]) -> (usize, usize) {
    let mut best = -1.0;
    let mut a = 0usize;
    let mut b = 1usize;
    for i in 0..entries.len() {
        let ci = entries[i].centroid();
        for (j, entry_j) in entries.iter().enumerate().skip(i + 1) {
            let cj = entry_j.centroid();
            let d = centroid_distance_sq(&ci, &cj);
            if d > best {
                best = d;
                a = i;
                b = j;
            }
        }
    }
    (a, b)
}

fn split_leaf(entries: Vec<CfEntry>) -> (Vec<CfEntry>, Vec<CfEntry>) {
    let (ia, ib) = farthest_pair(&entries);
    let ca = entries[ia].centroid();
    let cb = entries[ib].centroid();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for e in entries {
        let c = e.centroid();
        let da = centroid_distance_sq(&c, &ca);
        let db = centroid_distance_sq(&c, &cb);
        if da <= db {
            left.push(e);
        } else {
            right.push(e);
        }
    }
    if left.is_empty() {
        left.push(right.pop().unwrap());
    }
    if right.is_empty() {
        right.push(left.pop().unwrap());
    }
    (left, right)
}

fn insert_into_node(node: &mut Node, x: &[f64], params: &BirchParams) -> Option<(Node, Node)> {
    match node {
        Node::Leaf { entries } => {
            if entries.is_empty() {
                entries.push(CfEntry::new_from_point(x));
                return None;
            }
            let idx = choose_closest_entry(entries, x);
            if entries[idx].radius_after_adding(x) <= params.threshold {
                entries[idx].add_point(x);
            } else {
                entries.push(CfEntry::new_from_point(x));
            }
            if entries.len() > params.branching_factor {
                let old = std::mem::take(entries);
                let (l, r) = split_leaf(old);
                return Some((Node::Leaf { entries: l }, Node::Leaf { entries: r }));
            }
            None
        }
        Node::Internal { entries, children } => {
            if children.is_empty() {
                // initialize with a leaf
                let leaf = Node::Leaf {
                    entries: vec![CfEntry::new_from_point(x)],
                };
                children.push(Box::new(leaf));
                entries.push(children[0].summary());
                return None;
            }
            // choose closest child summary
            let mut best = f64::INFINITY;
            let mut best_i = 0usize;
            for (i, e) in entries.iter().enumerate() {
                let c = e.centroid();
                let d = centroid_distance_sq(&c, x);
                if d < best {
                    best = d;
                    best_i = i;
                }
            }
            let split = insert_into_node(children[best_i].as_mut(), x, params);
            // refresh this child's summary
            entries[best_i] = children[best_i].summary();

            if let Some((left, right)) = split {
                // replace child with two nodes
                children.remove(best_i);
                entries.remove(best_i);

                children.insert(best_i, Box::new(left));
                entries.insert(best_i, children[best_i].summary());
                children.insert(best_i + 1, Box::new(right));
                entries.insert(best_i + 1, children[best_i + 1].summary());
            }

            if entries.len() > params.branching_factor {
                // split this internal node
                let old_entries = std::mem::take(entries);
                let old_children = std::mem::take(children);

                // split based on entries (summaries)
                let (ia, ib) = farthest_pair(&old_entries);
                let ca = old_entries[ia].centroid();
                let cb = old_entries[ib].centroid();

                let mut left_entries = Vec::new();
                let mut left_children: Vec<Box<Node>> = Vec::new();
                let mut right_entries = Vec::new();
                let mut right_children: Vec<Box<Node>> = Vec::new();

                for (e, cnode) in old_entries.into_iter().zip(old_children.into_iter()) {
                    let cent = e.centroid();
                    let da = centroid_distance_sq(&cent, &ca);
                    let db = centroid_distance_sq(&cent, &cb);
                    if da <= db {
                        left_entries.push(e);
                        left_children.push(cnode);
                    } else {
                        right_entries.push(e);
                        right_children.push(cnode);
                    }
                }
                if left_entries.is_empty() {
                    left_entries.push(right_entries.pop().unwrap());
                    left_children.push(right_children.pop().unwrap());
                }
                if right_entries.is_empty() {
                    right_entries.push(left_entries.pop().unwrap());
                    right_children.push(left_children.pop().unwrap());
                }

                let left_node = Node::Internal {
                    entries: left_entries,
                    children: left_children,
                };
                let right_node = Node::Internal {
                    entries: right_entries,
                    children: right_children,
                };
                return Some((left_node, right_node));
            }

            None
        }
    }
}

fn collect_leaf_entries(node: &Node, out: &mut Vec<CfEntry>) {
    match node {
        Node::Leaf { entries } => out.extend(entries.iter().cloned()),
        Node::Internal { children, .. } => {
            for ch in children {
                collect_leaf_entries(ch, out);
            }
        }
    }
}

pub fn fit_birch(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &BirchParams,
) -> ClustorResult<BirchOutput> {
    validate_inputs(data, n_samples, n_features, params)?;

    let mut root = Node::Leaf {
        entries: Vec::new(),
    };

    for i in 0..n_samples {
        let x = &data[i * n_features..(i + 1) * n_features];
        let split = insert_into_node(&mut root, x, params);
        if let Some((left, right)) = split {
            // root split => create new root
            let children: Vec<Box<Node>> = vec![Box::new(left), Box::new(right)];
            let entries = vec![children[0].summary(), children[1].summary()];
            root = Node::Internal { entries, children };
        }
    }

    let mut leaf_entries = Vec::new();
    collect_leaf_entries(&root, &mut leaf_entries);
    let n_sub = leaf_entries.len();

    // Build subcluster centroids and weights
    let mut sub_centers = Vec::with_capacity(n_sub * n_features);
    let mut sub_weights = Vec::with_capacity(n_sub);
    for e in &leaf_entries {
        let c = e.centroid();
        sub_centers.extend_from_slice(&c);
        sub_weights.push(e.n);
    }

    // If n_clusters provided, run weighted kmeans on subcluster centroids
    let (centers, labels) = if let Some(k) = params.n_clusters {
        let km_params = KMeansParams {
            n_clusters: k,
            n_init: 5,
            max_iter: 200,
            tol: 1e-4,
            metric: Metric::Euclidean,
            normalize_input: false,
            normalize_centers: false,
            random_state: Some(0),
            verbose: false,
        };
        let out = fit_kmeans(
            &sub_centers,
            n_sub,
            n_features,
            Some(&sub_weights),
            &km_params,
        )?;
        // assign each sample to nearest final center
        let mut labels = vec![0usize; n_samples];
        for i in 0..n_samples {
            let x = &data[i * n_features..(i + 1) * n_features];
            let mut best = f64::INFINITY;
            let mut best_k = 0usize;
            for cidx in 0..k {
                let c = &out.centers[cidx * n_features..(cidx + 1) * n_features];
                let d = euclidean_sq(x, c);
                if d < best {
                    best = d;
                    best_k = cidx;
                }
            }
            labels[i] = best_k;
        }
        (out.centers, labels)
    } else {
        // Use subcluster centers; assign each sample to nearest subcluster
        let mut labels = vec![0usize; n_samples];
        for i in 0..n_samples {
            let x = &data[i * n_features..(i + 1) * n_features];
            let mut best = f64::INFINITY;
            let mut best_k = 0usize;
            for sc in 0..n_sub {
                let c = &sub_centers[sc * n_features..(sc + 1) * n_features];
                let d = euclidean_sq(x, c);
                if d < best {
                    best = d;
                    best_k = sc;
                }
            }
            labels[i] = best_k;
        }
        (sub_centers, labels)
    };

    Ok(BirchOutput {
        centers,
        labels,
        n_subclusters: n_sub,
    })
}

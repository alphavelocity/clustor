<h1 align="center">

Clustor

</h1>

<div align="center">

[![Current Release](https://img.shields.io/github/release/alphavelocity/clustor.svg)](https://github.com/alphavelocity/clustor/releases)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![rustc 1.89+](https://img.shields.io/badge/rustc-1.89+-blue.svg?logo=rust&logoColor=white)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Test Linux](https://github.com/alphavelocity/clustor/actions/workflows/test_ubuntu.yml/badge.svg)](https://github.com/alphavelocity/clustor/actions/workflows/test_ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/alphavelocity/clustor/actions/workflows/test_windows.yml/badge.svg)](https://github.com/alphavelocity/clustor/actions/workflows/test_windows.yml?query=branch%3Amain)
[![Lints](https://github.com/alphavelocity/clustor/actions/workflows/lints.yml/badge.svg)](https://github.com/alphavelocity/clustor/actions/workflows/lints.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

Clustor is a Rust-accelerated clustering toolkit with a Python-first API built on PyO3 and
maturin. It delivers fast, minimal-dependency implementations of common clustering algorithms and
cluster validation metrics for use inside production data pipelines.

Clustor focuses on unsupervised learning primitives. Use it alongside dataset balancing tools when
you need reweighting or resampling.

> [!CAUTION]
> Active development. APIs may evolve before a 1.0 release.

## Features

- KMeans and MiniBatchKMeans with KMeans++ initialization, `n_init`, Euclidean/Cosine metrics,
  optional normalization, `sample_weight`, and streaming `partial_fit`.
- BisectingKMeans for hierarchical top-down splitting.
- DBSCAN with noise labeling and core sample indices.
- OPTICS reachability ordering and core distances.
- Affinity Propagation for exemplar-based clustering.
- BIRCH streaming clustering using CF-tree summaries.
- GaussianMixture (EM, diagonal covariance): `predict_proba`, `score_samples`, `score`, `aic`,
  `bic`, `sample`, and model selection metrics.
- Cluster validation: Silhouette, Calinski–Harabasz, Davies–Bouldin.
- HAC linkage (dendrogram output) with SciPy-style linkage matrices.

## Installation

### Python (recommended)

Install from pypi:

```bash
pip install clustor
```

### Local development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -U maturin numpy pytest pre-commit
pre-commit run --all-files
maturin develop --release
pytest -q
```

## Quick usage

```python
import numpy as np
import clustor

X = np.vstack([np.random.randn(200, 2), np.random.randn(200, 2) + 6])

km = clustor.KMeans(n_clusters=2, n_init=10, random_state=0)
result = km.fit(X)
print(result["inertia"])

op = clustor.OPTICS(min_samples=10, max_eps=1.0)
res = op.fit(X)
print(res["reachability"].shape, res["ordering"].shape)

ap = clustor.AffinityPropagation(damping=0.7)
res = ap.fit(X)
print("clusters:", len(res["exemplars"]), "converged:", res["converged"])

br = clustor.Birch(threshold=0.7, branching_factor=30, n_clusters=2)
res = br.fit(X)
print(res["centers"].shape)

Z = clustor.hac_dendrogram(X, method="average", metric="euclidean")
print(Z.shape)  # (n-1, 4)
```

## Advanced usage

### Streaming MiniBatchKMeans

```python
mb = clustor.MiniBatchKMeans(n_clusters=2, batch_size=2, max_steps=20, random_state=0)
mb.fit(X[:3])
mb.partial_fit(X[3:])
print(mb.predict(X))
```

Output:

```
[0 0 0 1 1 1]
```

### GaussianMixture model selection metrics

```python
gm = clustor.GaussianMixture(2, max_iter=50, tol=1e-4, random_state=0)
gm.fit(X)
print(gm.score_samples(X))
print(gm.score(X))
print(gm.aic(X))
print(gm.bic(X))
```

Output:

```
[-1.526949 -2.276946 -2.276946 -1.526949 -2.276946 -2.276946]
-2.026946850203142
42.3233622024377
40.4491974254902
```

## Examples with outputs

The following examples use a fixed dataset and deterministic seeds to keep outputs stable.

```python
import numpy as np
import clustor

np.set_printoptions(precision=6, suppress=True)
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [10.0, 10.0],
        [10.0, 11.0],
        [11.0, 10.0],
    ],
    dtype=np.float64,
)
```

### KMeans

```python
km = clustor.KMeans(n_clusters=2, n_init=1, max_iter=50, random_state=0)
km_out = km.fit(X)
print(km_out["centers"])
print(km_out["labels"])
```

Output:

```
[[10.333333 10.333333]
 [ 0.333333  0.333333]]
[1 1 1 0 0 0]
```

### MiniBatchKMeans

```python
mb = clustor.MiniBatchKMeans(n_clusters=2, batch_size=3, max_steps=200, random_state=0)
mb_out = mb.fit(X)
print(mb_out["centers"])
print(mb.predict(X))
```

Output:

```
[[10.374126 10.300699]
 [ 0.33121   0.321656]]
[1 1 1 0 0 0]
```

### BisectingKMeans

```python
bk = clustor.BisectingKMeans(n_clusters=2, n_init=1, max_iter=50, random_state=0)
bk_out = bk.fit(X)
print(bk_out["centers"])
print(bk.predict(X))
```

Output:

```
[[10.333333 10.333333]
 [ 0.333333  0.333333]]
[1 1 1 0 0 0]
```

### DBSCAN

```python
db = clustor.DBSCAN(eps=1.6, min_samples=2)
db_out = db.fit(X)
print(db_out["labels"])
print(db_out["core_sample_indices"])
```

Output:

```
[0 0 0 1 1 1]
[0 1 2 3 4 5]
```

### OPTICS

```python
opt = clustor.OPTICS(min_samples=2, max_eps=2.0)
opt_out = opt.fit(X)
print(opt_out["ordering"])
print(opt_out["reachability"])
print(opt_out["core_distances"])
print(opt_out["predecessor"])
```

Output:

```
[0 2 1 3 5 4]
[inf  1.  1. inf  1.  1.]
[1. 1. 1. 1. 1. 1.]
[-1  0  0 -1  3  3]
```

### Affinity Propagation

```python
ap = clustor.AffinityPropagation(damping=0.6, max_iter=200, convergence_iter=15)
ap_out = ap.fit(X)
print(ap_out["exemplars"])
print(ap_out["labels"])
print(ap_out["converged"])
```

Output:

```
[0 3]
[0 0 0 1 1 1]
True
```

### Birch

```python
br = clustor.Birch(threshold=1.5, branching_factor=10, n_clusters=2)
br_out = br.fit(X)
print(br_out["centers"])
print(br_out["labels"])
print(br_out["n_subclusters"])
```

Output:

```
[[10.333333 10.333333]
 [ 0.333333  0.333333]]
[1 1 1 0 0 0]
2
```

### GaussianMixture

```python
gm = clustor.GaussianMixture(2, max_iter=50, tol=1e-4, random_state=0)
gm_out = gm.fit(X)
print(gm_out["weights"])
print(gm_out["means"])
print(gm_out["covars"])
print(gm_out["converged"])
```

Output:

```
[0.5 0.5]
[[10.333333 10.333333]
 [ 0.333333  0.333333]]
[[0.222223 0.222223]
 [0.222223 0.222223]]
True
```

### Validation metrics

```python
labels = km_out["labels"]
print(clustor.silhouette_score(X, labels))
print(clustor.calinski_harabasz_score(X, labels))
print(clustor.davies_bouldin_score(X, labels))
```

Output:

```
0.9196222281154851
450.00000000000017
0.09249505911485287
```

### HAC linkage

```python
Z = clustor.hac_dendrogram(X, method="average", metric="euclidean")
print(Z)
```

Output:

```
[[ 0.        1.        1.        2.      ]
 [ 3.        4.        1.        2.      ]
 [ 6.        2.        1.207107  3.      ]
 [ 7.        5.        1.207107  3.      ]
 [ 8.        9.       14.165681  6.      ]]
```

## API overview

### Estimators

- `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`
- `DBSCAN`
- `OPTICS`
- `AffinityPropagation`
- `Birch`
- `GaussianMixture`

All estimators accept NumPy arrays and return NumPy arrays for outputs. Methods that accept
`sample_weight` validate length consistency with the input data.

### Functional APIs

- `kmeans`, `bisecting_kmeans`, `dbscan`, `gaussian_mixture`
- `optics`, `affinity_propagation`, `birch`, `hac_dendrogram`

### Metrics

- `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`

## Example notebooks

Each notebook is executed with deterministic inputs and includes narrative context plus outputs.

- [`examples/01_kmeans_family.ipynb`](https://github.com/alphavelocity/clustor/blob/main/examples/01_kmeans_family.ipynb): KMeans, MiniBatchKMeans, and BisectingKMeans.
- [`examples/02_density_clustering.ipynb`](https://github.com/alphavelocity/clustor/blob/main/examples/02_density_clustering.ipynb): DBSCAN and OPTICS.
- [`examples/03_affinity_birch.ipynb`](https://github.com/alphavelocity/clustor/blob/main/examples/03_affinity_birch.ipynb): Affinity Propagation and Birch.
- [`examples/04_gaussian_mixture_metrics.ipynb`](https://github.com/alphavelocity/clustor/blob/main/examples/04_gaussian_mixture_metrics.ipynb): GaussianMixture and validation metrics.
- [`examples/05_hac_linkage.ipynb`](https://github.com/alphavelocity/clustor/blob/main/examples/05_hac_linkage.ipynb): HAC linkage matrices.

## Testing and quality

- Rust tests: `cargo test`
- Python tests: `pytest`
- Linting and formatting: `pre-commit run --all-files`

## Citation
If you use clustor in your work and wish to refer to it, please use the following BibTeX entry.
```bibtex
@software{clustor,
  author = {Soumyadip Sarkar},
  title = {Clustor: Rust-accelerated High-Performance Clustering Algorithms},
  year = {2026},
  url = {https://github.com/alphavelocity/clustor}
}
```

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

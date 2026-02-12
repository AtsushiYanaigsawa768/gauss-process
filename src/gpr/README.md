# gpr/ -- Gaussian Process Regression

[日本語版はこちら](../../docs/ja/src/gpr/README.md)

## Purpose

Fits smooth transfer function models to noisy frequency response data using
Gaussian Process regression, with multiple kernel choices, hyperparameter
optimization, and outlier-robust variants.

## Files

| File | Description |
|---|---|
| `kernels.py` | 14 kernels (RBF, Matern, RQ, Exponential, TC, DC, DI, SS1, SS2, HFSS, StableSpline, ...) + `create_kernel()` factory |
| `gpr_fitting.py` | `GaussianProcessRegressor` class with gradient-based and grid-search optimization |
| `grid_search.py` | Exhaustive and random hyperparameter search with cross-validation |
| `itgp.py` | Iteratively-Trimmed GP (ITGP) using `robustgp` library with Hampel outlier filtering |
| `t_distribution.py` | Student-t likelihood GP via GPflow VGP for heavy-tailed noise robustness |
| `linear_interpolation.py` | Linear interpolation baseline with Hampel-filtered evaluation |
| `visualization.py` | GP-specific plotting (Bode, Nyquist, predicted vs actual) |
| `pure_gp_kernels.py` | From-scratch kernel implementations (no sklearn dependency) |
| `pure_gp_fitting.py` | From-scratch GP fitting with manual marginal likelihood optimization |
| `knn_noise_filter.py` | k-NN based noise filtering for frequency response data |
| `least_squares.py` | Least-squares polynomial fitting for comparison |
| `descriptive_stats.py` | Summary statistics for frequency response datasets |

## Kernel Reference

The `create_kernel(name, **kwargs)` factory supports these names:

| Kernel | Key Parameters |
|---|---|
| `rbf` | `length_scale`, `variance` |
| `matern` | `length_scale`, `variance`, `nu` (0.5, 1.5, 2.5) |
| `rational_quadratic` | `length_scale`, `variance`, `alpha` |
| `exponential` | `length_scale`, `variance` |
| `tc`, `dc`, `di` | `variance`, `beta` (system identification kernels) |
| `ss1`, `ss2`, `hfss` | `variance`, `beta` (stable spline kernels) |

## Usage

```python
from src.gpr.kernels import create_kernel
from src.gpr.gpr_fitting import GaussianProcessRegressor

kernel = create_kernel("rbf", length_scale=1.0, variance=1.0)
gp = GaussianProcessRegressor(kernel=kernel, noise_variance=1e-6)
gp.fit(X_train, y_train, optimize=True, use_grid_search=True)
mean, var = gp.predict(X_test)
```

## ITGP (Outlier-Robust GP)

The ITGP method iteratively trims outlier data points and refits the GP:

```python
from src.gpr.itgp import run_itgp_pipeline
results = run_itgp_pipeline(data_dir="data/gp_training/")
```

## T-Distribution GP

Uses GPflow's Variational GP with Student-t likelihood for robustness to
heavy-tailed noise:

```python
from src.gpr.t_distribution import run_t_distribution_pipeline
results = run_t_distribution_pipeline(data_dir="data/gp_training/")
```

## Data Preprocessing

- Frequency inputs are log-transformed: `X = log10(omega)`
- Features are standardized with `sklearn.preprocessing.StandardScaler`
- GP modes: `separate` (fit Re/Im independently) or `polar` (fit log-magnitude/phase)

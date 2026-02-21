# classical_methods/ -- Classical System Identification

[日本語版はこちら](../../docs/ja/src/classical_methods/README.md)

## Purpose

Provides classical frequency-domain system identification methods and
machine-learning regressors as alternatives to Gaussian Process regression.

## Files

| File | Description |
|---|---|
| `frequency_domain.py` | Parametric estimators that fit rational transfer functions H(jw) = N(jw)/D(jw) |
| `ml_methods.py` | ML regressors (RF, GBR, SVM) and local polynomial/rational methods (LPM, LRMP), plus `create_estimator()` factory |

## Frequency-Domain Estimators

All estimators model H(jw) as a ratio of polynomials and share a common
`FrequencyDomainEstimator` base class.

| Estimator | Method |
|---|---|
| `NLSEstimator` | Nonlinear Least Squares |
| `LSEstimator` | Linear Least Squares (Levy's method) |
| `IWLSEstimator` | Iteratively Weighted Least Squares (Sanathanan-Koerner) |
| `TLSEstimator` | Total Least Squares with Euclidean norm constraint |
| `MLEstimator` | Maximum Likelihood for complex Gaussian noise |
| `LOGEstimator` | Logarithmic Least Squares |

## ML and Local Methods

| Estimator | Method |
|---|---|
| `LPMEstimator` | Local Polynomial Method for nonparametric FRF estimation |
| `LRMPEstimator` | Local Rational Method with prior poles |
| `RFEstimator` | Random Forest regressor |
| `GBREstimator` | Gradient Boosting regressor |
| `SVMEstimator` | Support Vector Machine regressor |

## Usage

```python
from src.classical_methods.ml_methods import create_estimator

# Classical frequency-domain
estimator = create_estimator("nls", n_numerator=2, n_denominator=3)
estimator.fit(omega, H_measured)
H_pred = estimator.predict(omega_grid)

# Machine-learning
estimator = create_estimator("rf", n_estimators=100)
estimator.fit(omega, H_measured)
H_pred = estimator.predict(omega_grid)
```

## When to Use

- **NLS/ML**: Best accuracy when model order is known
- **LS/IWLS**: Fast, good initial estimates
- **TLS**: When both input and output have measurement noise
- **LPM/LRMP**: Nonparametric smoothing without assuming model order
- **RF/GBR/SVM**: Data-driven alternatives; useful for benchmarking

## Results with Default Parameters

### Default Configuration

| Parameter | Value |
|:---|:---|
| Model structure | G(jw) = N(jw) / D(jw) |
| Numerator degree (n_b) | 2 |
| Denominator degree (n_a) | 4 |
| N_d (frequency points) | 50 |
| T (observation duration) | 1 hour |

The rational function model structure (n_b = 2, n_a = 4) is **physically derived** from the Quanser Rotary Flexible Link dynamics -- the plant has 2 zeros and 4 poles.

### Comparison with GPR (N_d = 50, T = 1 hour)

| Method | Model Order | Multisine RMSE (x10^-2 rad) | Square Wave RMSE (x10^-2 rad) |
|:---|:---:|:---:|:---:|
| LS (Least Squares) | n_b=2, n_a=4 | 9.79 | 26.9 |
| **NLS (Nonlinear LS)** | n_b=2, n_a=4 | **2.75** | **5.77** |
| GPR (Matern-5/2) | -- | 2.90 | 5.89 |

- **NLS** achieves the best overall accuracy but **requires** knowing the correct model order (n_b, n_a)
- **LS** (Levy's method) is fast but biased toward high-frequency errors
- **GPR** achieves comparable accuracy to NLS **without** any parametric model structure -- a key advantage for unknown systems

See [gpr/README.md](../gpr/README.md#kernel-comparison-nd--50-t--1-hour) for the full 11-kernel comparison table.

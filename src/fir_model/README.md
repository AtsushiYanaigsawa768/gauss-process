# fir_model/ -- FIR Model Identification

[日本語版はこちら](../../docs/ja/src/fir_model/README.md)

## Purpose

Converts GP-predicted frequency response functions into Finite Impulse Response
(FIR) filter coefficients and validates the resulting models against recorded
time-domain input/output data.

## Files

| File | Description |
|---|---|
| `fir_fitting.py` | Main GP-to-FIR pipeline: uniform omega grid, Hermitian-symmetric IDFT, time-domain validation |
| `fir_helpers.py` | Low-level DFT utilities: uniform omega grid construction, two-sided Hermitian extension, IDFT |
| `fir_validation.py` | Time-domain FIR validation: convolve FIR with input, compute RMSE/NRMSE/R2 |
| `fir_legacy.py` | Legacy IRFFT-based FIR extraction (preserved for backward compatibility) |
| `kernel_regularized.py` | Kernel-regularized FIR identification with DC, SS(2), and SI kernels via Tikhonov/GP regularization |
| `lms_filter.py` | LMS (Least Mean Squares) adaptive FIR identification |
| `rls_filter.py` | RLS (Recursive Least Squares) real-time FIR identification with live plotting |
| `partial_update_lms.py` | Partial-update LMS variant for reduced computation |

## GP-to-FIR Pipeline

The main pipeline (`fir_fitting.py`) performs:
1. Interpolate GP-predicted G(jw) onto a uniform omega grid
2. Build two-sided Hermitian-symmetric spectrum
3. Compute impulse response via IDFT
4. Validate by convolving FIR with recorded input u(t), comparing against y(t)

## Kernel-Regularized FIR

The `kernel_regularized.py` module implements time-domain FIR identification
using Tikhonov regularization with GP-motivated kernel matrices:

| Kernel | Description |
|---|---|
| DC | Diagonal Correlated -- exponential decay |
| SS | Second-order Stable Spline |
| SI | First-order Stable Spline (integrator-like) |

Hyperparameters are selected by maximizing the GP marginal likelihood.

## Usage

```bash
# Kernel-regularized FIR from CLI
python -m src.fir_model.kernel_regularized --io data/sample_data/input_test_20250913_010037.mat \
    --kernel dc --out output/

# LMS adaptive filter
python -m src.fir_model.lms_filter

# RLS real-time identification
python -m src.fir_model.rls_filter
```

## Metrics

All FIR validation routines report: RMSE, NRMSE, FIT (%), and R2.

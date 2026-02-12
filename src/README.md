# src/ -- Source Package

[日本語版はこちら](../docs/ja/src/README.md)

## Overview

This package implements system identification for a flexible link mechanism.
The root entry point is `main.py`, which calls `src.pipeline.unified_pipeline.main()`.

## Module Map

| Module | Purpose |
|---|---|
| `utils/` | Shared utilities -- Hampel filter, Bode/MAT data loaders |
| `visualization/` | Publication-quality plot helpers for I/O signals and styles |
| `frequency_transform/` | FRF estimation (synchronous demodulation and FFT), disk cache |
| `gpr/` | Gaussian Process Regression -- 14 kernels, ITGP, T-distribution GP |
| `fir_model/` | FIR identification -- GP-to-FIR pipeline, kernel-regularized, LMS, RLS |
| `classical_methods/` | Classical system ID (NLS, LS, IWLS, TLS, ML) and ML regressors |
| `pipeline/` | End-to-end pipeline with CLI, configuration dataclasses, batch testing |
| `examples/` | Example scripts (`run_all_gp.py`, `paper_mode_usage.py`, `gp_sample.py`) |
| `tests/` | Unit and integration tests (`test_pipeline.py`, `test_fourier.py`) |

## Data Flow

```
.mat time-series --> frequency_transform/ --> FRF DataFrame
                                                  |
                                                  v
data/gp_training/*.dat --> gpr/ --> Smoothed G(jw) --> fir_model/ --> FIR coefficients
                                                                         |
                                                                         v
                                                              Time-domain validation
```

## Quick Start

```bash
# Full pipeline (frequency estimation + GP + optional FIR)
python main.py data/sample_data/*.mat --kernel rbf --out-dir output

# With FIR extraction
python main.py data/sample_data/*.mat --kernel rbf --extract-fir --fir-length 1024 \
    --fir-validation-mat data/sample_data/input_test_20250913_010037.mat --out-dir output

# Use existing frequency response data
python main.py --use-existing output/frf.csv --kernel matern --nu 2.5
```

## Dependencies

Core: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`
GP-specific: `gpflow`, `tensorflow`, `robustgp`

See `requirements.txt` in the repository root.

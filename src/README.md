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

## Default Parameters (Paper Baseline)

| Parameter | Value |
|:---|:---|
| Plant | Quanser Rotary Flexible Link |
| Controller | P-controller (K_p = 1.65) |
| Sampling rate | 500 Hz (dt = 0.002 s) |
| Frequency range | [0.1, 250] Hz (log-spaced) |
| N_d (frequency points) | 50 |
| T (observation duration) | 1 hour |
| Best GPR kernel | Matern-5/2 |
| FIR length | 1024 taps |
| Training signal | Multisine |
| Validation signal | Random square wave |

<p align="center">
<img src="../docs/images/control_block_diagram.jpg" alt="Control block diagram" width="450"><br>
<em>Closed-loop feedback control system (P-controller, K_p = 1.65)</em>
</p>

## Summary Results

| | Finding |
|:---|:---|
| **Best GPR kernel** | Matern-5/2: RMSE = 0.0290 (multisine), 0.0589 (square wave) |
| **Best classical method** | NLS: RMSE = 0.0275 (multisine), 0.0577 (square wave) |
| **Key advantage** | GPR matches NLS accuracy **without** parametric model structure |
| **Sparse data winner** | DI kernel excels at N_d <= 30 |
| **Most robust** | RBF and SS1 maintain stable accuracy across all observation durations |

<p align="center">
<img src="../docs/images/flexlink.jpg" alt="Quanser Rotary Flexible Link" width="400"><br>
<em>Quanser Rotary Flexible Link -- experimental apparatus</em>
</p>

See individual module READMEs for detailed results:
- [gpr/](gpr/) -- Kernel comparison tables, N_d and T effect analysis
- [fir_model/](fir_model/) -- Time-domain FIR validation
- [frequency_transform/](frequency_transform/) -- FRF estimation output (Bode/Nyquist)
- [classical_methods/](classical_methods/) -- LS/NLS comparison
- [visualization/](visualization/) -- I/O signal examples
- [pipeline/](pipeline/) -- End-to-end pipeline results

## Dependencies

Core: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`
GP-specific: `gpflow`, `tensorflow`, `robustgp`

See `requirements.txt` in the repository root.

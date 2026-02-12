# frequency_transform/ -- Frequency Response Estimation

[日本語版はこちら](../../docs/ja/src/frequency_transform/README.md)

## Purpose

Estimates the Frequency Response Function (FRF) G(jw) from time-domain `.mat`
recordings using two methods, with optional disk-based caching.

## Files

| File | Description |
|---|---|
| `frf_estimator.py` | Synchronous demodulation on a logarithmic frequency grid (MATLAB-compatible) |
| `fourier_estimator.py` | FFT-based estimation on a linear frequency grid with windowing |
| `transform.py` | Unified interface -- dispatches to either estimator, optional caching |
| `cache.py` | `FrequencyDataCache` -- SHA-256 keyed disk cache storing DataFrames as CSV |

## Two Estimation Methods

| Feature | FRF (`frf`) | Fourier (`fourier`) |
|---|---|---|
| Frequency grid | Logarithmic (MATLAB-compatible) | Linear |
| Core algorithm | Trapezoidal synchronous demodulation | FFT with configurable window |
| Aggregation | Cross-power average G = sum(Y*conj(U)) / sum(\|U\|^2) | Complex mean of interpolated estimates |
| Non-uniform dt | Handled natively | Requires uniform sampling |

## Usage

```python
from src.frequency_transform.transform import estimate_frequency_response

# FRF method (default, log grid)
df = estimate_frequency_response(
    mat_files=["data/sample_data/*.mat"],
    method="frf", nd=100, n_files=3
)

# Fourier method (linear grid)
df = estimate_frequency_response(
    mat_files=["data/sample_data/*.mat"],
    method="fourier", nd=200, window="hann"
)

# With caching enabled
df = estimate_frequency_response(
    mat_files=["data/sample_data/*.mat"],
    method="frf", use_cache=True
)
```

## Output Format

Both methods return a `pandas.DataFrame` with columns:

| Column | Unit |
|---|---|
| `omega_rad_s` | rad/s |
| `freq_Hz` | Hz |
| `ReG`, `ImG` | -- |
| `absG` | -- |
| `phase_rad` | rad |

## Cache

Cache files are stored in `.cache/frequency_data/` by default. Each entry is
keyed by a truncated SHA-256 hash of the input configuration (file list,
frequency parameters, estimation method). Use `cache.invalidate()` to clear.

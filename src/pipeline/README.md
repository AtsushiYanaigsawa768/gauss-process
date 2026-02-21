# pipeline/ -- End-to-End Pipeline

[日本語版はこちら](../../docs/ja/src/pipeline/README.md)

## Purpose

Orchestrates the full system identification workflow: frequency response
estimation, GP regression, and optional FIR extraction, all from a single
command line.

## Files

| File | Description |
|---|---|
| `unified_pipeline.py` | CLI entry point (`main()`) with argparse-based argument parsing |
| `gp_pipeline.py` | Core pipeline logic: FRF estimation, GP fitting, FIR extraction |
| `config.py` | Dataclass configurations: `GPConfig`, `FIRConfig`, `FrequencyConfig` |
| `data_loader.py` | Data loading and preprocessing utilities for the pipeline |
| `comprehensive_test.py` | Batch testing across multiple kernel/parameter combinations |
| `gp_fir_legacy.py` | Legacy GP-to-FIR pipeline (preserved for backward compatibility) |

## Configuration

Three dataclass groups control pipeline behavior:

| Config | Key Fields |
|---|---|
| `GPConfig` | `kernel_type`, `noise_variance`, `optimize`, `gp_mode` (`separate`/`polar`) |
| `FIRConfig` | `extract_fir`, `fir_length`, `validation_mat` |
| `FrequencyConfig` | `n_files`, `time_duration`, `nd`, `freq_method` (`frf`/`fourier`) |

## CLI Usage

```bash
# Basic GP fitting
python main.py data/sample_data/*.mat --kernel rbf --out-dir output

# Matern kernel with grid search
python main.py data/sample_data/*.mat --kernel matern --nu 2.5 --grid-search --out-dir output

# Full pipeline with FIR extraction
python main.py data/sample_data/*.mat --n-files 1 --nd 100 \
    --kernel rbf --normalize --log-frequency \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat data/sample_data/input_test_20250913_010037.mat \
    --out-dir output_complete

# FFT-based frequency estimation
python main.py data/sample_data/*.mat --freq-method fourier --nd 200 \
    --kernel rbf --out-dir output_fourier

# Use pre-computed FRF data
python main.py --use-existing output/frf.csv --kernel rbf --out-dir output
```

## Pipeline Stages

1. **Frequency estimation** -- compute G(jw) from `.mat` time-series (or load existing CSV)
2. **GP regression** -- fit GP to frequency response, predict on dense grid
3. **FIR extraction** (optional) -- convert GP output to FIR, validate in time domain

## Output

The pipeline writes to `--out-dir`:
- FRF data (CSV and MAT)
- GP predictions (CSV)
- Bode and Nyquist plots (PNG)
- FIR coefficients and validation metrics (when `--extract-fir` is set)

## Results with Default Parameters

### Paper Baseline Command

The following command reproduces the paper's baseline result (Matern-5/2, N_d = 50, T = 1 hour):

```bash
python main.py data/sample_data/*.mat --kernel matern --nu 2.5 \
    --normalize --log-frequency --nd 50 --n-files 1 \
    --extract-fir --fir-length 1024 \
    --fir-validation-mat data/sample_data/input_test_20250913_010037.mat \
    --out-dir output
```

### End-to-End Results

| Stage | Output | Key Metric |
|:---|:---|:---|
| FRF Estimation | 50 frequency points, [0.1, 250] Hz | Synchronous demodulation, log-spaced grid |
| GP Regression | Smoothed G(jw) with +/-2 sigma bands | Best kernel: Matern-5/2 |
| FIR Validation | 1024-tap FIR coefficients | RMSE = 0.0290 rad (multisine), 0.0589 rad (square wave) |

<p align="center">
<img src="../../docs/images/control_block_diagram.jpg" alt="Control block diagram" width="450"><br>
<em>Closed-loop feedback control system (P-controller, K_p = 1.65)</em>
</p>

### Comprehensive Test Summary

`comprehensive_test.py` evaluates all 11 kernels + LS/NLS across multiple N_d (10, 30, 50, 100) and T (10 min, 30 min, 60 min, 600 min) combinations. Results are consistent with the paper's Tables I--III:

- **Best GPR kernel**: Matern-5/2 (RMSE = 0.0290, N_d = 50, T = 60 min)
- **Best classical**: NLS (RMSE = 0.0275, requires model order n_b = 2, n_a = 4)
- **Most robust**: RBF and SS1 (stable across all observation durations)

See [gpr/README.md](../gpr/README.md#results-with-default-parameters) for detailed kernel comparison tables.

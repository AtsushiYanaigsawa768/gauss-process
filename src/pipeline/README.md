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

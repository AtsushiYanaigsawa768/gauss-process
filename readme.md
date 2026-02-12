# Flexible Link System Identification

System identification for a flexible link mechanism using Gaussian Process Regression (GP) and FIR models.

The pipeline estimates the frequency response (transfer function) from experimental data using GP regression, then converts it to a Finite Impulse Response (FIR) model for time-domain prediction.

[Japanese documentation / 日本語版ドキュメント](docs/ja/src/README.md)

## Setup

```bash
conda create --name GaussProcess python=3.11
conda activate GaussProcess
git clone https://github.com/AtsushiYanaigsawa768/gauss_process.git
cd gauss_process
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full GP + FIR pipeline
python main.py input/*.mat --kernel rbf --normalize --log-frequency --nd 50 \
  --extract-fir --fir-length 1024 --out-dir gp_output

# Show all options
python main.py --help
```

## Project Structure

```
gauss_process/
├── main.py                  Entry point
├── requirements.txt         Dependencies
├── src/                     All source code
│   ├── utils/               Shared utilities (Hampel filter, data I/O)
│   ├── visualization/       Plot styles and I/O signal plotting
│   ├── frequency_transform/ FRF / FFT estimation with disk cache
│   ├── gpr/                 GP regression (14 kernels, grid search, ITGP, etc.)
│   ├── fir_model/           FIR extraction, validation, LMS/RLS filters
│   ├── classical_methods/   NLS, LS, IWLS, TLS, ML, RF, GBR, SVM
│   ├── pipeline/            End-to-end pipeline and configuration
│   ├── examples/            Example scripts
│   └── tests/               Test scripts
├── data/
│   ├── sample_data/         10 .mat files (1-hour recordings)
│   └── gp_training/         4 .dat files (frequency response data)
├── input/                   Raw experimental .mat files
├── docs/ja/                 Japanese documentation
├── flexible_link/           MATLAB/Simulink models
└── paper_figures/           Figure generation for publications
```

See [src/README.md](src/README.md) for detailed module documentation.

## Data Flow

```
Experimental .mat files (time-domain input/output)
        |
        v
Frequency Response Estimation (FRF or FFT)
        |
        v
GP Regression (smooth transfer function with uncertainty)
        |
        v
FIR Model Extraction (IDFT of GP-predicted spectrum)
        |
        v
Time-domain Validation (RMSE, FIT%, R2)
```

## Key Features

- **14 GP kernels**: RBF, Matern (3/2, 5/2), Rational Quadratic, Exponential, TC, DC, DI, Stable Spline (1st/2nd/high-freq), and combinations
- **Grid search**: Automatic hyperparameter optimization with validation
- **Robust methods**: ITGP (outlier-robust GP), T-distribution GP
- **FIR extraction**: Paper-mode (uniform grid + Hermitian + IDFT) and legacy (IRFFT)
- **Kernel-regularized FIR**: DC, SS2, SI kernels with marginal likelihood optimization
- **Classical methods**: NLS, LS, IWLS, TLS, ML estimators
- **ML methods**: Random Forest, Gradient Boosting, SVM regressors
- **Disk caching**: SHA-256 based cache for frequency-domain computations

## References

- ITGP: [Robust Gaussian Process Regression](https://arxiv.org/abs/2011.11057) (robustgp library)
- Kernel-regularized FIR: DC, SS2, SI kernels for system identification

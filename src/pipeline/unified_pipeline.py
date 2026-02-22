#!/usr/bin/env python3
"""
unified_pipeline.py

Command-line entry point for the system identification pipeline.

Usage examples:
    python -m src.pipeline.unified_pipeline data/sample_data/*.mat --kernel rbf --out-dir gp_output
    python -m src.pipeline.unified_pipeline --use-existing output/matched_frf.csv --kernel matern --nu 2.5
    python -m src.pipeline.unified_pipeline data/sample_data/*.mat --n-files 1 --nd 50 \\
        --kernel rbf --normalize --log-frequency \\
        --extract-fir --fir-length 1024 \\
        --fir-validation-mat data/sample_data/input_test_20250912_165937.mat \\
        --out-dir output_complete
    python -m src.pipeline.unified_pipeline data/sample_data/*.mat --n-files 1 --nd 100 --freq-method fourier \\
        --kernel rbf --normalize --extract-fir --fir-length 1024 \\
        --fir-validation-mat data/sample_data/input_test_20250912_165937.mat \\
        --out-dir output_fourier
    python -m src.pipeline.unified_pipeline data/sample_data/*.mat --n-files 1 --nd 100 \\
        --kernel rbf --normalize --grid-search \\
        --fir-validation-mat data/sample_data/input_test_20250912_165937.mat \\
        --extract-fir --fir-length 1024 \\
        --out-dir output_grid_search
"""

from __future__ import annotations

import argparse

from src.pipeline.gp_pipeline import run_gp_pipeline


def main() -> None:
    """Parse CLI arguments and run the GP pipeline."""

    parser = argparse.ArgumentParser(
        description="Unified pipeline for frequency response analysis and GP regression",
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'mat_files', nargs='*', default=[],
        help='MAT files for frequency response analysis',
    )
    input_group.add_argument(
        '--use-existing', type=str,
        help='Use existing FRF CSV file instead of running frequency_response.py',
    )

    # Frequency response options
    parser.add_argument(
        '--n-files', type=int, default=1,
        help='Number of MAT files to process (default: 1)',
    )
    parser.add_argument(
        '--time-duration', type=float, default=None,
        help='Time duration in seconds to use from each file (only with --n-files 1)',
    )
    parser.add_argument(
        '--nd', type=int, default=100,
        help='Number of frequency points (N_d) for frequency response analysis (default: 100)',
    )
    parser.add_argument(
        '--freq-method', type=str, default='frf',
        choices=['frf', 'fourier'],
        help='Frequency analysis method: frf or fourier (default: frf)',
    )

    # Method selection
    parser.add_argument(
        '--method', type=str, default='gp',
        choices=(
            ['gp']
            + ['nls', 'ls', 'iwls', 'tls', 'ml', 'log', 'lpm', 'lrmp']
            + ['rf', 'gbr', 'svm']
        ),
        help='System identification method (default: gp)',
    )

    # GP options
    parser.add_argument(
        '--kernel', type=str, default='rbf',
        choices=[
            'rbf', 'matern', 'matern12', 'matern32', 'matern52', 'rq',
            'exp', 'tc', 'dc', 'di', 'ss1', 'ss2', 'sshf', 'stable_spline',
        ],
        help='Kernel type for GP method (default: rbf)',
    )
    parser.add_argument('--nu', type=float, default=None,
                        help='Nu parameter for Matern kernel (default: 1.5)')
    parser.add_argument(
        '--gp-mode', type=str, default='separate',
        choices=['separate', 'polar'],
        help='GP mode: separate (real/imag) or polar (mag/phase)',
    )
    parser.add_argument('--noise-variance', type=float, default=1e-6,
                        help='Initial noise variance (default: 1e-6)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalise inputs and outputs')
    parser.add_argument('--log-frequency', action='store_true',
                        help='Use log-frequency as GP input')
    parser.add_argument('--optimize', action='store_true', default=False,
                        help='Optimise hyperparameters (default: False)')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false',
                        help='Disable hyperparameter optimisation')
    parser.add_argument('--n-restarts', type=int, default=3,
                        help='Number of optimisation restarts (default: 3)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Use grid search for hyperparameter tuning')
    parser.add_argument('--grid-search-max-combinations', type=int, default=5000,
                        help='Maximum grid combinations before random sampling (default: 5000)')
    parser.add_argument('--validation-mat', type=str, default=None,
                        help='[DEPRECATED] Use --fir-validation-mat instead')

    # Classical/ML method options
    parser.add_argument('--n-numerator', type=int, default=2,
                        help='Numerator order for classical methods (default: 2)')
    parser.add_argument('--n-denominator', type=int, default=4,
                        help='Denominator order for classical methods (default: 4)')

    # Output options
    parser.add_argument('--out-dir', type=str, default='gp_output',
                        help='Output directory (default: gp_output)')

    # FIR extraction options
    parser.add_argument('--extract-fir', action='store_true',
                        help='Extract FIR model coefficients')
    parser.add_argument('--fir-length', type=int, default=1024,
                        help='FIR filter length (default: 1024)')
    parser.add_argument(
        '--fir-validation-mat', type=str, default=None,
        help='MAT file with [time, output, input] for FIR validation AND GP grid search validation',
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.use_existing and not args.mat_files:
        parser.error("Either provide MAT files or use --use-existing")

    if args.time_duration is not None:
        if args.time_duration <= 0:
            parser.error("--time-duration must be positive")
        if args.n_files != 1:
            parser.error("--time-duration only works with --n-files 1")

    if args.nd <= 0:
        parser.error("--nd must be positive")

    # Add method info
    args.is_gp = args.method == 'gp'
    args.use_grid_search = args.grid_search

    # Backward compatibility: deprecated --validation-mat
    if args.validation_mat is not None and args.fir_validation_mat is None:
        print("Warning: --validation-mat is deprecated. Use --fir-validation-mat instead.")
        args.fir_validation_mat = args.validation_mat

    # Run pipeline
    run_gp_pipeline(args)


if __name__ == '__main__':
    main()

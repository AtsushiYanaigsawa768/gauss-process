#!/usr/bin/env python3
"""
data_loader.py

Data-loading and frequency-response generation utilities for the pipeline.

Functions:
- load_frf_data:                     Load a FRF CSV into a DataFrame.
- run_frequency_response:            Invoke ``frequency_response.py`` via subprocess.
- run_fourier_transform:             Invoke ``fourier_transform.py`` via subprocess.
- generate_validation_data_from_mat: Compute validation FRF directly from a MAT file.

All functions preserve the exact numerical behaviour of the original
``unified_pipeline.py`` implementations.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =====================
# CSV Loader
# =====================

def load_frf_data(frf_file: Path) -> pd.DataFrame:
    """Load frequency response function data from a CSV file.

    The CSV must contain at least the columns:
    ``omega_rad_s``, ``ReG``, ``ImG``, ``absG``, ``phase_rad``.

    Args:
        frf_file: Path to the CSV file.

    Returns:
        A DataFrame with the frequency response data.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(frf_file)
    required_cols = ['omega_rad_s', 'ReG', 'ImG', 'absG', 'phase_rad']

    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"FRF file must contain columns: {required_cols}")

    return df


# =====================
# Subprocess Runners
# =====================

def run_frequency_response(
    mat_files: List[str],
    output_dir: Path,
    n_files: int = 1,
    time_duration: Optional[float] = None,
    nd: int = 100,
) -> Path:
    """Run ``frequency_response.py`` as a subprocess and return the output CSV path.

    Args:
        mat_files:     List of input MAT file paths.
        output_dir:    Directory for output files (created if missing).
        n_files:       Number of MAT files to process.
        time_duration: Time duration in seconds (only effective when *n_files* is 1).
        nd:            Number of frequency grid points.

    Returns:
        Path to the generated ``unified_frf.csv``.

    Raises:
        RuntimeError: If the subprocess exits with non-zero status or the
                      expected output file is not created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert mat_files to absolute paths
    mat_files_str = [str(Path(f).resolve()) for f in mat_files]

    cmd = [
        sys.executable,
        'src/frequency_response.py',
        *mat_files_str,
        '--recursive',
        '--n-files', str(n_files),
        '--out-dir', str(output_dir),
        '--out-prefix', 'unified',
        '--nd', str(nd),
    ]

    if time_duration is not None and n_files == 1:
        cmd.extend(['--time-duration', str(time_duration)])

    print(f"Running frequency_response.py with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running frequency_response.py:\n{result.stderr}")
        raise RuntimeError(f"frequency_response.py failed with code {result.returncode}")

    frf_csv = output_dir / 'unified_frf.csv'
    if not frf_csv.exists():
        raise RuntimeError(f"Expected output file not found: {frf_csv}")

    return frf_csv


def run_fourier_transform(
    mat_files: List[str],
    output_dir: Path,
    n_files: int = 1,
    time_duration: Optional[float] = None,
    nd: int = 100,
) -> Path:
    """Run ``fourier_transform.py`` as a subprocess and return the output CSV path.

    Args:
        mat_files:     List of input MAT file paths.
        output_dir:    Directory for output files (created if missing).
        n_files:       Number of MAT files to process.
        time_duration: Time duration in seconds.
        nd:            Number of frequency grid points.

    Returns:
        Path to the generated ``unified_fft.csv``.

    Raises:
        RuntimeError: If the subprocess exits with non-zero status or the
                      expected output file is not created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files_str = [str(Path(f).resolve()) for f in mat_files]

    cmd = [
        sys.executable,
        'src/fourier_transform.py',
        *mat_files_str,
        '--recursive',
        '--out-dir', str(output_dir),
        '--out-prefix', 'unified',
        '--nd', str(nd),
    ]

    if n_files is not None:
        cmd.extend(['--n-files', str(n_files)])

    if time_duration is not None:
        cmd.extend(['--time-duration', str(time_duration)])

    print(f"Running fourier_transform.py with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running fourier_transform.py:\n{result.stderr}")
        raise RuntimeError(f"fourier_transform.py failed with code {result.returncode}")

    fft_csv = output_dir / 'unified_fft.csv'
    if not fft_csv.exists():
        raise RuntimeError(f"Expected output file not found: {fft_csv}")

    return fft_csv


# =====================
# Validation Data
# =====================

def generate_validation_data_from_mat(
    mat_file: str,
    nd: int = 200,
    freq_method: str = 'frf',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate validation data from a test MAT file for grid search evaluation.

    This function always returns raw (non-normalised) data. Normalisation should
    be handled separately during GP training and prediction.

    Args:
        mat_file:    Path to the ``.mat`` file.
        nd:          Number of frequency points to generate.
        freq_method: Frequency analysis method (``'frf'`` or ``'fourier'``).

    Returns:
        Tuple of ``(omega, G_real, G_imag)`` where:
        - omega:  Angular frequencies (shape ``(nd,)``) -- raw scale.
        - G_real: Real part of the transfer function (shape ``(nd,)``) -- raw scale.
        - G_imag: Imaginary part of the transfer function (shape ``(nd,)``) -- raw scale.
    """
    from src.frequency_transform.frf_estimator import (
        load_time_u_y,
        synchronous_coefficients_trapz,
        frf_cross_power_average,
        matlab_freq_grid,
    )

    # Convert to absolute path
    mat_file_path = Path(mat_file).resolve()
    print(f"  Generating validation data from {mat_file_path}...")

    # Load time-series data from MAT file (FULL DURATION)
    t, u, y = load_time_u_y(mat_file_path, y_col=0)

    # Generate frequency grid
    if freq_method == 'frf':
        # Log-scale frequency grid (FRF method)
        _, omega = matlab_freq_grid(nd, f_low_log10=-1.0, f_up_log10=2.3)
    else:
        # Linear frequency grid (Fourier method)
        dt = np.median(np.diff(t))
        freqs = np.fft.rfftfreq(len(t), dt)
        omega = 2.0 * np.pi * freqs[:nd]

    # Compute synchronous demodulation coefficients
    subtract_mean = True
    drop_seconds = 0.0  # Use all data from start

    U = synchronous_coefficients_trapz(t, u, omega, drop_seconds, subtract_mean)
    Y = synchronous_coefficients_trapz(t, y, omega, drop_seconds, subtract_mean)

    # Compute FRF using cross-power average (single-file case: Y/U)
    G = frf_cross_power_average([U], [Y])

    # Extract real and imaginary parts
    G_real = np.real(G)
    G_imag = np.imag(G)

    # Diagnostic output
    time_duration = t[-1] - t[0]
    print(f"  Validation data generated: {nd} frequency points")
    print(f"  Time data: {len(t)} samples, duration={time_duration:.1f}s (FULL)")
    print(f"  Frequency range: {omega[0]/(2*np.pi):.3f} - {omega[-1]/(2*np.pi):.3f} Hz")
    print(f"  NOTE: This is the SAME data used for FIR model time-domain evaluation")

    return omega, G_real, G_imag

#!/usr/bin/env python3
"""
fir_legacy.py

Legacy IRFFT-based GP-to-FIR conversion.

This module implements the original (pre-paper-mode) approach:
  1. GP/cubic interpolation to a uniform frequency grid.
  2. Hermitian-symmetric spectrum via ifftshift + ifft.
  3. Hanning-windowed truncation of the impulse response.
  4. Optional validation against time-domain data from a MAT file.

Preserved verbatim from gp_to_fir_direct.py with all Japanese comments
translated to English.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ------------------------------------------------------------------ #
# Step 1: GP interpolation to uniform grid
# ------------------------------------------------------------------ #

def gp_interpolate_to_uniform_grid(
    omega: np.ndarray,
    G: np.ndarray,
    gp_predict_func: Optional[Callable] = None,
    N_d: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use GP interpolation to create a uniform frequency grid.

    Args:
        omega:           Original angular frequencies from GP (may be non-uniform).
        G:               Complex frequency response at *omega* (from GP).
        gp_predict_func: Optional GP prediction function for interpolation.
        N_d:             Number of frequency points for the uniform grid.

    Returns:
        omega_uniform: Uniformly spaced angular frequencies.
        G_uniform:     GP-interpolated complex frequency response.
    """
    omega_min = np.min(omega)
    omega_max = np.max(omega)

    if N_d is None:
        N_d = len(omega)

    # Create uniform grid in linear frequency space
    Delta_omega = (omega_max - omega_min) / (N_d - 1)
    omega_uniform = omega_min + np.arange(N_d) * Delta_omega

    print(f"GP interpolation to uniform grid: {N_d} points "
          f"from {omega_min:.2f} to {omega_max:.2f} rad/s")

    if gp_predict_func is not None:
        # Use provided GP prediction function
        G_uniform = gp_predict_func(omega_uniform)
    else:
        # Fallback: cubic interpolation on GP output
        from scipy.interpolate import interp1d

        interp_real = interp1d(omega, np.real(G), kind='cubic',
                               fill_value='extrapolate')
        interp_imag = interp1d(omega, np.imag(G), kind='cubic',
                               fill_value='extrapolate')

        G_uniform = interp_real(omega_uniform) + 1j * interp_imag(omega_uniform)
        print("Note: Using cubic interpolation on GP output "
              "(GP prediction function not provided)")

    return omega_uniform, G_uniform


# ------------------------------------------------------------------ #
# Step 2: Hermitian-symmetric spectrum
# ------------------------------------------------------------------ #

def construct_hermitian_symmetric_spectrum(
    G_uniform: np.ndarray,
) -> np.ndarray:
    """
    Construct a symmetric spectrum satisfying G(-omega) = conj(G(omega)).
    This ensures the IFFT produces a real-valued impulse response.

    Args:
        G_uniform: Complex frequency response on uniform grid.

    Returns:
        G_symmetric: Symmetric spectrum of length 2*N_d - 1.
    """
    N_d = len(G_uniform)
    N_total = 2 * N_d - 1

    G_symmetric = np.zeros(N_total, dtype=complex)

    # Positive frequencies (including DC)
    G_symmetric[N_d - 1:] = G_uniform

    # Negative frequencies via Hermitian symmetry
    for m in range(1, N_d):
        G_symmetric[N_d - 1 - m] = np.conj(G_uniform[m])

    # Verify Hermitian symmetry
    symmetry_error = np.max(
        np.abs(G_symmetric[N_d - 2::-1] - np.conj(G_symmetric[N_d:])))
    print(f"Hermitian symmetry check: "
          f"max|G(-omega) - conj(G(omega))| = {symmetry_error:.2e}")

    return G_symmetric


# ------------------------------------------------------------------ #
# Step 3: FIR coefficients via IFFT
# ------------------------------------------------------------------ #

def calculate_fir_coefficients_via_ifft(
    G_symmetric: np.ndarray,
    fir_length: int = 1024,
) -> np.ndarray:
    """
    Calculate FIR coefficients via inverse FFT.

    Args:
        G_symmetric: Symmetric frequency spectrum.
        fir_length:  Desired FIR filter length (default: 1024).

    Returns:
        h: Real-valued FIR coefficients.
    """
    # Perform IFFT
    h_full = np.fft.ifft(np.fft.ifftshift(G_symmetric))

    # Extract real part (should be real due to Hermitian symmetry)
    imag_magnitude = np.max(np.abs(h_full.imag))
    if imag_magnitude > 1e-10:
        print(f"Warning: Imaginary part magnitude {imag_magnitude:.2e} "
              "(should be ~0)")
    h_full = np.real(h_full)

    # Shift to make causal
    h_full = np.fft.fftshift(h_full)

    # Extract center portion and pad/truncate to desired length
    center = len(h_full) // 2

    if fir_length <= len(h_full):
        start = center - fir_length // 2
        h = h_full[start:start + fir_length]
    else:
        h = np.zeros(fir_length)
        start = (fir_length - len(h_full)) // 2
        h[start:start + len(h_full)] = h_full

    # Apply Hanning window to reduce truncation artifacts
    window = np.hanning(fir_length)
    h = h * window

    print(f"FIR coefficients: length={fir_length}, "
          f"energy in first 100 taps: "
          f"{100 * np.sum(h[:100]**2) / np.sum(h**2):.1f}%")

    return h


# ------------------------------------------------------------------ #
# Validation against MAT data
# ------------------------------------------------------------------ #

def validate_fir_with_data(
    h: np.ndarray,
    mat_file: Path,
) -> Dict[str, float]:
    """
    Validate an FIR model against actual input/output data.

    Args:
        h:        FIR coefficients.
        mat_file: Path to MAT file with [time, output, input] columns.

    Returns:
        Dictionary with validation metrics and time-domain arrays.
    """
    data = loadmat(mat_file)

    # Find 3-column (or 3-row) data matrix
    mat_data = None
    for key in data.keys():
        if not key.startswith('__') and isinstance(data[key], np.ndarray):
            if data[key].shape[0] == 3 or data[key].shape[1] == 3:
                mat_data = data[key]
                break

    if mat_data is None:
        raise ValueError(f"Could not find data matrix in {mat_file}")

    if mat_data.shape[0] == 3:
        t = mat_data[0, :].ravel()
        y_actual = mat_data[1, :].ravel()
        u = mat_data[2, :].ravel()
    else:
        t = mat_data[:, 0].ravel()
        y_actual = mat_data[:, 1].ravel()
        u = mat_data[:, 2].ravel()

    # Remove DC bias
    u = u - np.mean(u)
    y_actual = y_actual - np.mean(y_actual)

    # Predict output using FIR model
    y_pred = np.convolve(u, h, mode='same')

    # Calculate metrics (skip initial transient)
    skip = len(h) // 2
    y_actual_valid = y_actual[skip:]
    y_pred_valid = y_pred[skip:]

    # RMSE
    rmse = np.sqrt(np.mean((y_actual_valid - y_pred_valid)**2))

    # R-squared
    ss_res = np.sum((y_actual_valid - y_pred_valid)**2)
    ss_tot = np.sum((y_actual_valid - np.mean(y_actual_valid))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # FIT percentage (normalized error)
    fit = 100 * (1 - np.linalg.norm(y_actual_valid - y_pred_valid)
                 / np.linalg.norm(y_actual_valid - np.mean(y_actual_valid)))

    return {
        'rmse': float(rmse),
        'r2': float(r2),
        'fit_percent': float(fit),
        't': t,
        'y_actual': y_actual,
        'y_pred': y_pred,
        'u': u,
    }


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #

def plot_gp_fir_results(
    h: np.ndarray,
    omega_uniform: np.ndarray,
    G_uniform: np.ndarray,
    validation_data: Dict,
    output_dir: Path,
    prefix: str = "gp_fir",
):
    """
    Create visualization plots for GP-based FIR model results.
    Shows output vs predicted and error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if 't' in validation_data:
        t = validation_data['t']
        y_actual = validation_data['y_actual']
        y_pred = validation_data['y_pred']
        error = y_actual - y_pred

        # Figure 1: Output vs Predicted
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        t_max = min(t[-1], t[0] + 200)
        mask = t <= t_max

        ax1.plot(t[mask], y_actual[mask], 'k-', label='Measured Output',
                 linewidth=1.5, alpha=0.8)
        ax1.plot(t[mask], y_pred[mask], 'r--', label='FIR Predicted',
                 linewidth=1.5)
        ax1.set_xlabel('Time [s]', fontsize=12)
        ax1.set_ylabel('Output', fontsize=12)
        ax1.set_title(
            f'FIR Model Validation '
            f'(RMSE={validation_data["rmse"]:.3e}, '
            f'FIT={validation_data["fit_percent"]:.1f}%, '
            f'R2={validation_data["r2"]:.3f})',
            fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_output_vs_predicted.png",
                    dpi=300)
        plt.close(fig1)

        # Figure 2: Error
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(t[mask], error[mask], 'b-', linewidth=1, alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_xlabel('Time [s]', fontsize=12)
        ax2.set_ylabel('Error (Measured - Predicted)', fontsize=12)
        ax2.set_title('FIR Model Prediction Error', fontsize=14)
        ax2.grid(True, alpha=0.3)

        error_stats = (f'Mean Error: {np.mean(error[mask]):.3e}\n'
                       f'Std Error: {np.std(error[mask]):.3e}')
        ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                           alpha=0.8))
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_error.png", dpi=300)
        plt.close(fig2)

    else:
        # No validation data: informational summary
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        info_text = (
            f"FIR Model Extraction Complete\n"
            f"================================\n"
            f"Method: GP-based interpolation\n"
            f"FIR Length: {len(h)}\n"
            f"Frequency Points: {len(omega_uniform)}\n"
            f"Frequency Range: {omega_uniform[0]:.2f} - "
            f"{omega_uniform[-1]:.2f} rad/s\n\n"
            f"No validation data provided.\n"
            f"To validate, specify --fir-validation-mat\n"
        )
        ax.text(0.5, 0.5, info_text, fontsize=12, family='monospace',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray",
                          alpha=0.8))
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_summary.png", dpi=300)
        plt.close()


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #

def gp_to_fir_direct_pipeline(
    omega: np.ndarray,
    G: np.ndarray,
    gp_predict_func: Optional[Callable] = None,
    mat_file: Path = None,
    output_dir: Path = None,
    N_d: int = None,
    fir_length: int = 1024,
) -> Dict:
    """
    Complete pipeline for GP-to-FIR conversion using direct GP interpolation.

    Args:
        omega:           Angular frequencies from GP.
        G:               Complex frequency response (GP-smoothed).
        gp_predict_func: Optional GP prediction function.
        mat_file:        Optional MAT file for validation.
        output_dir:      Output directory for results.
        N_d:             Number of uniform frequency points.
        fir_length:      FIR filter length (default: 1024).

    Returns:
        Dictionary with results and metrics.
    """
    if output_dir is None:
        output_dir = Path("gp_fir_direct_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if N_d is None:
        N_d = len(omega)

    print(f"=== GP-Based FIR Extraction (Direct Method) ===")
    print(f"Input: {len(omega)} GP-smoothed frequency points")
    print(f"Uniform grid: {N_d} points")
    print(f"FIR length: {fir_length}")

    # Step 1: GP interpolation to uniform grid
    print("\nStep 1: GP interpolation to uniform frequency grid...")
    omega_uniform, G_uniform = gp_interpolate_to_uniform_grid(
        omega, G, gp_predict_func, N_d)

    # Step 2: Construct symmetric spectrum
    print("\nStep 2: Constructing symmetric spectrum with "
          "Hermitian symmetry...")
    G_symmetric = construct_hermitian_symmetric_spectrum(G_uniform)

    # Step 3: Calculate FIR coefficients via IFFT
    print(f"\nStep 3: Calculating FIR coefficients "
          f"(length={fir_length})...")
    h = calculate_fir_coefficients_via_ifft(G_symmetric, fir_length)

    # Save FIR coefficients
    np.savez(output_dir / "fir_coefficients_gp.npz",
             h=h, omega=omega, G=G, omega_uniform=omega_uniform,
             G_uniform=G_uniform, fir_length=fir_length)

    results = {
        'fir_length': fir_length,
        'N_d': N_d,
        'omega_min': float(np.min(omega)),
        'omega_max': float(np.max(omega)),
        'method': 'gp_direct_interpolation',
    }

    # Validation if MAT file provided
    validation_data = {}
    if mat_file and mat_file.exists():
        print(f"\nValidating FIR model with data from {mat_file}...")
        validation_data = validate_fir_with_data(h, mat_file)
        results.update({
            'rmse': validation_data['rmse'],
            'r2': validation_data['r2'],
            'fit_percent': validation_data['fit_percent'],
        })

        print(f"\nValidation Results:")
        print(f"  RMSE: {validation_data['rmse']:.3e}")
        print(f"  R2:   {validation_data['r2']:.3f}")
        print(f"  FIT:  {validation_data['fit_percent']:.1f}%")

    # Create plots
    plot_gp_fir_results(h, omega_uniform, G_uniform,
                        validation_data, output_dir)

    # Save results
    with open(output_dir / "fir_gp_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GP to FIR using direct GP interpolation (legacy)")
    parser.add_argument('--gp-csv', type=str, required=True,
                        help='GP-smoothed FRF CSV file')
    parser.add_argument('--mat-file', type=str,
                        help='MAT file for validation')
    parser.add_argument('--out-dir', type=str,
                        default='gp_fir_direct_output')
    parser.add_argument('--fir-length', type=int, default=1024,
                        help='FIR length (default: 1024)')
    parser.add_argument('--n-uniform', type=int, default=None,
                        help='Number of uniform frequency points')

    args = parser.parse_args()

    df = pd.read_csv(args.gp_csv)
    omega = df['omega_rad_s'].values
    G = df['ReG'].values + 1j * df['ImG'].values

    mat_file = Path(args.mat_file) if args.mat_file else None
    results = gp_to_fir_direct_pipeline(
        omega, G,
        mat_file=mat_file,
        output_dir=Path(args.out_dir),
        N_d=args.n_uniform,
        fir_length=args.fir_length,
    )

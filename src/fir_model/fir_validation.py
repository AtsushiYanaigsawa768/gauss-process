#!/usr/bin/env python3
"""
fir_validation.py

Consolidated time-domain validation utilities for FIR models.

Provides:
- load_mat_time_series:       Load [t, y, u] from a MAT file.
- compute_time_domain_metrics: Compute RMSE, FIT%, R2 with transient skip.
- validate_fir_with_mat:       End-to-end FIR evaluation on MAT data.

The metric formulas exactly match the "strict evaluation" used in
gp_to_fir_direct_fixed.py and unified_pipeline.py (Wave.mat blocks),
consolidating the duplicated logic into one reusable function.
"""

from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
# MAT file loader
# ------------------------------------------------------------------ #

def load_mat_time_series(
    mat_path: str | Path,
) -> Dict[str, np.ndarray]:
    """
    Load [t, y, u] from a MAT file.

    Handles several common layouts:
      - Named variables 't'/'time', 'y', 'u'.
      - A single 3xN or Nx3 array whose rows/columns are [time, output, input].

    Args:
        mat_path: Path to the .mat file.

    Returns:
        Dictionary with keys 't', 'y', 'u' (1-D float arrays).

    Raises:
        ValueError: If the required data cannot be extracted.
    """
    mat_path = Path(mat_path)
    data = loadmat(mat_path)

    T = None
    y = None
    u = None

    # Strategy 1: look for a 3xN or Nx3 numeric array
    for key, val in data.items():
        if key.startswith("__") or not isinstance(val, np.ndarray):
            continue
        if val.ndim == 2 and (val.shape[0] == 3 or val.shape[1] == 3):
            if val.shape[0] == 3:
                T = np.ravel(val[0, :]).astype(float)
                y = np.ravel(val[1, :]).astype(float)
                u = np.ravel(val[2, :]).astype(float)
            else:
                T = np.ravel(val[:, 0]).astype(float)
                y = np.ravel(val[:, 1]).astype(float)
                u = np.ravel(val[:, 2]).astype(float)
            break

    # Strategy 2: named variables
    if T is None:
        try:
            T = np.ravel(data.get("t", data.get("time"))).astype(float)
            y = np.ravel(data.get("y")).astype(float)
            u = np.ravel(data.get("u")).astype(float)
        except Exception:
            pass

    if T is None or y is None or u is None:
        raise ValueError(
            f"Could not extract [t, y, u] from {mat_path}. "
            "Expected named variables or a 3xN / Nx3 matrix."
        )

    return {"t": T, "y": y, "u": u}


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def compute_time_domain_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fir_length: int,
) -> Dict[str, float]:
    """
    Compute RMSE, FIT%, and R2 with a minimal transient skip.

    The skip length matches the "strict evaluation" convention:
        skip = min(10, fir_length // 10)

    R2 uses the first valid sample (not the mean) as the baseline,
    consistent with the original gp_to_fir_direct_fixed.py.

    Args:
        y_true:     Measured output (1-D array).
        y_pred:     Predicted output (1-D array, same length).
        fir_length: Number of FIR taps (controls transient skip).

    Returns:
        Dictionary with keys: rmse, fit_percent, r2, transient_skip, detrend.
    """
    skip = min(10, fir_length // 10)
    y_valid = y_true[skip:]
    y_pred_valid = y_pred[skip:]

    err = y_valid - y_pred_valid
    rmse = float(np.sqrt(np.mean(err**2)))

    norm_y = np.linalg.norm(y_valid)
    fit = (float(100 * (1.0 - np.linalg.norm(err) / norm_y))
           if norm_y > 0 else 0.0)

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_valid - y_valid[0])**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "rmse": rmse,
        "fit_percent": fit,
        "r2": r2,
        "transient_skip": skip,
        "detrend": False,
    }


# ------------------------------------------------------------------ #
# End-to-end validation
# ------------------------------------------------------------------ #

def validate_fir_with_mat(
    fir_coefficients: np.ndarray,
    mat_path: str | Path,
    output_dir: Optional[str | Path] = None,
    prefix: str = "fir",
    detrend: bool = False,
) -> Dict[str, object]:
    """
    Evaluate an FIR model against time-series data from a MAT file.

    Steps:
        1. Load [t, y, u] from *mat_path*.
        2. Optionally remove DC (detrend).
        3. Predict via causal convolution.
        4. Compute metrics with minimal transient skip.
        5. Optionally save plots to *output_dir*.

    Args:
        fir_coefficients: 1-D array of FIR taps.
        mat_path:         Path to the MAT file with [t, y, u].
        output_dir:       If not None, save output-vs-predicted and error
                          plots to this directory.
        prefix:           Filename prefix for saved plots.
        detrend:          If True, remove DC from u and y before evaluation.

    Returns:
        Dictionary with: rmse, fit_percent, r2, transient_skip, detrend,
        plus 'validation_file' (str path to the MAT used).
    """
    g = np.asarray(fir_coefficients, dtype=float).ravel()
    N = len(g)

    # Load time-series
    ts = load_mat_time_series(mat_path)
    T = ts["t"]
    y = ts["y"]
    u = ts["u"]

    # Optional detrend
    if detrend:
        u_eval = u - np.mean(u)
        y_eval = y - np.mean(y)
    else:
        u_eval = u.copy()
        y_eval = y.copy()

    # Predict with causal convolution
    y_pred = np.convolve(u_eval, g, mode="full")[:len(y_eval)]

    # Compute metrics
    metrics = compute_time_domain_metrics(y_eval, y_pred, fir_length=N)
    metrics["validation_file"] = str(mat_path)

    print(f"  Validation ({Path(mat_path).name}): "
          f"RMSE={metrics['rmse']:.3e}, "
          f"FIT={metrics['fit_percent']:.1f}%, "
          f"R2={metrics['r2']:.3f}")

    # Optional plots
    if output_dir is not None:
        _plot_validation(
            T, y, y_pred,
            rmse=metrics["rmse"],
            output_dir=Path(output_dir),
            prefix=prefix,
        )

    return metrics


# ------------------------------------------------------------------ #
# Internal plotting helper
# ------------------------------------------------------------------ #

def _plot_validation(
    t: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    rmse: float,
    output_dir: Path,
    prefix: str,
):
    """
    Save output-vs-predicted and error plots (PNG + EPS).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    t_max = min(t[-1], t[0] + 10)
    mask = t <= t_max

    # -- Output vs Predicted --
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(t[mask], y[mask], 'k-', label='Measured', linewidth=3.0,
             alpha=0.8)
    ax1.plot(t[mask], y_pred[mask], 'r--', label='FIR Predicted',
             linewidth=3.0)
    ax1.set_xlabel('Time [s]', fontsize=32, fontweight='bold')
    ax1.set_ylabel('Output [rad]', fontsize=32, fontweight='bold')
    ax1.set_title(f'FIR Validation  RMSE={rmse:.3e}', fontsize=36,
                  fontweight='bold', pad=20)
    ax1.legend(fontsize=26, framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.tick_params(labelsize=24, width=2.5, length=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_output_vs_predicted.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{prefix}_output_vs_predicted.eps",
                format='eps', bbox_inches='tight')
    plt.close(fig1)

    # -- Error --
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    error = y - y_pred
    ax2.plot(t[mask], error[mask], 'b-', linewidth=3.0, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax2.set_xlabel('Time [s]', fontsize=32, fontweight='bold')
    ax2.set_ylabel('Error', fontsize=32, fontweight='bold')
    ax2.set_title('FIR Prediction Error', fontsize=36, fontweight='bold',
                  pad=20)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.tick_params(labelsize=24, width=2.5, length=10)

    error_stats = (f'Mean: {np.mean(error[mask]):.3e}\n'
                   f'Std:  {np.std(error[mask]):.3e}')
    ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes,
             fontsize=20, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       alpha=0.8, linewidth=2))
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_error.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{prefix}_error.eps",
                format='eps', bbox_inches='tight')
    plt.close(fig2)

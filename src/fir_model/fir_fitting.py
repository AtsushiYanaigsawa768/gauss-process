#!/usr/bin/env python3
"""
fir_fitting.py

Pipeline functions for GP-to-FIR conversion.

Provides:
- gp_to_fir_direct_pipeline: Main pipeline supporting both paper-mode
  (uniform omega + two-sided Hermitian + IDFT) and legacy-mode (irfft).
- plot_gp_fir_results_fixed: Visualization of FIR model validation results.

Imports low-level helpers from fir_helpers.py.

Preserves the exact numerical behavior of the original
gp_to_fir_direct_fixed.py.
"""

from __future__ import annotations
from typing import Callable, Dict, Optional
from pathlib import Path
import warnings

import numpy as np
from scipy.io import loadmat
from numpy.fft import irfft
import matplotlib.pyplot as plt

from src.fir_model.fir_helpers import (
    build_uniform_omega_linear,
    build_two_sided_hermitian_from_positive,
    idft_to_fir_coeffs_two_sided,
    _load_timebase_from_mat,
    _build_H_for_irfft,
)


# ------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------ #

def plot_gp_fir_results_fixed(
    t: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    u: np.ndarray,
    rmse: float,
    fit_percent: float,
    r2: float,
    output_dir: Path,
    prefix: str = "gp_fir_fixed",
    save_eps: bool = True,
):
    """
    Create visualization plots for GP-based FIR model results (fixed version).
    Produces two figures: (1) output vs predicted, (2) prediction error.

    Args:
        t:           Time vector.
        y:           Actual output.
        y_pred:      Predicted output.
        u:           Input signal.
        rmse:        Root mean square error.
        fit_percent: FIT percentage metric.
        r2:          R-squared value.
        output_dir:  Directory to save plots.
        prefix:      Filename prefix.
        save_eps:    If True, also save EPS files (default True).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Figure 1: Output vs Predicted --
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Show full time range or limit to first 10 seconds
    t_max = min(t[-1], t[0] + 10)
    mask = t <= t_max

    ax1.plot(t[mask], y[mask], 'k-', label='Measured Output',
             linewidth=3.0, alpha=0.8)
    ax1.plot(t[mask], y_pred[mask], 'r--', label='FIR Predicted',
             linewidth=3.0)
    ax1.set_xlabel('Time [s]', fontsize=32, fontweight='bold')
    ax1.set_ylabel('Output [rad]', fontsize=32, fontweight='bold')

    title = (f'FIR Model Validation\n'
             f'RMSE={rmse:.3e}')
    ax1.set_title(title, fontsize=36, fontweight='bold', pad=20)

    ax1.legend(fontsize=26, framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.tick_params(labelsize=24, width=2.5, length=10)

    plt.tight_layout()

    png_path = output_dir / f"{prefix}_output_vs_predicted.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if save_eps:
        eps_path = output_dir / f"{prefix}_output_vs_predicted.eps"
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close(fig1)

    # -- Figure 2: Error --
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    error = y - y_pred
    ax2.plot(t[mask], error[mask], 'b-', linewidth=3.0, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax2.set_xlabel('Time [s]', fontsize=32, fontweight='bold')
    ax2.set_ylabel('Error (Measured - Predicted)', fontsize=32,
                   fontweight='bold')
    ax2.set_title('FIR Model Prediction Error', fontsize=36,
                  fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.tick_params(labelsize=24, width=2.5, length=10)

    error_stats = (f'Mean Error: {np.mean(error[mask]):.3e}\n'
                   f'Std Error: {np.std(error[mask]):.3e}')
    ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes,
             fontsize=20, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       alpha=0.8, linewidth=2))

    plt.tight_layout()

    png_path = output_dir / f"{prefix}_error.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if save_eps:
        eps_path = output_dir / f"{prefix}_error.eps"
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close(fig2)

    # Placeholder for optional frequency-response comparison figure
    fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    # (Currently unused -- kept for parity with original module)

    print(f"Plots saved to {output_dir}")


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #

def gp_to_fir_direct_pipeline(
    omega: np.ndarray,
    G: np.ndarray,
    gp_predict_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mat_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fir_length: int = 1024,
    N_fft: Optional[int] = None,
    paper_mode: bool = True,
    fir_order: Optional[int] = None,
) -> Dict[str, object]:
    """
    Corrected GP-to-FIR pipeline with paper-mode support.

    Paper mode (default):
        1) Build uniform omega grid (1000 pts) + GP prediction.
        2) Two-sided Hermitian spectrum (odd-length M = 2*Nd - 1).
        3) IDFT -> first N taps as FIR coefficients.

    Legacy mode (paper_mode=False):
        Maps continuous omega to digital Omega via Ts from MAT file,
        evaluates G on DFT bins, then uses irfft.

    Args:
        omega:           Angular frequencies [rad/s] for GP-smoothed FRF.
        G:               Complex FRF values.
        gp_predict_func: Optional callable for GP predictions at arbitrary omega.
        mat_file:        MAT file with timebase (required for correct mapping
                         in legacy mode; optional for paper mode validation).
        output_dir:      Directory to save artifacts.
        fir_length:      Number of FIR taps for legacy IRFFT mode.
        N_fft:           FFT length for legacy mode.
        paper_mode:      If True (default), use the paper-based procedure.
        fir_order:       FIR model order for paper mode (default: min(M, 1024)).

    Returns:
        Dictionary with FIR coefficients, metrics, and metadata.
    """
    omega = np.asarray(omega, dtype=float).ravel()
    G = np.asarray(G, dtype=complex).ravel()
    if output_dir is None:
        output_dir = Path("fir_gp_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Paper mode: uniform omega grid + two-sided Hermitian + IDFT
    # ============================================================
    if paper_mode:
        print("[Paper Mode] Using paper-based FIR procedure:")
        print("  Step 1: Uniform linear omega grid + GP prediction")
        print("  Step 2: Two-sided Hermitian spectrum (odd-length)")
        print("  Step 3: IDFT -> first N taps as FIR coefficients")

        # Step 1: Build uniform omega grid with 1000 points
        Nd = 1000
        omega_min = float(np.min(omega))
        omega_max = float(np.max(omega))
        omega_uni = np.linspace(omega_min, omega_max, Nd)

        # Use GP predictor if available, otherwise fall back to linear interp
        if gp_predict_func is not None:
            print("  Using GP predictor for uniform grid (higher accuracy)")
            G_uni = gp_predict_func(omega_uni)
        else:
            print("  Using linear interpolation (GP predictor not available)")
            Gr = np.interp(omega_uni, omega, np.real(G),
                           left=np.real(G[0]), right=np.real(G[-1]))
            Gi = np.interp(omega_uni, omega, np.imag(G),
                           left=np.imag(G[0]), right=np.imag(G[-1]))
            G_uni = Gr + 1j * Gi

        print(f"  Nd = {Nd} frequency points")

        # Step 2: Build two-sided Hermitian spectrum (M = 2*Nd - 1)
        X = build_two_sided_hermitian_from_positive(G_uni)
        M = X.size
        print(f"  M = {M} (IDFT length, odd)")

        # Verify Hermitian symmetry: DC imaginary part
        max_imag_dc = np.abs(X[0].imag)
        if max_imag_dc > 1e-10:
            warnings.warn(
                f"DC has significant imaginary part: {max_imag_dc:.3e}")

        # Step 3: IDFT -> impulse response, take first N taps
        N_requested = int(fir_order if fir_order is not None
                          else min(M, 1024))
        N = min(N_requested, M)
        if N < N_requested:
            warnings.warn(
                f"Requested fir_order={N_requested} exceeds M={M}. "
                f"Using N={N} instead.")
        g, h_full = idft_to_fir_coeffs_two_sided(X, N)
        print(f"  N = {N} FIR taps extracted (requested: {N_requested})")

        # Verify realness of impulse response
        max_imag_h = (np.max(np.abs(h_full.imag))
                      if np.iscomplexobj(h_full) else 0.0)
        if max_imag_h > 1e-10:
            warnings.warn(
                f"Impulse response has significant imaginary part: "
                f"{max_imag_h:.3e}")

        # Collect results (no .npz file written)
        results: Dict[str, object] = {
            "paper_mode": True,
            "Nd": int(Nd),
            "M_idft": int(M),
            "fir_order": int(N),
            "fir_coefficients": g,
            "method": "gp_paper_mode",
            "omega_range": [float(omega_uni[0]), float(omega_uni[-1])],
            "rmse": None,
            "fit_percent": None,
            "r2": None,
        }

        # Validation if MAT file provided
        if mat_file is not None and Path(mat_file).exists():
            t, Ts = _load_timebase_from_mat(Path(mat_file))
            print(f"  Ts = {Ts:.6f} s (from MAT file)")
            results["Ts"] = float(Ts)

            # Load time-series data for validation
            data = loadmat(mat_file)
            T, y, u = _extract_tyu_from_mat_data(data)

            if T is not None and y is not None and u is not None:
                # STRICT EVALUATION MODE: no detrend, minimal transient skip
                DETREND = False
                if DETREND:
                    u_eval = u - np.mean(u)
                    y_eval = y - np.mean(y)
                    print("  [WARNING] Detrend is ON - "
                          "DC errors will be hidden")
                else:
                    u_eval = u.copy()
                    y_eval = y.copy()
                    print("  [STRICT MODE] No detrend - evaluating full "
                          "frequency range including DC")

                # Predict with causal convolution
                y_pred = np.convolve(u_eval, g, mode="full")[:len(y_eval)]

                # Minimal transient skip
                skip = min(10, N // 10)
                y_valid = y_eval[skip:]
                y_pred_valid = y_pred[skip:]

                print(f"  Transient skip: {skip} samples "
                      f"(was: {N // 2} samples)")

                # Metrics WITHOUT additional detrending
                err = y_valid - y_pred_valid
                rmse = float(np.sqrt(np.mean(err**2)))

                norm_y = np.linalg.norm(y_valid)
                fit = (float(100 * (1.0 - np.linalg.norm(err) / norm_y))
                       if norm_y > 0 else 0.0)

                ss_res = float(np.sum(err**2))
                ss_tot = float(np.sum((y_valid - y_valid[0])**2))
                r2 = (float(1.0 - ss_res / ss_tot)
                      if ss_tot > 0 else 0.0)

                results.update({
                    "rmse": rmse,
                    "fit_percent": fit,
                    "r2": r2,
                    "transient_skip": skip,
                    "detrend": DETREND,
                })

                print(f"  Validation (STRICT): RMSE={rmse:.3e}, "
                      f"FIT={fit:.1f}%, R2={r2:.3f}")

                plot_gp_fir_results_fixed(
                    t=T, y=y, y_pred=y_pred, u=u,
                    rmse=rmse, fit_percent=fit, r2=r2,
                    output_dir=output_dir,
                    prefix="gp_fir_paper",
                )
            else:
                print("  Warning: Could not load validation data "
                      "from MAT file")
        else:
            print("  No validation MAT file provided")
            results["Ts"] = None

        print("[Paper Mode] FIR extraction complete")
        return results

    # ============================================================
    # Legacy mode: original IRFFT-based approach
    # ============================================================
    print("[Legacy Mode] Using IRFFT-based approach")

    if mat_file is None or not Path(mat_file).exists():
        warnings.warn(
            "No MAT file (timebase) provided. Falling back to *unit* "
            "Ts=1.0 s. This will likely distort the FIR. "
            "Provide --fir-validation-mat.")
        Ts = 1.0
        t = None
    else:
        t, Ts = _load_timebase_from_mat(Path(mat_file))

    # Build spectrum on the digital grid
    H_half, omega_k, Omega_k = _build_H_for_irfft(
        omega, G, L=fir_length, Ts=Ts, N_fft=N_fft,
        gp_predict_func=gp_predict_func, taper=True,
    )

    # Real impulse via irfft; take first L (causal) taps
    h_full = irfft(H_half, n=(2 * (len(H_half) - 1)))
    g = h_full[:fir_length].copy()

    results = {
        "fir_length": fir_length,
        "fir_coefficients": g,
        "Ts": Ts,
        "method": "gp_direct_fixed",
        "n_fft": int(2 * (len(H_half) - 1)),
        "rmse": None,
        "fit_percent": None,
        "r2": None,
    }

    # Optional validation if MAT present
    if mat_file is not None and Path(mat_file).exists():
        data = loadmat(mat_file)
        T, y, u = _extract_tyu_from_mat_data(data)

        if T is not None and y is not None and u is not None:
            # Use raw signals for evaluation (no detrending)
            u_eval = np.asarray(u, dtype=float)
            y_eval = np.asarray(y, dtype=float)

            # Predict with causal convolution
            y_pred = np.convolve(u_eval, g, mode="full")[:len(y_eval)]

            # Small leading transient skip
            skip = min(10, len(g) // 10)
            y_valid = y_eval[skip:]
            y_pred_valid = y_pred[skip:]

            err = y_valid - y_pred_valid
            rmse = float(np.sqrt(np.mean(err**2)))

            norm_y = np.linalg.norm(y_valid)
            fit = (float(100 * (1.0 - np.linalg.norm(err) / norm_y))
                   if norm_y > 0 else 0.0)

            ss_res = float(np.sum(err**2))
            ss_tot = float(np.sum((y_valid - y_valid[0])**2))
            r2 = (float(1.0 - ss_res / ss_tot)
                  if ss_tot > 0 else 0.0)

            results.update({
                "rmse": rmse,
                "fit_percent": fit,
                "r2": r2,
                "transient_skip": int(skip),
            })

            plot_gp_fir_results_fixed(
                t=T, y=y, y_pred=y_pred, u=u,
                rmse=rmse, fit_percent=fit, r2=r2,
                output_dir=output_dir,
            )

    return results


# ------------------------------------------------------------------ #
# Internal helper: extract [t, y, u] from MAT dict
# ------------------------------------------------------------------ #

def _extract_tyu_from_mat_data(data: dict):
    """
    Try to extract time, output, and input vectors from a loadmat dict.

    Looks for a 3xN or Nx3 array first, then tries named variables
    't'/'time', 'y', 'u'.

    Returns:
        (T, y, u) or (None, None, None) on failure.
    """
    T = None
    y = None
    u = None
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

    if T is None:
        # Try named variables
        try:
            T = np.ravel(data.get("t", data.get("time"))).astype(float)
            y = np.ravel(data.get("y"))
            u = np.ravel(data.get("u"))
        except Exception:
            T = None
            y = None
            u = None

    return T, y, u

#!/usr/bin/env python3
"""
fir_helpers.py

Low-level helper functions for FIR coefficient extraction from frequency
response data.

Provides:
- build_uniform_omega_linear: Build a uniform omega grid with linear interpolation
- build_two_sided_hermitian_from_positive: Construct a two-sided Hermitian spectrum
- idft_to_fir_coeffs_two_sided: IDFT to extract FIR coefficients
- _load_timebase_from_mat: Load timebase (t, Ts) from a MAT file
- _build_H_for_irfft: Build one-sided spectrum for irfft (legacy mode)

All functions preserve the exact numerical behavior of the original
gp_to_fir_direct_fixed.py.
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from numpy.fft import irfft


# ------------------------------------------------------------------ #
# Step 1: Uniform omega grid via linear interpolation
# ------------------------------------------------------------------ #

def build_uniform_omega_linear(
    omega_meas: np.ndarray,
    G_meas: np.ndarray,
    Nd: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a uniform-spacing grid on the continuous omega (linear scale)
    and evaluate G_uni(omega_m) by linear interpolation of real/imag parts.

    Args:
        omega_meas: Measured angular frequencies [rad/s] (ascending recommended).
        G_meas:     Complex FRF G(j*omega) values (same length as omega_meas).
        Nd:         Number of uniform grid points (None -> same as len(omega_meas)).

    Returns:
        omega_uni: Uniformly spaced omega grid (Nd points).
        G_uni:     Linearly interpolated complex FRF (Nd points).
    """
    omega_meas = np.asarray(omega_meas, dtype=float).ravel()
    G_meas = np.asarray(G_meas, dtype=complex).ravel()
    assert omega_meas.size == G_meas.size and omega_meas.size >= 2

    if Nd is None:
        Nd = omega_meas.size

    omega_min = float(np.min(omega_meas))
    omega_max = float(np.max(omega_meas))
    omega_uni = np.linspace(omega_min, omega_max, Nd)

    # Interpolate real and imaginary parts separately (edge values held)
    Gr = np.interp(omega_uni, omega_meas, np.real(G_meas),
                   left=np.real(G_meas[0]), right=np.real(G_meas[-1]))
    Gi = np.interp(omega_uni, omega_meas, np.imag(G_meas),
                   left=np.imag(G_meas[0]), right=np.imag(G_meas[-1]))
    G_uni = Gr + 1j * Gi
    return omega_uni, G_uni


# ------------------------------------------------------------------ #
# Step 2: Two-sided Hermitian spectrum for real impulse response
# ------------------------------------------------------------------ #

def build_two_sided_hermitian_from_positive(
    G_uni: np.ndarray,
) -> np.ndarray:
    """
    Construct a two-sided (odd-length) spectrum that satisfies Hermitian
    symmetry so that IFFT produces a real-valued impulse response.

    Length M = 2*Nd - 1 (odd, no Nyquist singularity).
    X[0]        = DC (forced real for real-impulse condition).
    X[1:Nd]     = positive-side frequencies.
    X[M-(Nd-1):] = conjugate-reversed positive side (negative frequencies).

    Args:
        G_uni: Positive-side samples (Nd points, DC through highest freq).

    Returns:
        X: Complex spectrum of length M suitable for np.fft.ifft.
    """
    G_uni = np.asarray(G_uni, dtype=complex).ravel()
    Nd = G_uni.size
    assert Nd >= 2

    M = 2 * Nd - 1  # odd length (no standalone Nyquist bin)
    X = np.zeros(M, dtype=complex)

    # Positive side: k = 0 .. Nd-1 (DC + positive frequencies)
    X[0] = np.real(G_uni[0])  # DC forced real (real impulse condition)
    if Nd > 1:
        X[1:Nd] = G_uni[1:Nd]

    # Negative side: k = M-(Nd-1) .. M-1 (conjugate reversal of positive side)
    if Nd > 1:
        X[M - (Nd - 1):] = np.conjugate(G_uni[1:Nd][::-1])
    return X


# ------------------------------------------------------------------ #
# Step 3: IDFT to FIR coefficients
# ------------------------------------------------------------------ #

def idft_to_fir_coeffs_two_sided(
    X: np.ndarray,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the impulse response from a two-sided spectrum via IFFT and
    return the first N taps as FIR coefficients.

    Args:
        X: 1-D complex spectrum (output of build_two_sided_hermitian_from_positive).
        N: Desired FIR model order (number of taps).

    Returns:
        h:      FIR coefficients (length N, real-valued).
        h_full: Full IFFT output (length M, real part).
    """
    X = np.asarray(X, dtype=complex).ravel()
    M = X.size
    h_full = np.fft.ifft(X, n=M).real  # NumPy uses 1/M normalization
    if N > M:
        raise ValueError(f"N={N} exceeds available length M={M}. Reduce N.")
    h = h_full[:N].copy()
    return h, h_full


# ------------------------------------------------------------------ #
# MAT file timebase loader
# ------------------------------------------------------------------ #

def _load_timebase_from_mat(
    mat_file: Path,
) -> Tuple[np.ndarray, float]:
    """
    Extract (t, Ts) from a MAT file.

    Tries named variables 't' / 'time' first, then looks for a 3xN or Nx3
    array whose first row/column is assumed to be a time vector.

    Args:
        mat_file: Path to the .mat file.

    Returns:
        t:  Time vector.
        Ts: Median sampling period.

    Raises:
        ValueError: If a time vector cannot be inferred.
    """
    data = loadmat(mat_file)
    t = None
    # 1) Direct 't' or 'time'
    for key in ('t', 'time'):
        if key in data:
            t = np.ravel(data[key]).astype(float)
            break
    # 2) 3xN or Nx3 array with [t, y, u]
    if t is None:
        for k, v in data.items():
            if k.startswith('__') or not isinstance(v, np.ndarray):
                continue
            if v.ndim == 2 and (v.shape[0] == 3 or v.shape[1] == 3):
                if v.shape[0] == 3:
                    t = np.ravel(v[0, :]).astype(float)
                else:
                    t = np.ravel(v[:, 0]).astype(float)
                break
    if t is None or t.size < 2:
        raise ValueError(
            "Could not infer time vector from MAT file; "
            "need 't'/'time' or 3-column array [t,y,u]."
        )
    dt = np.median(np.diff(t))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Invalid time vector; cannot compute Ts.")
    return t, float(dt)


# ------------------------------------------------------------------ #
# Legacy helper: one-sided spectrum for irfft
# ------------------------------------------------------------------ #

def _build_H_for_irfft(
    omega: np.ndarray,
    G: np.ndarray,
    L: int,
    Ts: float,
    N_fft: Optional[int] = None,
    gp_predict_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    taper: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one-sided spectrum H_k for numpy.fft.irfft.

    Maps continuous omega [rad/s] to digital Omega [rad/sample] on the
    DFT grid, evaluates G(j*omega_k) via a GP predictor or cubic
    interpolation, and optionally applies a cosine taper near Nyquist.

    Args:
        omega:           Measured angular frequencies [rad/s].
        G:               Complex FRF values at *omega*.
        L:               Desired FIR length (taps).
        Ts:              Sampling period [s].
        N_fft:           FFT length (None -> next power-of-two >= 4*L).
        gp_predict_func: Optional callable returning complex G at arbitrary omega.
        taper:           Apply cosine roll-off near Nyquist (default True).

    Returns:
        H_half:  One-sided spectrum (length N_fft//2 + 1).
        omega_k: Continuous frequencies at DFT bins [rad/s].
        Omega_k: Digital frequencies at DFT bins [rad/sample].
    """
    if N_fft is None:
        # Reasonable resolution: at least 4*L and power-of-two
        N = 1
        while N < 4 * L:
            N <<= 1
        N_fft = N
    if N_fft % 2 == 1:
        N_fft += 1  # enforce even for irfft

    k = np.arange(N_fft // 2 + 1, dtype=int)
    Omega_k = 2.0 * np.pi * k / N_fft            # [rad/sample]
    omega_k = Omega_k / Ts                        # [rad/s]

    # Predict/interpolate G at omega_k
    if gp_predict_func is not None:
        Gk = gp_predict_func(omega_k)
    else:
        # Safe cubic interpolation with edge hold
        interp_real = interp1d(omega, np.real(G), kind='cubic',
                               bounds_error=False,
                               fill_value=(np.real(G[0]), np.real(G[-1])))
        interp_imag = interp1d(omega, np.imag(G), kind='cubic',
                               bounds_error=False,
                               fill_value=(np.imag(G[0]), np.imag(G[-1])))
        Gk = interp_real(omega_k) + 1j * interp_imag(omega_k)

    # Enforce DC and Nyquist realness for real impulse response
    Gk = np.asarray(Gk, dtype=complex)
    Gk[0] = np.real(Gk[0])
    if (N_fft % 2 == 0):
        Gk[-1] = np.real(Gk[-1])

    # Optional mild taper near Nyquist to reduce circular wrap/alias
    if taper and len(Gk) > 8:
        m = len(Gk)
        n_roll = max(4, int(0.1 * m))
        w = np.ones(m)
        win = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n_roll)))
        w[-n_roll:] *= win
        Gk = Gk * w

    return Gk, omega_k, Omega_k

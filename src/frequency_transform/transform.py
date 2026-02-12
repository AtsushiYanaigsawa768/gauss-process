#!/usr/bin/env python3
"""
transform.py

Unified interface for frequency-domain estimation.  Dispatches to either
the FRF estimator (trapezoidal synchronous demodulation on a log frequency
grid) or the Fourier estimator (FFT-based on a linear frequency grid),
with optional disk-based caching.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.frequency_transform.frf_estimator import (
    resolve_mat_files,
    load_time_u_y,
    try_load_frequency_from_mat,
    matlab_freq_grid,
    describe_timebase,
    synchronous_coefficients_trapz,
    frf_cross_power_average,
)
from src.frequency_transform.fourier_estimator import (
    load_time_data,
    apply_time_window,
    compute_transfer_function,
    linear_frequency_grid,
    interpolate_to_grid,
    average_multiple_estimates,
)
from src.frequency_transform.cache import FrequencyDataCache


# Default constants (overridable via CLI flags or caller arguments)
DEFAULT_ND = 100
DEFAULT_F_LOW_LOG10 = -1.0
DEFAULT_F_UP_LOG10 = 2.3
DEFAULT_DROP_SECONDS = 0.0
DEFAULT_WINDOW = "hann"


def estimate_frequency_response(
    mat_files: List[str],
    method: str = "frf",
    nd: int = DEFAULT_ND,
    f_low: float = DEFAULT_F_LOW_LOG10,
    f_up: float = DEFAULT_F_UP_LOG10,
    drop_seconds: float = DEFAULT_DROP_SECONDS,
    time_duration: Optional[float] = None,
    n_files: Optional[int] = None,
    y_col: int = 0,
    subtract_mean: bool = True,
    window: str = DEFAULT_WINDOW,
    use_cache: bool = False,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Estimate frequency response from MAT-file time-series data.

    Args:
        mat_files: Paths / globs to .mat input files.
        method: 'frf' for trapezoidal synchronous demodulation (log grid),
                'fourier' for FFT-based estimation (linear grid).
        nd: Number of frequency grid points.
        f_low: log10 lower frequency bound (frf) or min Hz (fourier).
        f_up: log10 upper frequency bound (frf) or max Hz (fourier).
        drop_seconds: Seconds to discard from start of each record.
        time_duration: Optional duration limit per file [s].
        n_files: Use only the first N files (None = all).
        y_col: Column of y to use when y is 2D.
        subtract_mean: Remove DC offset before estimation (frf only).
        window: FFT window function name (fourier only).
        use_cache: If True, check/store results in the disk cache.
        cache_dir: Override default cache directory.

    Returns:
        DataFrame with columns [omega_rad_s, freq_Hz, ReG, ImG, absG, phase_rad].
    """
    # Resolve file list
    paths = resolve_mat_files(mat_files)
    if not paths:
        raise FileNotFoundError("No MAT files found for the given input.")
    if n_files is not None:
        paths = paths[:n_files]

    effective_n = len(paths)

    # Optional caching
    cache = None
    if use_cache:
        cache = FrequencyDataCache(cache_dir) if cache_dir else FrequencyDataCache()
        cache_key = cache.get_cache_key(
            [str(p) for p in paths], effective_n, time_duration, nd, method
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # Dispatch
    if method == "frf":
        df = _estimate_frf(paths, nd, f_low, f_up, drop_seconds, time_duration, y_col, subtract_mean)
    elif method == "fourier":
        df = _estimate_fourier(paths, nd, f_low, f_up, drop_seconds, time_duration, y_col, window)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'frf' or 'fourier'.")

    # Store in cache
    if cache is not None:
        cache.put(cache_key, df)

    return df


# ------------------------------------------------------------------ #
#  Internal dispatch helpers                                          #
# ------------------------------------------------------------------ #

def _estimate_frf(paths, nd, f_low, f_up, drop_seconds, time_duration, y_col, subtract_mean):
    """Run the FRF (trapezoidal demodulation) pipeline."""
    _, omega = matlab_freq_grid(nd, f_low, f_up)

    U_list, Y_list = [], []
    for mat_path in paths:
        t, u, y = load_time_u_y(mat_path, y_col=y_col)

        # Optional time-duration windowing
        drop_to_use = drop_seconds
        if time_duration is not None:
            t_start = t[0] + drop_seconds
            t_end = t_start + time_duration
            mask = (t >= t_start) & (t <= t_end)
            if not np.any(mask):
                raise ValueError(f"No data in [{t_start:.2f}, {t_end:.2f}]")
            t, u, y = t[mask], u[mask], y[mask]
            drop_to_use = 0.0

        # Ensure monotonic time
        _, _, is_mono = describe_timebase(t)
        if not is_mono:
            order = np.argsort(t)
            t, u, y = t[order], u[order], y[order]

        U = synchronous_coefficients_trapz(t, u, omega, drop_to_use, subtract_mean)
        Y = synchronous_coefficients_trapz(t, y, omega, drop_to_use, subtract_mean)
        U_list.append(U)
        Y_list.append(Y)

    G = frf_cross_power_average(U_list, Y_list)

    # Sort by omega (defensive)
    order = np.argsort(omega)
    omega = omega[order]
    G = G[order]

    return _build_dataframe(omega, G)


def _estimate_fourier(paths, nd, f_min, f_max, drop_seconds, time_duration, y_col, window):
    """Run the Fourier (FFT) pipeline."""
    G_list = []
    f_max_actual = 0.0
    win = None if window == "none" else window

    for mat_path in paths:
        t, u, y = load_time_data(mat_path, y_col=y_col)
        t, u, y = apply_time_window(t, u, y, drop_seconds, time_duration)
        freq, G = compute_transfer_function(t, u, y, window=win)
        f_max_actual = max(f_max_actual, freq[-1])
        G_list.append((freq, G))

    effective_f_max = min(f_max, f_max_actual) if f_max is not None else f_max_actual
    f_grid = linear_frequency_grid(f_min, effective_f_max, nd)

    G_interp_list = [interpolate_to_grid(freq, G, f_grid) for freq, G in G_list]

    G_avg = average_multiple_estimates(G_interp_list) if len(G_interp_list) > 1 else G_interp_list[0]

    omega = 2.0 * np.pi * f_grid
    return _build_dataframe(omega, G_avg)


def _build_dataframe(omega: np.ndarray, G: np.ndarray) -> pd.DataFrame:
    """Assemble the standard output DataFrame."""
    return pd.DataFrame({
        "omega_rad_s": omega,
        "freq_Hz": omega / (2.0 * np.pi),
        "ReG": np.real(G),
        "ImG": np.imag(G),
        "absG": np.abs(G),
        "phase_rad": np.angle(G),
    })

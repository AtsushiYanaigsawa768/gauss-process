#!/usr/bin/env python3
"""
lms_filter.py

Basic LMS (Least Mean Squares) adaptive FIR filter identification.

Loads input/output time-series from a MAT file, runs a standard LMS
update loop, and plots measured vs predicted output together with the
prediction error.

Originally fir/gp_fir_model.py. All Japanese comments translated to
English; computational logic unchanged.

Constants (tunables):
    MAT_PATH   : Path to the MAT file with [time; y; u].
    N_SAMPLES  : Number of samples to use from the data.
    L          : Number of FIR taps.
    MU         : LMS step size (learning rate).
    PLOT_START : Time [s] at which to begin plotting (skip transient).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.lib.stride_tricks import sliding_window_view

# ------------------------------------------------------------------ #
# Tunables (override via CLI or caller if this module is imported)
# ------------------------------------------------------------------ #
MAT_PATH = './fir/data/data_hour.mat'
N_SAMPLES = 100_000
L = 32        # Number of FIR taps (fixed for lightweight operation)
MU = 0.01     # LMS learning rate (adjust as needed)
PLOT_START = 10.0  # Start plotting from this time [s]


def run_lms(mat_path: str = MAT_PATH,
            n_samples: int = N_SAMPLES,
            num_taps: int = L,
            mu: float = MU,
            plot_start: float = PLOT_START):
    """
    Run LMS identification on time-series data from a MAT file.

    Args:
        mat_path:   Path to the MAT file (3xN or Nx3: [time, y, u]).
        n_samples:  Number of samples to use.
        num_taps:   FIR filter length.
        mu:         LMS step size.
        plot_start: Start time for plotting (skips initial transient).

    Returns:
        Dictionary with keys: h, yhat, e, rmse.
    """
    # -- Load data --
    io = loadmat(mat_path)
    for k, v in io.items():
        if not k.startswith('__'):
            mat = v
            break
    time = mat[0, :n_samples]
    y = mat[1, :n_samples]
    u = mat[2, :n_samples]
    N = len(u)

    # -- Initialize filter --
    h = np.zeros(num_taps)
    yhat = np.zeros(N)
    e = np.zeros(N)

    # -- Build sliding-window regressor [u[n], u[n-1], ..., u[n-L+1]] --
    u_pad = np.concatenate([np.zeros(num_taps - 1), u])
    U = sliding_window_view(u_pad, window_shape=num_taps)[:N, ::-1]

    # -- LMS update loop --
    for n in range(N):
        phi = U[n]
        yhat[n] = phi.dot(h)
        e[n] = y[n] - yhat[n]
        h += mu * e[n] * phi  # LMS update rule

    # -- Plot results (data from time >= plot_start) --
    start_idx = np.searchsorted(time, plot_start)

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time[start_idx:], y[start_idx:], 'k', label='y (measured)')
    plt.plot(time[start_idx:], yhat[start_idx:], 'r--', label='yhat (LMS)')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(time[start_idx:], e[start_idx:], 'b')
    plt.title('error')
    plt.tight_layout()
    plt.show()

    rmse = float(np.sqrt(np.mean(e[num_taps:]**2)))
    return {"h": h, "yhat": yhat, "e": e, "rmse": rmse}


if __name__ == "__main__":
    run_lms()

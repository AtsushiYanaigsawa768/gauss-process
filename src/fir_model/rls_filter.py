#!/usr/bin/env python3
"""
rls_filter.py

Real-time FIR identification using RLS (Recursive Least Squares).

Workflow:
  1. Load frequency response from a CSV file (omega, ReG, ImG).
  2. Interpolate to a uniform grid and compute the impulse response via
     Hermitian-symmetric IFFT.
  3. Trim the impulse response by cumulative energy to determine FIR
     length L, then apply a Hann window.
  4. Load recorded input/output from a MAT file.
  5. Run an RLS loop, updating the FIR coefficients in real time with
     live plotting.
  6. Report RMSE, NRMSE, and R2 error metrics and save results.

Originally fir/test_fir.py. All Japanese comments translated to English;
computational logic unchanged.

Constants (tunables):
    FRF_FILE       : CSV with columns [omega, ReG, ImG].
    IO_FILE        : MAT with recorded I/O data ([time; y; u]).
    LAMBDA_FACTOR  : RLS forgetting factor.
    ENERGY_CUT     : Cumulative energy threshold for tap selection.
    PLOT_RATE      : Samples between live-plot refreshes.
    N_SAMPLES      : Number of samples to use (None = all).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Tunables
# ------------------------------------------------------------------ #
FRF_FILE = 'linear_predicted_G_values.csv'   # FRF CSV: [omega, ReG, ImG]
IO_FILE = 'data_hour.mat'                    # recorded I/O to replay
LAMBDA_FACTOR = 0.995                        # RLS forgetting factor
ENERGY_CUT = 0.99                            # keep >= 99% of |g| energy
PLOT_RATE = 100                              # samples between plot refreshes
N_SAMPLES = 100_000                          # limit samples (None = all)


def run_rls(
    frf_file: str = FRF_FILE,
    io_file: str = IO_FILE,
    lambda_factor: float = LAMBDA_FACTOR,
    energy_cut: float = ENERGY_CUT,
    plot_rate: int = PLOT_RATE,
    n_samples: int | None = N_SAMPLES,
):
    """
    Run RLS-based real-time FIR identification.

    Args:
        frf_file:       Path to FRF CSV (omega, ReG, ImG).
        io_file:        Path to MAT file with [time; y; u].
        lambda_factor:  RLS forgetting factor.
        energy_cut:     Cumulative energy threshold for tap trimming.
        plot_rate:      Live-plot refresh interval (samples).
        n_samples:      Max samples to use (None for all).

    Returns:
        Dictionary with h, rmse, nrmse, R2, yhat, err, Ts.
    """
    # 1) Load FRF from CSV (header row skipped)
    data = np.loadtxt(frf_file, delimiter=',', skiprows=1)
    omega = data[:, 0]
    ReG = data[:, 1]
    ImG = data[:, 2]
    G_pos = ReG + 1j * ImG

    # Uniform frequency grid (zero-padded FFT length)
    Npos = len(omega)
    omega_min = np.min(omega)
    omega_max = np.max(omega)
    Nfft = 2 ** math.ceil(math.log2(4 * Npos))
    omega_uni = np.linspace(omega_min, omega_max, Nfft // 2 + 1)

    # Complex interpolation (real and imaginary separately)
    G_uni = (np.interp(omega_uni, omega, G_pos.real)
             + 1j * np.interp(omega_uni, omega, G_pos.imag))

    # Build full Hermitian spectrum
    G_full = np.concatenate([np.conj(G_uni[-2:0:-1]), G_uni])

    # 2) Impulse response via IFFT
    g_full = np.real(np.fft.ifft(np.fft.ifftshift(G_full)))

    # Sampling frequency from frequency grid spacing
    Dw = omega_uni[1] - omega_uni[0]
    Fs = Dw * Nfft / (2 * np.pi)
    Ts = 1 / Fs

    # Trim impulse response by cumulative energy
    Etotal = np.sum(np.abs(g_full) ** 2)
    cumE = np.cumsum(np.abs(g_full) ** 2)
    L_indices = np.where(cumE / Etotal >= energy_cut)[0]
    if len(L_indices) > 0:
        L = L_indices[0] + 1
    else:
        L = len(g_full)
    L = max(L, 4)  # at least 4 taps

    # Apply Hann window and take first L taps
    w = signal.windows.hann(L)
    h_init = g_full[:L] * w

    print(f'[INFO] FIR length L = {L} (Ts = {Ts:.4g} s)')

    # 3) Load I/O data to replay
    io_data = loadmat(io_file)

    # Show all variables and grab the first numeric array
    for name, arr in io_data.items():
        if not name.startswith('__'):
            print(f'{name} => shape: {arr.shape}')
            mat = arr
            break

    # Columns: time, output, input
    if n_samples is not None:
        time = mat[0, :n_samples].ravel()
        y = mat[1, :n_samples].ravel()
        u = mat[2, :n_samples].ravel()
    else:
        time = mat[0, :].ravel()
        y = mat[1, :].ravel()
        u = mat[2, :].ravel()
    print(time.shape, y.shape, u.shape)

    # Calculate dt from time array
    if len(time) > 1:
        dt_values = np.diff(time)
        dt = np.mean(dt_values)
        # Check if time steps are reasonably uniform
        if np.max(np.abs(dt_values - dt)) > 0.01 * dt:
            print('Warning: Non-uniform time steps detected in input data.')
    else:
        dt = Ts  # Fallback to FIR sampling time

    print(f"Loaded data with {len(time)} samples, dt = {dt:.4g} s")

    N = len(u)

    # 4) RLS initialisation
    h = h_init.copy()
    P = 1e4 * np.eye(L)       # covariance matrix
    phi = np.zeros(L)         # regressor buffer

    yhat = np.zeros(N)
    err = np.zeros(N)

    # 5) Real-time (offline replay) loop with live plotting
    plt.figure(figsize=(10, 12), num='Real-Time FIR Identification')

    ax1 = plt.subplot(3, 1, 1)
    h_meas, = ax1.plot([], [], 'k', label='Measured y')
    h_pred, = ax1.plot([], [], 'r--', label='Predicted yhat')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('y')
    ax1.set_title('Output vs Prediction')

    ax2 = plt.subplot(3, 1, 2)
    h_err, = ax2.plot([], [], 'b', label='Error e')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Error')
    ax2.set_title('Prediction Error')

    ax3 = plt.subplot(3, 1, 3)
    h_para, = ax3.plot([], [], 'm-o', label='h')
    ax3.grid(True)
    ax3.legend(loc='upper right')
    ax3.set_xlabel('parameter index')
    ax3.set_ylabel('magnitude')
    ax3.set_title('FIR Coefficients Magnitude')
    ax3.set_xlim(1, L)

    plt.ion()
    plt.show()

    for n in range(N):
        # Update regressor: phi[n] = [u[n], u[n-1], ..., u[n-L+1]]
        phi = np.roll(phi, 1)
        phi[0] = u[n]

        if n >= L - 1:
            # Prediction
            yhat[n] = np.dot(phi, h)
            err[n] = y[n] - yhat[n]

            # RLS update
            K = np.dot(P, phi) / (lambda_factor
                                  + np.dot(phi, np.dot(P, phi)))
            h = h + K * err[n]
            P = (P - np.outer(K, np.dot(phi, P))) / lambda_factor
        else:
            yhat[n] = 0
            err[n] = y[n]

        # Live plot refresh
        if (n % plot_rate == 0) or (n == N - 1):
            n_range = np.arange(n + 1)
            h_meas.set_data(n_range, y[:n + 1])
            h_pred.set_data(n_range, yhat[:n + 1])
            h_err.set_data(n_range, err[:n + 1])

            x_idx = np.arange(1, L + 1)
            h_para.set_data(x_idx, h)
            ax3.relim()
            ax3.autoscale_view()

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            plt.draw()
            plt.pause(0.001)

    # 6) Error metrics
    rmse = np.sqrt(np.mean(err[L:] ** 2))
    ynorm = y[L:] - np.mean(y[L:])
    nrmse = 1 - np.linalg.norm(err[L:]) / np.linalg.norm(ynorm)
    R2 = 1 - np.sum(err[L:] ** 2) / np.sum(ynorm ** 2)

    print('\n=====  FINAL ERROR  ====================================')
    print(f'RMSE   = {rmse:.4g}')
    print(f'NRMSE  = {nrmse * 100:.2f} %')
    print(f'R^2    = {R2:.3f}')

    # Save results
    savemat('fir_rt_results.mat', {
        'h': h, 'rmse': rmse, 'nrmse': nrmse, 'R2': R2,
        'yhat': yhat, 'err': err, 'Ts': Ts,
    })

    plt.ioff()
    plt.show()

    return {
        'h': h, 'rmse': float(rmse), 'nrmse': float(nrmse),
        'R2': float(R2), 'yhat': yhat, 'err': err, 'Ts': float(Ts),
    }


if __name__ == "__main__":
    run_rls()

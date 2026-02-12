"""
Least-squares frequency-response fitting.

Fits a rational polynomial model to measured transfer-function data using
``scipy.optimize.least_squares``.  Includes Bode/Nyquist plotting and
Hampel-filtered MSE evaluation.

Model: (b1*s^2 + b2) / (p1*s^4 + p2*s^3 + p3*s^2 + p4*s + p5)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import warnings

from src.utils.data_io import load_bode_data
from src.utils.filters import hampel_filter

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
#                     First-order model (example)                             #
# --------------------------------------------------------------------------- #

def model(params, omega):
    """First-order transfer function: G(jw) = K / (1 + jw*tau)."""
    K, tau = params
    return K / (1 + 1j * omega * tau)


def residuals(params, omega, G_meas):
    """Concatenated real + imaginary residuals for least_squares."""
    G_pred = model(params, omega)
    res = np.hstack([np.real(G_pred - G_meas), np.imag(G_pred - G_meas)])
    return res


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #

def main():
    # --- 1) Load and pre-process data ---------------------------
    DEFAULT_DIR = Path("data_prepare")
    dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    dat_files = sorted(dir_path.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in '{dir_path}'")
    omega_list, mag_list, phase_list = [], [], []
    for f in dat_files:
        w, m, p = load_bode_data(f)
        omega_list.append(w)
        mag_list.append(m)
        phase_list.append(p)
    omega = np.hstack(omega_list)
    mag   = np.hstack(mag_list)
    phase = np.hstack(phase_list)
    idx = np.argsort(omega)
    omega, mag, phase = omega[idx], mag[idx], phase[idx]
    G_meas = mag * np.exp(1j * phase)

    # --- 2) Polynomial model fitting (lsqpmlin-equivalent) ------
    # Model: (b1*s^2 + b2) / (p1*s^4 + p2*s^3 + p3*s^2 + p4*s + p5)
    b1 = 1.0
    # Search the first 60% of data for the minimum-magnitude point
    # and set b2 = omega_min^2 + 0.01
    cutoff = int(len(omega) * 0.6)
    zpid = np.argmin(np.abs(G_meas[:cutoff]))
    b2 = omega[zpid]**2 + 0.01

    # Frequency-domain variable s = j*omega
    s = 1j * omega

    def model_poly(p, s):
        num = b1 * s**2 + b2
        den = p[0]*s**4 + p[1]*s**3 + p[2]*s**2 + p[3]*s + p[4]
        return num / den

    def resid_poly(p):
        R = model_poly(p, s) - G_meas
        return np.hstack([R.real, R.imag])

    # Multiple random initial values -- keep the best solution
    best_resnorm = np.inf
    best_p = None
    for _ in range(50):
        p0 = np.random.rand(5) * 1e5
        sol = least_squares(resid_poly, p0)
        resnorm = np.sum(sol.fun**2)
        if resnorm < best_resnorm:
            best_resnorm = resnorm
            best_p = sol.x

    print("Best raw parameters:", best_p)

    # --- Prepare fitted curves for plotting ------------------
    N_TEST = 500
    omega_test = np.logspace(np.log10(omega.min()),
                             np.log10(omega.max()),
                             N_TEST)
    s_test = 1j * omega_test
    G_fit = model_poly(best_p, s_test)

    y_mag_db = 20 * np.log10(mag)
    y_mag_fit = 20 * np.log10(np.abs(G_fit))
    y_phase = phase
    y_phase_fit = np.angle(G_fit)

    # --- 3) Bode magnitude plot -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(omega, y_mag_db, "b*", label="Observed")
    ax.semilogx(omega_test, y_mag_fit, "r-", lw=2, label="lsqfit")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("bode_mag_lsq.png", dpi=300)

    # --- 4) Bode phase plot -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(omega, y_phase, "b*", label="Observed")
    ax.semilogx(omega_test, y_phase_fit, "r-", lw=2, label="lsqfit")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("Phase [rad]")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("bode_phase_lsq.png", dpi=300)

    # --- 5) Nyquist plot --------------------------------------
    G_dataset = G_meas
    order = np.argsort(omega_test)
    plt.figure(figsize=(6, 6))
    plt.plot(G_dataset.real, G_dataset.imag, 'b*', label='Data')
    plt.plot(G_fit.real[order], G_fit.imag[order], 'r-', label='lsqfit')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Nyquist')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nyquist_lsq.png", dpi=300)
    plt.show()

    # --- Hampel-filtered Nyquist MSE --------------------------
    # Original Nyquist data (unfiltered)
    G_meas = mag * np.exp(1j * phase)

    # Model prediction at original frequencies
    s_orig = 1j * omega
    G_pred = model_poly(best_p, s_orig)

    # Apply Hampel filter to measured real/imaginary parts
    G_real_filt = hampel_filter(G_meas.real, window_size=7, n_sigmas=3)
    G_imag_filt = hampel_filter(G_meas.imag, window_size=7, n_sigmas=3)
    G_filt = G_real_filt + 1j * G_imag_filt

    # Compute complex MSE
    mse = np.mean(np.abs(G_filt - G_pred)**2)
    print(f"Nyquist MSE (after Hampel filter): {mse:.4e}")

    # Plot filtered vs. fit
    order = np.argsort(omega)
    plt.figure(figsize=(6, 6))
    plt.plot(G_filt.real, G_filt.imag, 'b*', label='Filtered Data')
    plt.plot(G_pred.real[order], G_pred.imag[order], 'r-', lw=2,
             label=f'LSQ-Poly Fit (MSE={mse:.3e})')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Nyquist with Hampel-Filtered MSE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nyquist_lsq_mse.png", dpi=300)
    plt.show()

    # --- Save predicted data to CSV ---
    output_data = np.column_stack((omega_test, G_filt.real, G_filt.imag))
    csv_filepath = Path("predicted_G_values.csv")
    header = "omega,Re_G,Im_G"
    np.savetxt(csv_filepath, output_data, delimiter=",",
               header=header, comments='')


if __name__ == "__main__":
    main()

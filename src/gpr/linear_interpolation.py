"""
Linear interpolation of frequency-response data with Hampel filtering.

Applies a data-modifying Hampel filter to clean outliers from the complex
transfer function, then fits a linear interpolant on log-frequency for the
real and imaginary parts independently.  Produces Nyquist plots and CSV
output of grid predictions.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

from src.utils.data_io import load_bode_data
from src.utils.filters import hampel_filter, hampel_mask

warnings.filterwarnings("ignore")


def main() -> None:
    N_TEST_POINTS = 50000
    TEST_FILES = {
        "SKE2024_data18-Apr-2025_1205.dat",
    }

    # Load data ----------------------------------------------------------------
    DEFAULT_DIR = Path("data/gp_training")
    dir_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    dat_files = sorted(dir_path.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in '{dir_path}'")

    train_files = [p for p in dat_files if p.name not in TEST_FILES]
    test_files  = [p for p in dat_files if p.name in TEST_FILES]
    if not train_files or not test_files:
        raise RuntimeError("Train / Test split failed. Check file names.")

    def stack(files):
        w_l, m_l, p_l = [], [], []
        for f in files:
            w, m, p = load_bode_data(f)
            w_l.append(w); m_l.append(m); p_l.append(p)
        w = np.hstack(w_l); m = np.hstack(m_l); p = np.hstack(p_l)
        idx = np.argsort(w)
        return w[idx], m[idx], p[idx]

    # Train and test raw data --------------------------------------------------
    w_tr, mag_tr, ph_tr = stack(train_files)
    w_te, mag_te, ph_te = stack(test_files)

    G_tr = mag_tr * np.exp(1j * ph_tr)
    G_te = mag_te * np.exp(1j * ph_te)

    # Log-frequency ------------------------------------------------------------
    X_tr = np.log10(w_tr).reshape(-1, 1)
    X_te = np.log10(w_te).reshape(-1, 1)

    # Hampel filter on data (replaces outliers with local median) --------------
    G_tr_f = hampel_filter(G_tr.real) + 1j * hampel_filter(G_tr.imag)
    G_te_f = hampel_filter(G_te.real) + 1j * hampel_filter(G_te.imag)

    # Linear interpolation model -----------------------------------------------
    interp_r = interp1d(X_tr.ravel(), G_tr_f.real, kind='linear', fill_value="extrapolate")
    interp_i = interp1d(X_tr.ravel(), G_tr_f.imag, kind='linear', fill_value="extrapolate")

    # Continuous grid for smooth curve -----------------------------------------
    w_grid = np.logspace(np.log10(min(w_tr.min(), w_te.min())),
                         np.log10(max(w_tr.max(), w_te.max())),
                         N_TEST_POINTS)
    X_grid = np.log10(w_grid).reshape(-1, 1)

    r_grid = interp_r(X_grid.ravel())
    i_grid = interp_i(X_grid.ravel())
    G_grid = r_grid + 1j * i_grid

    # Prediction on each set ---------------------------------------------------
    r_tr = interp_r(X_tr.ravel())
    i_tr = interp_i(X_tr.ravel())
    G_tr_pred = r_tr + 1j * i_tr

    r_te = interp_r(X_te.ravel())
    i_te = interp_i(X_te.ravel())
    G_te_pred = r_te + 1j * i_te

    # MSE (compute RMSE on filtered data) --------------------------------------
    err_tr = np.abs(G_tr_f - G_tr_pred)
    err_tr = np.where(np.isnan(err_tr), 0, err_tr)
    keep_tr = err_tr
    mse_tr = np.mean(err_tr ** 2)
    mse_tr = np.sqrt(mse_tr)  # Convert to RMSE for consistency

    err_te = np.abs(G_te_f - G_te_pred)
    err_te = np.where(np.isnan(err_te), 0, err_te)
    keep_te = err_te
    mse_te = np.mean(err_te ** 2)
    mse_te = np.sqrt(mse_te)  # Convert to RMSE for consistency
    print(f"MSE  (train): {mse_tr:.4e}")
    print(f"MSE  (test) : {mse_te:.4e}")

    # Nyquist plot -------------------------------------------------------------
    TITLE_FONTSIZE = 15
    AXIS_LABEL_FONTSIZE = 12
    LEGEND_FONTSIZE = 10

    fig, ax = plt.subplots(2, 1, figsize=(7, 10))

    # --- Train ---
    ax[0].plot(G_tr_f.real, G_tr_f.imag, 'b*', ms=6, label='Train')
    ax[0].plot(G_grid.real, G_grid.imag, 'r*', lw=4, label='Linear Interpolation')
    ax[0].set_title(f"Train Nyquist (MSE={mse_tr:.2e})", fontsize=TITLE_FONTSIZE)
    ax[0].set_xlabel('Re', fontsize=AXIS_LABEL_FONTSIZE)
    ax[0].set_ylabel('Im', fontsize=AXIS_LABEL_FONTSIZE)
    ax[0].set_xlim([-0.7, 0.7]); ax[0].set_ylim([-0.6, 0.2])
    ax[0].grid(True)
    ax[0].legend(fontsize=LEGEND_FONTSIZE)

    # --- Test ---
    ax[1].plot(G_te_f.real, G_te_f.imag, 'g*', ms=6, label='Test')
    ax[1].plot(G_grid.real, G_grid.imag, 'r*', lw=4, label='Linear Interpolation')
    ax[1].set_title(f"Test Nyquist (MSE={mse_te:.2e})", fontsize=TITLE_FONTSIZE)
    ax[1].set_xlabel('Re', fontsize=AXIS_LABEL_FONTSIZE)
    ax[1].set_ylabel('Im', fontsize=AXIS_LABEL_FONTSIZE)
    ax[1].set_xlim([-0.7, 0.7]); ax[1].set_ylim([-0.6, 0.2])
    ax[1].grid(True)
    ax[1].legend(fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    fig.savefig("linear_nyquist_train_test_interp.png", dpi=300)

    # Output grid data to CSV --------------------------------------------------
    grid_output_data = np.column_stack((w_grid, r_grid, i_grid))
    output_csv_path = "linear_grid_predictions.csv"
    header = "omega,Re_G,Im_G"
    np.savetxt(output_csv_path, grid_output_data, delimiter=",",
               header=header, comments='', fmt='%e')
    print(f"Grid predictions saved to {output_csv_path}")

    plt.show()


if __name__ == "__main__":
    main()

"""
ITGP regression on measured frequency-response data.

Iteratively-trimmed Gaussian Process (ITGP) smoothing with outlier-robust
processing.  Uses StandardScaler to pre-process log-frequency inputs and
a Hampel filter to exclude outliers when computing MSE.
"""

from pathlib import Path
import sys
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from robustgp import ITGP

from src.utils.data_io import load_bode_data
from src.utils.filters import hampel_mask

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#                               Hyperparameters                               #
# --------------------------------------------------------------------------- #
N_TEST_POINTS = 50000
TEST_FILENAMES = {
    "SKE2024_data18-Apr-2025_1205.dat",
}


# --------------------------------------------------------------------------- #
#                                  Utilities                                  #
# --------------------------------------------------------------------------- #
def prepare_inputs(omega: np.ndarray):
    """Scale log10(omega) with StandardScaler and return (X_scaled, scaler)."""
    X_raw = np.log10(omega).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    # -------------------- Load data --------------------
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/gp_training")

    train_om, train_mag, train_ph = [], [], []
    test_om, test_mag, test_ph = [], [], []

    for fp in sorted(data_dir.glob("SKE2024_data*.dat")):
        om, mag, ph = load_bode_data(fp)
        if fp.name in TEST_FILENAMES:
            test_om.append(om), test_mag.append(mag), test_ph.append(ph)
        else:
            train_om.append(om), train_mag.append(mag), train_ph.append(ph)

    if not train_om or not test_om:
        raise RuntimeError("Train / test split failed. Check data files.")

    # Concatenate and sort by ascending frequency
    omega_tr = np.concatenate(train_om)
    mag_tr = np.concatenate(train_mag)
    phase_tr = np.concatenate(train_ph)
    idx_tr = np.argsort(omega_tr)
    omega_tr, mag_tr, phase_tr = omega_tr[idx_tr], mag_tr[idx_tr], phase_tr[idx_tr]

    omega_te = np.concatenate(test_om)
    mag_te = np.concatenate(test_mag)
    phase_te = np.concatenate(test_ph)
    idx_te = np.argsort(omega_te)
    omega_te, mag_te, phase_te = omega_te[idx_te], mag_te[idx_te], phase_te[idx_te]

    # -------------------- Input pre-processing --------------------
    X_tr, scaler = prepare_inputs(omega_tr)
    X_te = scaler.transform(np.log10(omega_te).reshape(-1, 1))

    # Target outputs (real and imaginary parts)
    y_tr_r = mag_tr * np.cos(phase_tr)
    y_tr_i = mag_tr * np.sin(phase_tr)
    y_te_r = mag_te * np.cos(phase_te)
    y_te_i = mag_te * np.sin(phase_te)

    # -------------------- ITGP fit --------------------
    res_r = ITGP(X_tr, y_tr_r, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1)
    res_i = ITGP(X_tr, y_tr_i, alpha1=0.50, alpha2=0.975, nsh=2, ncc=2, nrw=1)
    gp_r, gp_i = res_r.gp, res_i.gp

    # -------------------- Prediction --------------------
    # Train and test predictions
    y_tr_r_pred, _ = gp_r.predict(X_tr)
    y_tr_r_pred = y_tr_r_pred.ravel()
    y_tr_i_pred, _ = gp_i.predict(X_tr)
    y_tr_i_pred = y_tr_i_pred.ravel()
    y_te_r_pred, _ = gp_r.predict(X_te)
    y_te_r_pred = y_te_r_pred.ravel()
    y_te_i_pred, _ = gp_i.predict(X_te)
    y_te_i_pred = y_te_i_pred.ravel()

    # Dense grid for smooth curve
    omega_dense = np.logspace(np.log10(min(omega_tr.min(), omega_te.min())),
                              np.log10(max(omega_tr.max(), omega_te.max())),
                              N_TEST_POINTS)
    X_dense = scaler.transform(np.log10(omega_dense).reshape(-1, 1))
    y_dense_r, _ = gp_r.predict(X_dense)
    y_dense_i, _ = gp_i.predict(X_dense)
    H_dense = (y_dense_r + 1j * y_dense_i).ravel()

    # -------------------- MSE (Hampel-filtered) --------------------
    # Complex gain (train and test)
    G_tr_true = y_tr_r + 1j * y_tr_i
    G_tr_pred = y_tr_r_pred + 1j * y_tr_i_pred
    G_te_true = y_te_r + 1j * y_te_i
    G_te_pred = y_te_r_pred + 1j * y_te_i_pred

    # Hampel filter on |G_true| sorted by frequency
    mask_tr = hampel_mask(G_tr_true)
    mask_te = hampel_mask(G_te_true)

    # Compute error after filtering
    err_tr = np.abs(G_tr_true - G_tr_pred)
    err_te = np.abs(G_te_true - G_te_pred)
    mse_tr = np.sqrt(np.mean(err_tr[mask_tr] ** 2))
    mse_te = np.sqrt(np.mean(err_te[mask_te] ** 2))

    # -------------------- Nyquist plot --------------------
    order = np.argsort(omega_dense)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    # Train
    ax1.plot(G_tr_true.real, G_tr_true.imag, "b.", label="Train data")
    ax1.plot(G_tr_pred.real, G_tr_pred.imag, "r.", label="ITGP pred")
    ax1.set_title(f"Nyquist - Train | MSE={mse_tr:.3e}")
    ax1.set_xlabel("Re{G}", fontsize=14)
    ax1.set_ylabel("Im{G}", fontsize=14)
    ax1.set_xlim([-0.7, 0.7])
    ax1.set_ylim([-0.6, 0.2])
    ax1.grid(True)
    ax1.legend()

    # Test
    ax2.plot(G_te_true.real, G_te_true.imag, "g^", label="Test data")
    ax2.plot(G_te_pred.real, G_te_pred.imag, "ys", label="ITGP pred")
    ax2.set_title(f"Nyquist - Test | MSE={mse_te:.3e}")
    ax2.set_xlabel("Re{G}", fontsize=14)
    ax2.set_ylabel("Im{G}", fontsize=14)
    ax2.set_xlim([-0.7, 0.7])
    ax2.set_ylim([-0.6, 0.2])
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    out_png = Path("./gp/output/test_itgp_nyquist_train_test.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.show()

    # -------------------- Console --------------------
    print(f"Train MSE : {mse_tr:.4e}")
    print(f"Test  MSE : {mse_te:.4e}")
    print(f"Figure saved to {out_png}")


if __name__ == "__main__":
    main()

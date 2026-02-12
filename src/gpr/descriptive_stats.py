"""
Descriptive statistics and data visualisation for frequency-response data.

Loads all Bode data files, computes the complex transfer function, and
produces a frequency-coloured Nyquist scatter plot.
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from src.utils.data_io import load_bode_data, load_all_bode_data

warnings.filterwarnings("ignore")


def load_all_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load all .dat files and return concatenated data with filenames.

    This is a thin wrapper around :func:`load_all_bode_data` kept for
    backward compatibility.
    """
    return load_all_bode_data(data_dir)


def main():
    DATA_DIR = Path("./data/gp_training")

    # Load all data files
    omega, mag, phase, filenames = load_all_data(DATA_DIR)

    if omega.size == 0:
        raise RuntimeError("No data files found. Check data directory.")

    # Convert to complex transfer function
    G = mag * np.exp(1j * phase)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Colour mapping based on log frequency
    log_omega = np.log10(omega)
    norm = Normalize(vmin=log_omega.min(), vmax=log_omega.max())

    # Nyquist diagram with log-frequency colouring (dark blue -> green -> yellow)
    scatter = ax.scatter(G.real, G.imag, c=log_omega, cmap='viridis',
                         s=20, alpha=0.9)

    # Axis limits
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.1])

    # Labels and title
    ax.set_xlabel("Re{G}", fontsize=18)
    ax.set_ylabel("Im{G}", fontsize=18)
    ax.set_title("Nyquist Plot - All Data Files (Color: log10(omega))",
                 fontsize=20)

    # Tick sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Colour bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log10(omega) [rad/s]', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    # Grid
    ax.grid(True, ls="--", alpha=0.3)

    plt.tight_layout()

    # Save figure
    out_png = Path("./gp/output/nyquist_all_data_raw.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"Loaded {len(filenames)} data files:")
    for fname in filenames:
        print(f"  - {fname}")


if __name__ == "__main__":
    main()

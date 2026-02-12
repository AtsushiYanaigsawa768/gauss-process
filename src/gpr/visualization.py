"""
Visualization functions for Gaussian Process regression results.

Provides Nyquist and metric-computation helpers used after GP fitting
on complex-valued frequency response data.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.visualization.plot_styles import configure_plot_style


def plot_gp_results(
    omega: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray],
    title: str,
    output_path: Path,
) -> Dict[str, float]:
    """Compute GP regression quality metrics (RMSE, R-squared).

    This function calculates error metrics between true and predicted
    values.  No figure is produced (real/imaginary parts are plotted
    together in :func:`plot_complex_gp` instead).

    Args:
        omega: Angular frequencies (unused, kept for API compatibility).
        y_true: Ground-truth values.
        y_pred: Predicted values.
        y_std: Predictive standard deviation (unused, kept for API).
        title: Plot title (unused, kept for API).
        output_path: Output path (unused, kept for API).

    Returns:
        Dictionary with keys ``'rmse'`` and ``'r2'``.
    """
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {'rmse': rmse, 'r2': r2}


def plot_complex_gp(
    omega: np.ndarray,
    G_true: np.ndarray,
    G_pred: np.ndarray,
    G_std_real: Optional[np.ndarray],
    G_std_imag: Optional[np.ndarray],
    output_prefix: Path,
    save_eps: bool = True,
) -> None:
    """Plot complex-valued GP results as a Nyquist diagram.

    Produces a Nyquist plot comparing measured and GP-predicted transfer
    functions.  Optionally draws 2-sigma confidence ellipses at selected
    frequency points.

    Args:
        omega: Angular frequencies.
        G_true: True (measured) complex transfer function.
        G_pred: GP-predicted complex transfer function.
        G_std_real: Predictive std of the real part (may be None).
        G_std_imag: Predictive std of the imaginary part (may be None).
        output_prefix: File path prefix (extension is appended automatically).
        save_eps: If True, save an EPS file alongside the PNG.
    """
    configure_plot_style()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(np.real(G_true), np.imag(G_true), 'k.', markersize=14,
            label='Measured', alpha=0.6)
    ax.plot(np.real(G_pred), np.imag(G_pred), 'r-', linewidth=4.0,
            label='GP mean')

    if G_std_real is not None and G_std_imag is not None:
        # Draw confidence ellipses at selected frequency indices
        n_ellipses = min(20, len(omega))
        indices = np.linspace(0, len(omega) - 1, n_ellipses, dtype=int)
        for i in indices:
            theta = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = np.real(G_pred[i]) + 2 * G_std_real[i] * np.cos(theta)
            ellipse_y = np.imag(G_pred[i]) + 2 * G_std_imag[i] * np.sin(theta)
            ax.plot(ellipse_x, ellipse_y, 'r-', alpha=0.2, linewidth=1.0)

    ax.set_xlabel('Real{G(jw)}', fontsize=32, fontweight='bold')
    ax.set_ylabel('Imag{G(jw)}', fontsize=32, fontweight='bold')
    ax.set_title('Nyquist Plot with GP Regression', fontsize=36,
                 fontweight='bold', pad=20)
    ax.legend(fontsize=26, framealpha=0.9, edgecolor='black', loc='best')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.axis('equal')
    ax.tick_params(labelsize=24, width=2.5, length=10)
    plt.tight_layout()

    # Save PNG
    png_path = str(output_prefix) + '_nyquist_gp.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    # Save EPS
    if save_eps:
        eps_path = str(output_prefix) + '_nyquist_gp.eps'
        plt.savefig(eps_path, format='eps', bbox_inches='tight')

    plt.close(fig)

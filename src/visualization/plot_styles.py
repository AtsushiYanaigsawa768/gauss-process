"""
Shared matplotlib plot style configuration.

Provides a single function to set consistent font sizes, line widths,
and tick parameters across all figures in the project.
"""

import matplotlib.pyplot as plt


def configure_plot_style() -> None:
    """Configure matplotlib rcParams for consistent, publication-quality figures.

    Sets large font sizes suitable for papers and presentations:
        - Base font: 22 pt
        - Axis labels: 30 pt
        - Title: 32 pt
        - Legend: 24 pt
        - Thick lines and tick marks
    """
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 30,
        'axes.titlesize': 32,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 24,
        'figure.titlesize': 34,
        'lines.linewidth': 3.5,
        'lines.markersize': 12,
        'axes.linewidth': 2.5,
        'grid.linewidth': 1.5,
        'xtick.major.width': 2.5,
        'ytick.major.width': 2.5,
        'xtick.major.size': 10,
        'ytick.major.size': 10,
    })

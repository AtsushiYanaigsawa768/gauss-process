#!/usr/bin/env python3
"""
Plot input/output time-domain signals from .mat data files.

This module provides visualization of recorded time-domain signals
used in system identification experiments.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from src.visualization.plot_styles import configure_plot_style
from src.frequency_transform.frf_estimator import load_time_u_y


def parse_time_window(time_window: str) -> float:
    """Parse time window string to seconds.

    Args:
        time_window: Duration string like "5s", "1min", "30min", "1h"
    Returns:
        Duration in seconds
    """
    match = re.match(
        r'^(\d+(?:\.\d+)?)\s*(s|sec|min|h|hr|hour)$',
        time_window.strip().lower()
    )
    if not match:
        raise ValueError(
            f"Invalid time_window format: '{time_window}'. "
            "Use e.g. '5s', '1min', '30min'"
        )
    value = float(match.group(1))
    unit = match.group(2)
    multipliers = {
        's': 1, 'sec': 1, 'min': 60,
        'h': 3600, 'hr': 3600, 'hour': 3600,
    }
    return value * multipliers[unit]


def plot_io_signals(mat_file_path: str, time_window: str = "5s",
                    output_path: str = None) -> str:
    """Plot input/output time-domain signals from .mat data.

    Args:
        mat_file_path: Path to .mat file containing I/O data
        time_window: Duration to plot (e.g., "5s", "1min", "30min")
        output_path: Output PNG path (auto-generated if None)
    Returns:
        Path to saved PNG file
    """
    configure_plot_style()

    # Load data
    t, u, y = load_time_u_y(Path(mat_file_path), y_col=0)

    # Parse time window
    window_sec = parse_time_window(time_window)

    # Find data within window
    t_start = t[0]
    mask = (t - t_start) <= window_sec
    t_plot = t[mask] - t_start
    u_plot = u[mask]
    y_plot = y[mask]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Input signal
    ax1.plot(t_plot, u_plot, 'b-', linewidth=1.5, label='Input u(t)')
    ax1.set_ylabel('Input u(t)')
    ax1.set_title(f'I/O Signals -- first {time_window}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Output signal
    ax2.plot(t_plot, y_plot, 'r-', linewidth=1.5, label='Output y(t)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Output y(t)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    if output_path is None:
        stem = Path(mat_file_path).stem
        output_path = f"io_signals_{stem}_{time_window}.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot I/O signals from .mat file')
    parser.add_argument('mat_file', help='Path to .mat file')
    parser.add_argument('--time-window', default='5s',
                        help='Time window (e.g. 5s, 1min, 30min)')
    parser.add_argument('--output', default=None, help='Output PNG path')
    args = parser.parse_args()

    result = plot_io_signals(args.mat_file, args.time_window, args.output)
    print(f"Saved plot to: {result}")

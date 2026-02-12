"""
Data I/O utilities for loading frequency response and time-series data.

Provides functions for reading Bode plot data from .dat files and
time-series data from MATLAB .mat files used in system identification.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat


def load_bode_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Bode plot data from a comma-separated .dat file.

    Expects a file with three rows (not columns):
        Row 0: angular frequency omega [rad/s]
        Row 1: magnitude |G(jw)|
        Row 2: phase angle [rad]

    Args:
        filepath: Path to the .dat file.

    Returns:
        Tuple of (omega, magnitude, phase) as 1-D arrays.
    """
    data = np.loadtxt(filepath, delimiter=",")
    omega, mag, phase = data
    return omega, mag, phase


def load_all_bode_data(
    data_dir: Path,
    pattern: str = "SKE2024_data*.dat",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load all matching .dat files and return concatenated Bode data.

    Args:
        data_dir: Directory containing .dat files.
        pattern: Glob pattern to match data files.

    Returns:
        Tuple of (omega, magnitude, phase, filenames).
        Arrays are concatenated across all files.  *filenames* is a list
        of matched file names (stem + extension).
    """
    all_omega: List[np.ndarray] = []
    all_mag: List[np.ndarray] = []
    all_phase: List[np.ndarray] = []
    filenames: List[str] = []

    for fp in sorted(data_dir.glob(pattern)):
        omega, mag, phase = load_bode_data(fp)
        all_omega.append(omega)
        all_mag.append(mag)
        all_phase.append(phase)
        filenames.append(fp.name)

    if not all_omega:
        return np.empty(0), np.empty(0), np.empty(0), []

    omega_concat = np.concatenate(all_omega)
    mag_concat = np.concatenate(all_mag)
    phase_concat = np.concatenate(all_phase)

    return omega_concat, mag_concat, phase_concat, filenames


# ---------------------------------------------------------------------------
# MAT file helpers
# ---------------------------------------------------------------------------

def _ravel1d(x) -> Optional[np.ndarray]:
    """Try to squeeze and ravel *x* into a finite 1-D float array."""
    try:
        arr = np.asarray(x).squeeze()
        arr = arr.astype(float).ravel()
        return arr if arr.size > 0 and np.isfinite(arr).all() else None
    except Exception:
        return None


def load_mat_time_series(
    mat_path: Path,
    y_col: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, input, and output vectors from a MATLAB .mat file.

    Accepts either:
      (A) Separate variables ``t``, ``u``, ``y`` (``y`` may be 1-D or 2-D;
          use *y_col* to select a column).
      (B) A 3xN or Nx3 numeric array (preferably named ``output``) with
          rows/columns ordered ``[t, y, u]`` (time, output, input).

    Args:
        mat_path: Path to the .mat file.
        y_col: Column index to select from ``y`` when it is 2-D.

    Returns:
        Tuple of (t, u, y) as 1-D float arrays.

    Raises:
        RuntimeError: If compatible data cannot be located in the file.
        ValueError: If *y_col* is out of range for a 2-D ``y``.
    """
    data = loadmat(mat_path)

    # --- Path (A): explicit t / u / y variables ---
    t = _ravel1d(data.get("t"))
    u = _ravel1d(data.get("u"))
    y_raw = data.get("y")
    if t is not None and u is not None and y_raw is not None:
        y_arr = np.asarray(y_raw)
        if y_arr.ndim == 1:
            y = _ravel1d(y_arr)
        elif y_arr.ndim == 2:
            if y_arr.shape[1] == 1:
                y = _ravel1d(y_arr[:, 0])
            else:
                if not (0 <= y_col < y_arr.shape[1]):
                    raise ValueError(
                        f"y_col={y_col} out of range for y with shape {y_arr.shape}"
                    )
                y = _ravel1d(y_arr[:, y_col])
        else:
            y = None
        if y is not None and len(t) > 1 and len(u) == len(t) and len(y) == len(t):
            return t, u, y

    # --- Path (B): 3xN or Nx3 array ---
    def _try_matrix(arr):
        arr = np.asarray(arr)
        if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
            return None
        if arr.shape[0] == 3:
            t_v, y_v, u_v = arr[0], arr[1], arr[2]
        elif arr.shape[1] == 3:
            t_v, y_v, u_v = arr[:, 0], arr[:, 1], arr[:, 2]
        else:
            return None
        return _ravel1d(t_v), _ravel1d(u_v), _ravel1d(y_v)

    # Prefer the variable named "output"
    if "output" in data:
        candidate = _try_matrix(data["output"])
        if candidate is not None:
            t_c, u_c, y_c = candidate
            if t_c is not None and u_c is not None and y_c is not None:
                return t_c, u_c, y_c

    # Scan remaining top-level variables
    for key, value in data.items():
        if key.startswith("__"):
            continue
        candidate = _try_matrix(value)
        if candidate is not None:
            t_c, u_c, y_c = candidate
            if t_c is not None and u_c is not None and y_c is not None:
                return t_c, u_c, y_c

    raise RuntimeError(
        f"Could not locate compatible [t,u,y] in {mat_path}.\n"
        f"Expected either t/u/y variables (with y 1D or 2D) "
        f"or a 3xN (or Nx3) array with rows/cols [t,y,u]."
    )

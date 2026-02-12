"""
Outlier detection and filtering utilities.

Provides Hampel filter implementations for robust outlier detection and
replacement in 1-D signals. Used across GP regression and data preprocessing
pipelines.
"""

import numpy as np


# Scale factor for Gaussian distribution: k ~ 1 / scipy.stats.norm.ppf(0.75)
HAMPEL_SCALE_FACTOR = 1.4826

# Threshold for treating MAD as zero (floating-point tolerance)
ZERO_MAD_THRESHOLD = 1e-9


def hampel_filter(x: np.ndarray, window_size: int = 7, n_sigmas: int = 3) -> np.ndarray:
    """Replace outliers in a 1-D array with the local median (Hampel filter).

    Operates on a copy of *x*.  For each element, a sliding window of
    *window_size* centered on the element is used to compute the median and
    the Median Absolute Deviation (MAD).  Points that deviate from the median
    by more than *n_sigmas* scaled MADs are replaced with the median.

    NaN values are skipped (left unchanged).  Inf values are replaced by the
    local median when a valid median can be computed.

    Args:
        x: Input 1-D array.
        window_size: Total window width (must be odd for symmetric windows).
        n_sigmas: Number of scaled MADs beyond which a point is an outlier.

    Returns:
        A copy of *x* with outliers replaced by local medians.
    """
    x_out = x.copy()
    k = HAMPEL_SCALE_FACTOR
    L = len(x_out)
    if L == 0:
        return x_out

    half_window = window_size // 2

    for i in range(L):
        current_val_at_i = x_out[i]

        # Skip NaN values -- they will not be replaced
        if np.isnan(current_val_at_i):
            continue

        # Define window around the current point
        s = max(0, i - half_window)
        e = min(L, i + half_window + 1)
        window_values = x_out[s:e]

        # Treat Inf as NaN for robust statistics
        window_for_stats = window_values.copy()
        window_for_stats[np.isinf(window_for_stats)] = np.nan

        med = np.nanmedian(window_for_stats)

        # If median could not be computed (all NaNs), skip
        if np.isnan(med):
            continue

        # Replace Inf with the computed median
        if np.isinf(current_val_at_i):
            x_out[i] = med
            continue

        # Calculate Median Absolute Deviation (MAD)
        abs_devs_from_med = np.abs(window_for_stats - med)
        median_abs_dev = np.nanmedian(abs_devs_from_med)

        # If MAD could not be computed, skip
        if np.isnan(median_abs_dev):
            continue

        mad_scaled = k * median_abs_dev

        is_outlier = False
        if mad_scaled > 0:
            # Standard case: compare against scaled MAD
            if np.abs(current_val_at_i - med) > n_sigmas * mad_scaled:
                is_outlier = True
        else:
            # MAD is zero (all valid data equal to median).
            # Mark as outlier if different from median beyond tolerance.
            if np.abs(current_val_at_i - med) > ZERO_MAD_THRESHOLD:
                is_outlier = True

        if is_outlier:
            x_out[i] = med

    return x_out


def hampel_mask(x: np.ndarray, win: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Return a boolean mask where True marks *non-outlier* elements.

    Unlike :func:`hampel_filter`, this variant does **not** modify data.
    Instead it returns a boolean keep-mask suitable for indexing.

    NaN and Inf values are always marked as False (outliers / not kept).

    Args:
        x: Input 1-D array.
        win: Total window width.
        n_sigmas: Number of scaled MADs for the outlier threshold.

    Returns:
        Boolean array of the same length as *x*.
    """
    k = HAMPEL_SCALE_FACTOR
    n = x.size
    if n == 0:
        return np.array([], dtype=bool)

    # Initialize: True for finite values, False for NaN/Inf
    keep = np.isfinite(x)

    # Replace non-finite values with NaN for statistics
    x_for_stats = x.copy()
    x_for_stats[~keep] = np.nan

    half_window = win // 2

    for i in range(n):
        if not keep[i]:
            continue

        # Window bounds
        i_min = max(0, i - half_window)
        i_max = min(n, i + half_window + 1)

        window_curr_for_stats = x_for_stats[i_min:i_max]

        med = np.nanmedian(window_curr_for_stats)
        if np.isnan(med):
            continue

        abs_devs_from_med = np.abs(window_curr_for_stats - med)
        median_abs_dev = np.nanmedian(abs_devs_from_med)
        if np.isnan(median_abs_dev):
            continue

        sigma_hampel = k * median_abs_dev

        if sigma_hampel > 0:
            if np.abs(x[i] - med) > n_sigmas * sigma_hampel:
                keep[i] = False
        else:
            # All non-NaN window values identical to median
            if np.abs(x[i] - med) > 1e-1:
                keep[i] = False

    return keep

#!/usr/bin/env python3
"""
grid_search.py

Grid search hyperparameter optimization for GP kernels.

Extracted from the GaussianProcessRegressor._grid_search_hyperparameters
and _get_default_param_grids methods in unified_pipeline.py.
Computational logic is identical -- same grid sizes (30 points),
same random seed (42), same evaluation logic.
"""

from itertools import product
from typing import Dict, Optional

import numpy as np

from src.gpr.kernels import Kernel


def get_default_param_grids(kernel: Kernel) -> Dict:
    """Get default parameter grids based on kernel type and bounds.

    Uses 30 points per parameter. Log scale for parameters whose
    bound ratio exceeds 100.

    Note: Noise variance is NOT included in grid search (kept fixed).

    Args:
        kernel: The GP kernel whose bounds define the grid.

    Returns:
        Dictionary mapping 'param_0', 'param_1', ... to 1-D arrays.
    """
    grids = {}

    # Kernel-specific grids based on bounds - 30 points for better coverage
    for i, (low, high) in enumerate(kernel.bounds):
        param_name = f'param_{i}'
        if low > 0 and high / low > 100:  # Log scale for large ranges
            grids[param_name] = np.logspace(np.log10(low), np.log10(high), 30)
        else:
            grids[param_name] = np.linspace(low, high, 30)

    return grids


def grid_search_hyperparameters(kernel: Kernel,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 noise_variance: float,
                                 param_grids: Optional[Dict] = None,
                                 max_combinations: int = 5000,
                                 validation_X: Optional[np.ndarray] = None,
                                 validation_y: Optional[np.ndarray] = None,
                                 y_scaler: object = None):
    """Grid search for kernel hyperparameters.

    Evaluates all combinations of kernel parameter grid values and
    selects the set that minimises either validation RMSE (if
    validation data is supplied) or negative log marginal likelihood.

    Noise variance is held fixed throughout the search.

    Args:
        kernel: GP kernel whose parameters will be optimised in-place.
        X_train: Training inputs (N x D).
        y_train: Training outputs (N,).
        noise_variance: Fixed observation noise variance.
        param_grids: Dictionary mapping parameter names to grid values.
                     If None, uses default grids via get_default_param_grids().
        max_combinations: Maximum number of combinations to try.
                          If exceeded, random sampling with seed 42 is used.
        validation_X: Validation inputs for error-based evaluation (optional).
        validation_y: Validation outputs for error-based evaluation (optional).
        y_scaler: Scaler for denormalizing predictions to original scale (optional).
    """
    import random

    # Determine evaluation method
    use_validation = validation_X is not None and validation_y is not None

    if use_validation:
        if y_scaler is not None:
            print(f"  Grid search: Using validation data (N={len(validation_y)}) for error-based evaluation (original scale)")
        else:
            print(f"  Grid search: Using validation data (N={len(validation_y)}) for error-based evaluation (normalized scale)")
    else:
        print(f"  Grid search: Using log marginal likelihood for evaluation")

    def compute_evaluation_metric(params):
        """Compute evaluation metric (validation RMSE if available, otherwise NLL).

        Note: Only kernel parameters are optimized. Noise variance is fixed.
        """
        kernel.set_params(params)
        # Use fixed noise variance (0 for pure GP) + small jitter for numerical stability

        K = kernel(X_train)
        K += (noise_variance) * np.eye(K.shape[0])

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y_train)
            alpha = np.linalg.solve(L.T, alpha)

            if use_validation:
                # Use validation error as metric
                # Compute predictions on validation data
                K_star = kernel(validation_X, X_train)
                y_pred = K_star @ alpha

                # Denormalize predictions if scaler is provided
                # NOTE: validation_y is already in original scale (not normalized)
                if y_scaler is not None:
                    # Convert predictions to original scale
                    y_pred_original = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                    # validation_y is already in original scale, use as-is
                    y_val_original = validation_y
                    # Root mean squared error in original scale
                    rmse = np.sqrt(np.mean((y_val_original - y_pred_original) ** 2))
                else:
                    # Both in same scale, compute RMSE directly
                    rmse = np.sqrt(np.mean((validation_y - y_pred) ** 2))
                return rmse
            else:
                # Use negative log marginal likelihood
                nll = 0.5 * (y_train @ alpha)
                nll += np.sum(np.log(np.diag(L)))
                nll += 0.5 * len(y_train) * np.log(2 * np.pi)
                return nll
        except np.linalg.LinAlgError:
            return 1e10

    # Get default grids if not provided
    if param_grids is None:
        param_grids = get_default_param_grids(kernel)

    # NOTE: Noise variance is NOT included in grid search (fixed at initial value)
    # This reduces the search space and prevents overfitting to noise characteristics
    print(f"  Noise variance: {noise_variance:.3e} (fixed, not optimized)")

    # Extract kernel parameter grids
    kernel_param_names = []
    kernel_param_grids = []

    # Get parameter names from kernel
    current_params = kernel.get_params()

    # Determine metric name
    metric_name = "RMSE" if use_validation else "NLL"

    # Diagnostic: Evaluate initial parameters before grid search
    if use_validation:
        initial_metric = compute_evaluation_metric(current_params)
        print(f"  Initial (pre-grid-search) {metric_name}: {initial_metric:.6e}")
        print(f"  Initial params: {current_params}, noise: {noise_variance:.3e} (fixed)")

    for i, (low, high) in enumerate(kernel.bounds):
        param_name = f'param_{i}'
        if param_name in param_grids:
            grid = param_grids[param_name]
        else:
            # Create default grid based on bounds - 30 points for better coverage
            if low > 0 and high / low > 100:  # Log scale for large ranges
                grid = np.logspace(np.log10(low), np.log10(high), 30)
            else:
                grid = np.linspace(low, high, 30)
        kernel_param_names.append(param_name)
        kernel_param_grids.append(grid)

    # Generate all combinations (kernel parameters only, no noise variance)
    all_combinations = list(product(*kernel_param_grids))
    n_combinations = len(all_combinations)

    print(f"  Grid search: {n_combinations} combinations to evaluate...")

    # If too many combinations, use random sampling
    if n_combinations > max_combinations:
        print(f"  Too many combinations ({n_combinations}), randomly sampling {max_combinations}...")
        random.seed(42)  # Set seed for reproducibility
        all_combinations = random.sample(all_combinations, max_combinations)
        n_combinations = max_combinations

    # Evaluate all combinations
    best_params = None
    best_metric = np.inf

    for i, combination in enumerate(all_combinations):
        kernel_params = np.array(combination)

        metric = compute_evaluation_metric(kernel_params)

        if metric < best_metric:
            best_metric = metric
            best_params = kernel_params

        # Progress reporting
        if (i + 1) % max(1, n_combinations // 10) == 0:
            print(f"    Progress: {i+1}/{n_combinations} ({100*(i+1)/n_combinations:.1f}%), best {metric_name}: {best_metric:.6e}")

    # Set optimal parameters
    kernel.set_params(best_params)
    # Noise variance remains fixed (not changed by grid search)

    print(f"  Grid search complete. Best {metric_name}: {best_metric:.6e}")
    print(f"  Optimal params: {best_params}, noise: {noise_variance:.3e} (fixed)")

    # Diagnostic: Compute training error with optimal parameters for comparison
    if use_validation:
        K = kernel(X_train)
        K += (noise_variance + 1e-10) * np.eye(K.shape[0])
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y_train)
            alpha = np.linalg.solve(L.T, alpha)

            # Training error
            y_train_pred = K @ alpha
            if y_scaler is not None:
                y_train_pred_orig = y_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).ravel()
                y_train_orig = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
                train_rmse = np.sqrt(np.mean((y_train_orig - y_train_pred_orig) ** 2))
            else:
                train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))

            print(f"  Training RMSE: {train_rmse:.6e}, Validation RMSE: {best_metric:.6e}")
            if best_metric > train_rmse * 1.5:
                print(f"  WARNING: Validation RMSE is {best_metric/train_rmse:.2f}x higher than training RMSE")
                print(f"           This may indicate poor generalization or data mismatch")
        except:
            pass

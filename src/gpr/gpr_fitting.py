#!/usr/bin/env python3
"""
gpr_fitting.py

Gaussian Process Regressor with extensible kernel support.

Extracted from unified_pipeline.py -- computational logic is identical.
The grid search functionality lives in grid_search.py and is called
when use_grid_search=True in fit().
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from src.gpr.kernels import Kernel


class GaussianProcessRegressor:
    """Gaussian Process Regressor with extensible kernel support."""

    def __init__(self, kernel: Kernel, noise_variance: float = 0.0):
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        self.X_scaler = None
        self.y_scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True, n_restarts: int = 3,
            use_grid_search: bool = False, param_grids: Dict = None, max_grid_combinations: int = 5000,
            validation_X: np.ndarray = None, validation_y: np.ndarray = None,
            y_scaler: object = None):
        """Fit the GP to training data.

        Args:
            X: Training inputs (N x D)
            y: Training outputs (N,)
            optimize: Whether to optimize hyperparameters
            n_restarts: Number of restarts for gradient-based optimization (ignored if use_grid_search=True)
            use_grid_search: If True, use grid search instead of gradient-based optimization
            param_grids: Custom parameter grids for grid search (optional)
            max_grid_combinations: Maximum number of grid combinations before random sampling
            validation_X: Validation inputs for grid search evaluation (optional)
            validation_y: Validation outputs for grid search evaluation (optional)
            y_scaler: Scaler for denormalizing predictions (optional)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()

        if optimize:
            if use_grid_search:
                from src.gpr.grid_search import grid_search_hyperparameters, get_default_param_grids
                if param_grids is None:
                    param_grids = get_default_param_grids(self.kernel)
                grid_search_hyperparameters(
                    kernel=self.kernel,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    noise_variance=self.noise_variance,
                    param_grids=param_grids,
                    max_combinations=max_grid_combinations,
                    validation_X=validation_X,
                    validation_y=validation_y,
                    y_scaler=y_scaler,
                )
            else:
                self._optimize_hyperparameters(n_restarts)

        # Compute kernel matrix and its inverse
        K = self.kernel(self.X_train)
        # Add noise variance (0 for pure GP) + small jitter for numerical stability
        K += (self.noise_variance + 1e-10) * np.eye(K.shape[0])

        # Cholesky decomposition for stable inversion
        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L, self.y_train)
            self.alpha = np.linalg.solve(L.T, self.alpha)
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ self.y_train

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict mean and optionally standard deviation at test points."""
        K_star = self.kernel(X, self.X_train)
        mean = K_star @ self.alpha

        if return_std:
            K_star_star = self.kernel(X)
            var = np.diag(K_star_star) - np.sum((K_star @ self.K_inv) * K_star, axis=1)
            std = np.sqrt(np.maximum(var, 0))
            return mean, std

        return mean

    def _optimize_hyperparameters(self, n_restarts: int):
        """Optimize kernel hyperparameters by maximizing log marginal likelihood."""
        def neg_log_marginal_likelihood(params):
            self.kernel.set_params(params[:-1])
            self.noise_variance = np.exp(params[-1])  # Log-transform noise

            K = self.kernel(self.X_train)
            K += self.noise_variance * np.eye(K.shape[0])

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L, self.y_train)
                alpha = np.linalg.solve(L.T, alpha)

                # Negative log marginal likelihood
                nll = 0.5 * (self.y_train @ alpha)
                nll += np.sum(np.log(np.diag(L)))
                nll += 0.5 * len(self.y_train) * np.log(2 * np.pi)

                return nll
            except np.linalg.LinAlgError:
                return 1e10

        # Multiple random restarts
        best_params = None
        best_nll = np.inf

        # Get bounds for optimization
        bounds = self.kernel.bounds + [(np.log(1e-10), np.log(1e-1))]  # Noise variance bounds (log scale)

        for _ in range(n_restarts):
            # Random initialization
            init_params = []
            for low, high in bounds:
                if low > 0 and high / low > 100:  # Log scale for large ranges
                    init_params.append(np.exp(np.random.uniform(np.log(low), np.log(high))))
                else:
                    init_params.append(np.random.uniform(low, high))
            init_params = np.array(init_params)
            init_params[-1] = np.log(self.noise_variance)  # Convert noise to log scale

            # Add current parameters as one of the starting points
            if _ == 0:
                init_params[:-1] = self.kernel.get_params()

            result = minimize(neg_log_marginal_likelihood, init_params, bounds=bounds, method='L-BFGS-B')

            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x

        # Set optimal parameters
        self.kernel.set_params(best_params[:-1])
        self.noise_variance = np.exp(best_params[-1])

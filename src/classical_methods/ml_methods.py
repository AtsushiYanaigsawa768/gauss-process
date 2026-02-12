#!/usr/bin/env python3
"""
ml_methods.py

Machine-learning and local polynomial/rational methods for frequency response
estimation, plus a unified ``create_estimator()`` factory.

Methods included:
- LPMEstimator  (Local Polynomial Method)
- LRMPEstimator (Local Rational Method with Prior poles)
- RFEstimator   (Random Forest)
- GBREstimator  (Gradient Boosting)
- SVMEstimator  (Support Vector Machine)

The ``create_estimator()`` factory function is the public entry point and can
create any estimator -- classical (imported from frequency_domain.py) or
ML-based (defined here).
"""

import numpy as np
from typing import List, Union
from scipy.linalg import lstsq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# =====================================================
# Local Polynomial / Rational Methods
# =====================================================

class LocalPolynomialMethod:
    """Local Polynomial Method (LPM) for nonparametric FRF estimation."""

    def __init__(self, order: int = 2, half_window: int = 5):
        self.order = order
        self.half_window = half_window
        self.omega = None
        self.G_estimate = None

    def fit(self, omega: np.ndarray, Y: np.ndarray, U: np.ndarray,
            estimate_transient: bool = True):
        """
        Fit LPM to frequency-domain data.

        Args:
            omega: Angular frequencies
            Y: Output spectrum
            U: Input spectrum
            estimate_transient: Whether to estimate transient terms
        """
        self.omega = omega
        n_freq = len(omega)
        self.G_estimate = np.zeros_like(Y)

        R = self.order
        n = self.half_window

        for k in range(n_freq):
            # Define local window
            k_min = max(0, k - n)
            k_max = min(n_freq - 1, k + n)
            indices = np.arange(k_min, k_max + 1)

            # Local indices relative to centre
            r = indices - k

            # Build regression matrix
            if estimate_transient:
                # Model: Y = G*U + T with polynomial approximations
                # G = g0 + g1*r + g2*r^2 + ...
                # T = t0 + t1*r + t2*r^2 + ...

                A = []
                # G terms
                for p in range(R + 1):
                    A.append(U[indices] * r**p)
                # T terms
                for p in range(R + 1):
                    A.append(r**p)

                A = np.column_stack(A)
                b = Y[indices]

                # Solve
                theta, _, _, _ = lstsq(A, b)

                # Extract G at centre (r=0)
                self.G_estimate[k] = theta[0]

            else:
                # Simple case: Y = G*U
                A = []
                for p in range(R + 1):
                    A.append(U[indices] * r**p)

                A = np.column_stack(A)
                b = Y[indices]

                theta, _, _, _ = lstsq(A, b)
                self.G_estimate[k] = theta[0]

        return self

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Interpolate to new frequencies."""
        from scipy.interpolate import interp1d

        # Interpolate real and imaginary parts
        interp_real = interp1d(self.omega, np.real(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')
        interp_imag = interp1d(self.omega, np.imag(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')

        return interp_real(omega) + 1j * interp_imag(omega)


class LocalRationalMethodPrior:
    """Local Rational Method with Prior poles (LRMP)."""

    def __init__(self, prior_poles: List[complex], order: int = 5, half_window: int = 10):
        self.prior_poles = prior_poles
        self.order = order
        self.half_window = half_window
        self.omega = None
        self.G_estimate = None

    def _compute_obf_basis(self, z: np.ndarray, poles: List[complex]) -> np.ndarray:
        """Compute Orthonormal Basis Functions (Takenaka-Malmquist)."""
        n_basis = len(poles)
        n_points = len(z)
        B = np.zeros((n_points, n_basis), dtype=complex)

        for b in range(n_basis):
            zeta_b = poles[b]

            # First factor
            B[:, b] = z * np.sqrt(1 - np.abs(zeta_b)**2) / (z - zeta_b)

            # Product of previous factors
            for i in range(b):
                zeta_i = poles[i]
                B[:, b] *= (1 - np.conj(zeta_i) * z) / (z - zeta_i)

        return B

    def fit(self, omega: np.ndarray, Y: np.ndarray, U: np.ndarray, Ts: float = 1.0):
        """Fit LRMP using orthonormal rational basis."""
        self.omega = omega
        n_freq = len(omega)
        self.G_estimate = np.zeros_like(Y)

        # Convert to z-domain
        z = np.exp(1j * omega * Ts)

        n = self.half_window

        for k in range(n_freq):
            # Define local window
            k_min = max(0, k - n)
            k_max = min(n_freq - 1, k + n)
            indices = np.arange(k_min, k_max + 1)

            # Compute basis functions at local points
            z_local = z[indices]
            B = self._compute_obf_basis(z_local, self.prior_poles)

            # Build regression: Y = sum(theta_b^G * B_b * U) + sum(theta_b^T * B_b)
            A = []

            # G terms
            for b_idx in range(B.shape[1]):
                A.append(B[:, b_idx] * U[indices])

            # T terms (transient)
            for b_idx in range(B.shape[1]):
                A.append(B[:, b_idx])

            A = np.column_stack(A)
            b_vec = Y[indices]

            # Solve
            theta, _, _, _ = lstsq(A, b_vec)

            # Reconstruct G at centre point
            B_center = self._compute_obf_basis(np.array([z[k]]), self.prior_poles)
            self.G_estimate[k] = np.sum(theta[:len(self.prior_poles)] * B_center[0, :])

        return self

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Interpolate to new frequencies."""
        from scipy.interpolate import interp1d

        interp_real = interp1d(self.omega, np.real(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')
        interp_imag = interp1d(self.omega, np.imag(self.G_estimate),
                              kind='cubic', fill_value='extrapolate')

        return interp_real(omega) + 1j * interp_imag(omega)


# =====================================================
# Machine Learning Methods
# =====================================================

class MachineLearningRegressor:
    """Base class for ML-based frequency response estimation."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.X_scaler = StandardScaler() if normalize else None
        self.y_real_scaler = StandardScaler() if normalize else None
        self.y_imag_scaler = StandardScaler() if normalize else None
        self.model_real = None
        self.model_imag = None

    def fit(self, omega: np.ndarray, H_measured: np.ndarray):
        """Fit ML model to frequency response data."""
        # Prepare features
        X = omega.reshape(-1, 1)

        # Separate real and imaginary parts
        y_real = np.real(H_measured)
        y_imag = np.imag(H_measured)

        # Normalise if requested
        if self.normalize:
            X = self.X_scaler.fit_transform(X)
            y_real = self.y_real_scaler.fit_transform(y_real.reshape(-1, 1)).ravel()
            y_imag = self.y_imag_scaler.fit_transform(y_imag.reshape(-1, 1)).ravel()

        # Fit models
        self.model_real.fit(X, y_real)
        self.model_imag.fit(X, y_imag)

        return self

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Predict frequency response at new frequencies."""
        X = omega.reshape(-1, 1)

        if self.normalize:
            X = self.X_scaler.transform(X)

        y_real_pred = self.model_real.predict(X)
        y_imag_pred = self.model_imag.predict(X)

        if self.normalize:
            y_real_pred = self.y_real_scaler.inverse_transform(y_real_pred.reshape(-1, 1)).ravel()
            y_imag_pred = self.y_imag_scaler.inverse_transform(y_imag_pred.reshape(-1, 1)).ravel()

        return y_real_pred + 1j * y_imag_pred


class RandomForestFrequencyResponse(MachineLearningRegressor):
    """Random Forest Regression for frequency response estimation."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 normalize: bool = True, **rf_params):
        super().__init__(normalize)
        self.model_real = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               **rf_params)
        self.model_imag = RandomForestRegressor(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               **rf_params)


class GradientBoostingFrequencyResponse(MachineLearningRegressor):
    """Gradient Boosting Regression for frequency response estimation."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, normalize: bool = True, **gb_params):
        super().__init__(normalize)
        self.model_real = GradientBoostingRegressor(n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    max_depth=max_depth,
                                                    **gb_params)
        self.model_imag = GradientBoostingRegressor(n_estimators=n_estimators,
                                                    learning_rate=learning_rate,
                                                    max_depth=max_depth,
                                                    **gb_params)


class SVMFrequencyResponse(MachineLearningRegressor):
    """Support Vector Machine Regression for frequency response estimation."""

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 normalize: bool = True, **svm_params):
        super().__init__(normalize)
        self.model_real = SVR(kernel=kernel, C=C, gamma=gamma, **svm_params)
        self.model_imag = SVR(kernel=kernel, C=C, gamma=gamma, **svm_params)


# =====================================================
# Public aliases
# =====================================================

LPMEstimator = LocalPolynomialMethod
LRMPEstimator = LocalRationalMethodPrior
RFEstimator = RandomForestFrequencyResponse
GBREstimator = GradientBoostingFrequencyResponse
SVMEstimator = SVMFrequencyResponse


# =====================================================
# Unified Factory
# =====================================================

def create_estimator(method: str, **kwargs) -> Union[
    'FrequencyDomainEstimator',
    MachineLearningRegressor,
    LocalPolynomialMethod,
    LocalRationalMethodPrior,
]:
    """Factory function to create estimators by name.

    Classical frequency-domain estimators are imported lazily from
    ``src.classical_methods.frequency_domain`` to keep this module
    lightweight when only ML methods are needed.

    Args:
        method: One of 'nls', 'ls', 'iwls', 'tls', 'ml', 'log',
                'lpm', 'lrmp', 'rf', 'gbr', 'svm'.
        **kwargs: Forwarded to the estimator constructor.

    Returns:
        An estimator instance with ``fit`` and ``predict`` methods.
    """
    from src.classical_methods.frequency_domain import (
        NonlinearLeastSquares,
        LinearLeastSquares,
        IterativelyWeightedLS,
        TotalLeastSquares,
        MaximumLikelihood,
        LogarithmicLeastSquares,
    )

    estimator_map = {
        # Classical methods
        'nls': NonlinearLeastSquares,
        'ls': LinearLeastSquares,
        'iwls': IterativelyWeightedLS,
        'tls': TotalLeastSquares,
        'ml': MaximumLikelihood,
        'log': LogarithmicLeastSquares,

        # Local methods
        'lpm': LocalPolynomialMethod,
        'lrmp': LocalRationalMethodPrior,

        # Machine learning
        'rf': RandomForestFrequencyResponse,
        'gbr': GradientBoostingFrequencyResponse,
        'svm': SVMFrequencyResponse,
    }

    if method not in estimator_map:
        raise ValueError(f"Unknown method: {method}. Available: {list(estimator_map.keys())}")

    return estimator_map[method](**kwargs)

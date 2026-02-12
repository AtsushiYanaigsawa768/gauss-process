#!/usr/bin/env python3
"""
frequency_domain.py

Classical frequency-domain system identification methods.

Methods included:
- NLSEstimator (Nonlinear Least Squares)
- LSEstimator (Linear Least Squares / Levy's method)
- IWLSEstimator (Iteratively Weighted Least Squares / Sanathanan-Koerner)
- TLSEstimator (Total Least Squares with Euclidean norm constraint)
- MLEstimator (Maximum Likelihood for complex Gaussian noise)
- LOGEstimator (Logarithmic Least Squares)

All estimators share a common FrequencyDomainEstimator base class that
models H(jw) = N(jw)/D(jw) as a rational polynomial.
"""

import numpy as np
from typing import Optional
from scipy.optimize import minimize, least_squares
from scipy.linalg import svd, lstsq


# =====================================================
# Base Class
# =====================================================

class FrequencyDomainEstimator:
    """Base class for frequency-domain system identification methods.

    Models a transfer function H(jw) = N(jw)/D(jw) where N and D are
    polynomials parameterised by alpha and beta coefficients respectively.

    Attributes:
        n_numerator:   Degree of the numerator polynomial.
        n_denominator: Degree of the denominator polynomial.
        params:        Concatenated [alpha_0..alpha_n, beta_0..beta_d] after fitting.
        omega:         Angular frequencies used for fitting.
        H_measured:    Measured frequency response used for fitting.
    """

    def __init__(self, n_numerator: int = 2, n_denominator: int = 2):
        self.n_numerator = n_numerator
        self.n_denominator = n_denominator
        self.params = None
        self.omega = None
        self.H_measured = None

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        """Fit the model to frequency response data."""
        raise NotImplementedError

    def predict(self, omega: np.ndarray) -> np.ndarray:
        """Predict frequency response at given frequencies."""
        if self.params is None:
            raise ValueError("Model not fitted yet")
        return self._compute_transfer_function(omega, self.params)

    def _compute_transfer_function(self, omega: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute H(jw) = N(jw)/D(jw) for given parameters."""
        s = 1j * omega

        # Split parameters
        alpha = params[:self.n_numerator + 1]
        beta = params[self.n_numerator + 1:]

        # Compute numerator and denominator
        N = np.polyval(alpha[::-1], s)
        D = np.polyval(beta[::-1], s)

        return N / D


# =====================================================
# Estimator Implementations
# =====================================================

# Aliases used by the factory function in ml_methods.py
NLSEstimator = None   # forward-declared; assigned at module level below
LSEstimator = None
IWLSEstimator = None
TLSEstimator = None
MLEstimator = None
LOGEstimator = None


class NonlinearLeastSquares(FrequencyDomainEstimator):
    """Nonlinear Least Squares (NLS) method."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        # Initial guess
        p0 = np.ones(self.n_numerator + self.n_denominator + 2)
        p0[self.n_numerator + 1] = 1.0  # beta_0 = 1

        # Cost function
        def cost(p):
            H_model = self._compute_transfer_function(omega, p)
            return np.abs(H_measured - H_model)

        # Optimize
        result = least_squares(cost, p0, method='lm')
        self.params = result.x

        return self


class LinearLeastSquares(FrequencyDomainEstimator):
    """Linear Least Squares (LS) method -- Levy's method."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        s = 1j * omega

        # Build regression matrix
        # H(s)D(s) = N(s)  =>  H(s)(beta_0 + beta_1*s + ...) = alpha_0 + alpha_1*s + ...

        # Left side: H(s) * [1, s, s^2, ..., s^d]
        A_left = []
        for k in range(self.n_denominator + 1):
            A_left.append(H_measured * s**k)

        # Right side: [1, s, s^2, ..., s^n]
        A_right = []
        for k in range(self.n_numerator + 1):
            A_right.append(s**k)

        # Stack: left side negative (move to right), skip beta_0
        A = np.column_stack(A_right + [-col for col in A_left[1:]])

        # Target: H(s) * beta_0 (assuming beta_0 = 1)
        b = H_measured

        # Solve least squares (complex)
        x, _, _, _ = lstsq(A, b)

        # Extract parameters -- take real part as we expect real coefficients
        alpha = np.real(x[:self.n_numerator + 1])
        beta = np.ones(self.n_denominator + 1)
        beta[1:] = np.real(x[self.n_numerator + 1:])

        self.params = np.concatenate([alpha, beta])

        return self


class IterativelyWeightedLS(FrequencyDomainEstimator):
    """Iteratively Weighted Linear Least Squares (IWLS) -- Sanathanan-Koerner."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, max_iter: int = 10, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        # Initialise with LS solution
        ls_estimator = LinearLeastSquares(self.n_numerator, self.n_denominator)
        ls_estimator.fit(omega, H_measured)
        params = ls_estimator.params.copy()

        s = 1j * omega

        for iteration in range(max_iter):
            # Compute current denominator for weights
            beta = params[self.n_numerator + 1:]
            D = np.polyval(beta[::-1], s)
            weights = 1.0 / np.abs(D)

            # Weighted regression
            A_right = []
            for k in range(self.n_numerator + 1):
                A_right.append(weights * s**k)

            A_left = []
            for k in range(1, self.n_denominator + 1):
                A_left.append(-weights * H_measured * s**k)

            A = np.column_stack(A_right + A_left)
            b = weights * H_measured

            # Solve
            x, _, _, _ = lstsq(A, b)

            # Update parameters -- take real part
            alpha = np.real(x[:self.n_numerator + 1])
            beta = np.ones(self.n_denominator + 1)
            beta[1:] = np.real(x[self.n_numerator + 1:])

            params = np.concatenate([alpha, beta])

        self.params = params
        return self


class TotalLeastSquares(FrequencyDomainEstimator):
    """Total Least Squares (TLS) with Euclidean norm constraint."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        s = 1j * omega

        # Build augmented matrix [A | b]
        A_right = []
        for k in range(self.n_numerator + 1):
            A_right.append(s**k)

        A_left = []
        for k in range(self.n_denominator + 1):
            A_left.append(-H_measured * s**k)

        # Augmented matrix
        C = np.column_stack(A_right + A_left)

        # SVD of augmented matrix
        U, S, Vt = svd(C, full_matrices=False)

        # Solution is last column of V (last row of Vt)
        x = Vt[-1, :]

        # Normalise to have unit denominator constant
        x = x / x[self.n_numerator + 1]

        # Extract parameters
        alpha = x[:self.n_numerator + 1]
        beta = x[self.n_numerator + 1:]

        self.params = np.concatenate([alpha, beta])

        return self


class MaximumLikelihood(FrequencyDomainEstimator):
    """Maximum Likelihood (ML) estimator for complex Gaussian noise."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray,
            X_measured: Optional[np.ndarray] = None,
            Y_measured: Optional[np.ndarray] = None,
            sigma_x: Optional[np.ndarray] = None,
            sigma_y: Optional[np.ndarray] = None,
            rho: Optional[np.ndarray] = None,
            **kwargs):

        self.omega = omega
        self.H_measured = H_measured

        # If X, Y not provided, assume H = Y/X with X=1
        if X_measured is None:
            X_measured = np.ones_like(H_measured)
        if Y_measured is None:
            Y_measured = H_measured

        # Default noise statistics
        if sigma_x is None:
            sigma_x = np.ones(len(omega)) * 0.01
        if sigma_y is None:
            sigma_y = np.ones(len(omega)) * 0.01
        if rho is None:
            rho = np.zeros(len(omega))

        s = 1j * omega

        def ml_cost(params):
            alpha = params[:self.n_numerator + 1]
            beta = params[self.n_numerator + 1:]

            N = np.polyval(alpha[::-1], s)
            D = np.polyval(beta[::-1], s)

            # Error: N*X - D*Y
            E = N * X_measured - D * Y_measured

            # Denominator of ML cost
            denom = (sigma_x**2 * np.abs(N)**2 +
                    sigma_y**2 * np.abs(D)**2 -
                    2 * np.real(rho * D * np.conj(N)))

            # Avoid division by zero
            denom = np.maximum(denom, 1e-10)

            # ML cost
            cost = np.sum(np.abs(E)**2 / denom)

            return cost

        # Initial guess
        p0 = np.ones(self.n_numerator + self.n_denominator + 2)
        p0[self.n_numerator + 1] = 1.0

        # Optimise
        result = minimize(ml_cost, p0, method='L-BFGS-B')
        self.params = result.x

        return self


class LogarithmicLeastSquares(FrequencyDomainEstimator):
    """Logarithmic Least Squares (LOG) method."""

    def fit(self, omega: np.ndarray, H_measured: np.ndarray, **kwargs):
        self.omega = omega
        self.H_measured = H_measured

        # Take logarithm
        log_H = np.log(H_measured)

        def log_cost(params):
            H_model = self._compute_transfer_function(omega, params)
            # Avoid log of zero/negative
            H_model_safe = np.maximum(np.abs(H_model), 1e-10)
            log_H_model = np.log(H_model_safe)

            # Complex logarithm difference
            return np.abs(log_H_model - log_H)**2

        # Initial guess
        p0 = np.ones(self.n_numerator + self.n_denominator + 2)
        p0[self.n_numerator + 1] = 1.0

        # Optimise
        result = least_squares(log_cost, p0, method='lm')
        self.params = result.x

        return self


# =====================================================
# Public aliases for factory function
# =====================================================

NLSEstimator = NonlinearLeastSquares
LSEstimator = LinearLeastSquares
IWLSEstimator = IterativelyWeightedLS
TLSEstimator = TotalLeastSquares
MLEstimator = MaximumLikelihood
LOGEstimator = LogarithmicLeastSquares

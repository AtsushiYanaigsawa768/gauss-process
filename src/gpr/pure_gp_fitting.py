"""
Pure (from-scratch) Gaussian Process Regression.

Provides :class:`GaussianProcessRegressorPure` for exact GP inference with
Cholesky decomposition, as well as :class:`ComplexGPPure` -- a convenience
wrapper that fits independent GPs on the real and imaginary parts of a
complex transfer function.

Kernels are imported from :mod:`src.gpr.pure_gp_kernels`.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from src.gpr.pure_gp_kernels import (
    Kernel,
    RBFKernel,
    MaternKernel,
    ExpStableKernel,
    TCKernel,
)


# --------------------------------------------------------------------------- #
#                     Exact GP Regressor (O(n^3))                             #
# --------------------------------------------------------------------------- #

class GaussianProcessRegressorPure:
    """Exact Gaussian Process Regression with:
    - Pluggable kernel (RBF, Matern, or ARD RBF + linear + bias)
    - Constant mean mu
    - Homoskedastic noise sigma_n^2
    """

    def __init__(self, kernel: Kernel, log_sigma_n: float = math.log(1e-1),
                 mu: float = 0.0, jitter: float = 1e-10):
        self.kernel = kernel
        self.log_sigma_n = float(log_sigma_n)
        self.mu = float(mu)
        self.jitter = float(jitter)
        self.X_train = None
        self.y_train = None
        self._L = None
        self._alpha = None

    def _Ky(self, X: np.ndarray) -> np.ndarray:
        """Covariance matrix K(X, X) + noise * I."""
        K = self.kernel.K(X, None)
        sn2 = math.exp(2.0 * self.log_sigma_n)
        return K + (sn2 + self.jitter) * np.eye(X.shape[0])

    def _pack(self) -> np.ndarray:
        """Pack all hyperparameters into a flat vector."""
        return np.concatenate([self.kernel.get_theta(),
                               np.array([self.log_sigma_n, self.mu])])

    def _unpack(self, theta_all: np.ndarray) -> None:
        """Unpack a flat hyperparameter vector."""
        kdim = len(self.kernel.get_theta())
        self.kernel.set_theta(theta_all[:kdim])
        self.log_sigma_n = float(theta_all[kdim])
        self.mu = float(theta_all[kdim + 1])

    def _nll_and_grad(self, theta_all: np.ndarray,
                      X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Negative log-marginal-likelihood and its gradient."""
        self._unpack(theta_all)
        n = X.shape[0]
        one = np.ones(n)
        Ky = self._Ky(X)
        yc = y - self.mu * one
        try:
            L = cholesky(Ky, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            Ky = Ky + 1e-8 * np.eye(n)
            L = cholesky(Ky, lower=True, check_finite=False)
        alpha = solve_triangular(
            L.T, solve_triangular(L, yc, lower=True, check_finite=False),
            check_finite=False,
        )
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * yc.dot(alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)

        W = solve_triangular(L, np.eye(n), lower=True, check_finite=False)
        Ky_inv = W.T @ W
        A = np.outer(alpha, alpha) - Ky_inv
        dKs = self.kernel.grad_K_theta(X)
        g_kernel = np.array([0.5 * np.sum(A * dK) for dK in dKs], dtype=float)
        sn2 = math.exp(2.0 * self.log_sigma_n)
        g_logsn = 0.5 * np.sum(A * ((2.0 * sn2) * np.eye(n)))
        g_mu = -one.dot(alpha)
        grad = np.concatenate([g_kernel, np.array([g_logsn, g_mu])])
        return float(nll), grad

    def fit(self, X: np.ndarray, y: np.ndarray, optimize: bool = True,
            maxiter: int = 200, verbose: bool = False):
        """Fit the GP model to training data.

        Args:
            X: Input locations (n, d).
            y: Target values (n,).
            optimize: Whether to optimise hyperparameters via L-BFGS-B.
            maxiter: Maximum optimiser iterations.
            verbose: Print optimiser diagnostics.
        """
        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        assert X.shape[0] == y.shape[0]
        if optimize:
            theta0 = self._pack()

            def fval(t):
                return self._nll_and_grad(t, X, y)[0]

            def gval(t):
                return self._nll_and_grad(t, X, y)[1]

            res = minimize(fval, theta0, jac=gval, method="L-BFGS-B",
                           options={"maxiter": maxiter, "disp": verbose})
            self._unpack(res.x)
        Ky = self._Ky(X)
        L = cholesky(Ky, lower=True, check_finite=False)
        one = np.ones(X.shape[0])
        yc = y - self.mu * one
        alpha = solve_triangular(
            L.T, solve_triangular(L, yc, lower=True, check_finite=False),
            check_finite=False,
        )
        self.X_train = X
        self.y_train = y
        self._L = L
        self._alpha = alpha
        return self

    def predict(self, Xstar: np.ndarray, return_std: bool = True,
                return_cov: bool = False):
        """Predict at new locations.

        Args:
            Xstar: Evaluation points (m, d).
            return_std: Return standard deviations.
            return_cov: Return full posterior covariance.

        Returns:
            (mean, std_or_cov_or_None)
        """
        assert self.X_train is not None
        Xs = np.atleast_2d(Xstar)
        Kxs = self.kernel.K(self.X_train, Xs)
        mu_star = self.mu + Kxs.T @ self._alpha
        if not (return_std or return_cov):
            return mu_star, None
        v = solve_triangular(self._L, Kxs, lower=True, check_finite=False)
        Kss = self.kernel.K(Xs, None)
        cov = Kss - v.T @ v
        cov = 0.5 * (cov + cov.T)
        if return_cov:
            return mu_star, cov
        else:
            std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            return mu_star, std

    def hyperparameter_names(self) -> List[str]:
        return [*self.kernel.hyperparameter_names(), "log_sigma_n", "mu"]

    def get_hyperparameter_vector(self) -> np.ndarray:
        return self._pack()

    def set_hyperparameter_vector(self, theta_all: np.ndarray) -> None:
        self._unpack(theta_all)


# --------------------------------------------------------------------------- #
#                   Complex GP wrapper (real + imag)                           #
# --------------------------------------------------------------------------- #

class ComplexGPPure:
    """Convenience wrapper that fits two independent pure GPs on the real and
    imaginary parts of a complex target G = Re + j*Im.

    X provided as 1-D or 2-D (n, d); internally ensures shape (n, d).
    """

    def __init__(
        self,
        kernel: str = "matern32",
        noise: float = 1e-2,
        optimize: bool = True,
        maxiter: int = 200,
        jitter: float = 1e-10,
    ) -> None:
        self.kernel_name = kernel
        self.noise = float(noise)
        self.optimize = bool(optimize)
        self.maxiter = int(maxiter)
        self.jitter = float(jitter)
        self._gpr_r: Optional[GaussianProcessRegressorPure] = None
        self._gpr_i: Optional[GaussianProcessRegressorPure] = None

    def _make_kernel(self) -> Kernel:
        name = self.kernel_name.lower()
        if name in ("matern12", "matern-1/2", "matern0.5"):
            return MaternKernel(ell=1.0, sigma_f=1.0, nu=0.5)
        if name in ("matern32", "matern-3/2", "matern1.5"):
            return MaternKernel(ell=1.0, sigma_f=1.0, nu=1.5)
        if name in ("matern52", "matern-5/2", "matern2.5"):
            return MaternKernel(ell=1.0, sigma_f=1.0, nu=2.5)
        if name in ("rbf", "se", "squared_exponential"):
            return RBFKernel(ell=1.0, sigma_f=1.0)
        if name in ("exp", "exponential", "exp_stable", "stable_exp"):
            return ExpStableKernel(omega=1.0)
        if name in ("tc", "tuned_correlated", "tuned-correlated"):
            return TCKernel(omega=1.0)
        # Default: Matern 3/2
        return MaternKernel(ell=1.0, sigma_f=1.0, nu=1.5)

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X

    def fit(self, X_train: np.ndarray, G_train: np.ndarray) -> "ComplexGPPure":
        """Fit independent GPs on real and imaginary parts of G_train."""
        X = self._ensure_2d(X_train)
        y_r = np.asarray(np.real(G_train), dtype=float).reshape(-1)
        y_i = np.asarray(np.imag(G_train), dtype=float).reshape(-1)
        kern_r = self._make_kernel()
        kern_i = self._make_kernel()
        self._gpr_r = GaussianProcessRegressorPure(
            kernel=kern_r, log_sigma_n=np.log(self.noise),
            mu=0.0, jitter=self.jitter,
        )
        self._gpr_i = GaussianProcessRegressorPure(
            kernel=kern_i, log_sigma_n=np.log(self.noise),
            mu=0.0, jitter=self.jitter,
        )
        self._gpr_r.fit(X, y_r, optimize=self.optimize,
                        maxiter=self.maxiter, verbose=False)
        self._gpr_i.fit(X, y_i, optimize=self.optimize,
                        maxiter=self.maxiter, verbose=False)
        return self

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        """Predict complex G at evaluation points."""
        assert self._gpr_r is not None and self._gpr_i is not None, \
            "Call fit() before predict()"
        Xe = self._ensure_2d(X_eval)
        mr, _ = self._gpr_r.predict(Xe, return_std=True)
        mi, _ = self._gpr_i.predict(Xe, return_std=True)
        return mr + 1j * mi


# --------------------------------------------------------------------------- #
#                        One-shot convenience helper                          #
# --------------------------------------------------------------------------- #

def fit_predict_complex_gp(
    X_train: np.ndarray,
    G_train: np.ndarray,
    X_eval: np.ndarray,
    kernel: str = "matern32",
    noise: float = 1e-2,
    optimize: bool = True,
    maxiter: int = 200,
) -> np.ndarray:
    """One-shot helper: fit pure GP on complex G and predict at X_eval.

    Fits independent pure GPs on real/imag parts of G and returns complex
    predictions aligned with X_eval.
    """
    model = ComplexGPPure(kernel=kernel, noise=noise,
                          optimize=optimize, maxiter=maxiter)
    model.fit(X_train, G_train)
    return model.predict(X_eval)


# --------------------------------------------------------------------------- #
#                                  Demo                                       #
# --------------------------------------------------------------------------- #

def _demo_1d(savepath_png: str = "gpr_demo.png") -> None:
    """Simple 1-D regression demo with a Matern 3/2 kernel."""
    rng = np.random.default_rng(0)
    X = np.linspace(-5.0, 5.0, 25)[:, None]
    f_true = np.sin(X).ravel() + 0.3 * X.ravel()
    y = f_true + 0.15 * rng.standard_normal(X.shape[0])
    kernel = MaternKernel(ell=1.2, sigma_f=1.0, nu=1.5)
    gp = GaussianProcessRegressorPure(
        kernel=kernel, log_sigma_n=np.log(0.15), mu=0.0, jitter=1e-10,
    )
    gp.fit(X, y, optimize=True, maxiter=150, verbose=False)
    Xs = np.linspace(-6.0, 6.0, 400)[:, None]
    m, s = gp.predict(Xs, return_std=True)
    plt.figure(figsize=(7, 4))
    plt.plot(X, y, "o", label="observations")
    plt.plot(Xs, m, label="predictive mean")
    plt.fill_between(Xs.ravel(), m - 1.96 * s, m + 1.96 * s,
                     alpha=0.2, label="95% CI")
    plt.title("Pure Gaussian Process Regression (Matern 3/2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath_png, dpi=150)
    plt.close()


if __name__ == "__main__":
    _demo_1d("gpr_demo.png")
    print("Demo saved to gpr_demo.png")

"""
From-scratch GP kernel implementations.

Provides a ``Kernel`` base class and concrete kernel implementations
(RBF, ARD RBF + linear + bias, Matern, Exponential-Stable, Tuned-Correlated,
and a Sum kernel) for use with :class:`~src.gpr.pure_gp_fitting.PureGPR`.

All kernels store hyperparameters in log-space for unconstrained optimisation.
"""

import math
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#                          Squared-distance helper                            #
# --------------------------------------------------------------------------- #

def _sqdist(X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of X and X2."""
    X = np.atleast_2d(X)
    if X2 is None:
        X2 = X
    else:
        X2 = np.atleast_2d(X2)
    X_norm = np.sum(X**2, axis=1)[:, None]
    X2_norm = np.sum(X2**2, axis=1)[None, :]
    D = X_norm + X2_norm - 2 * X @ X2.T
    np.maximum(D, 0.0, out=D)
    return D


# --------------------------------------------------------------------------- #
#                              Kernel base class                              #
# --------------------------------------------------------------------------- #

class Kernel:
    """Abstract base for covariance kernels."""

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def get_theta(self) -> np.ndarray:
        raise NotImplementedError

    def set_theta(self, theta: np.ndarray) -> None:
        raise NotImplementedError

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError

    def hyperparameter_names(self) -> List[str]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#                               RBF Kernel                                    #
# --------------------------------------------------------------------------- #

class RBFKernel(Kernel):
    """Isotropic RBF (squared exponential) kernel.

    K = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / ell^2)

    Hyperparameters (log-space): theta = [log_ell, log_sigma_f]
    """

    def __init__(self, ell: float = 1.0, sigma_f: float = 1.0):
        assert ell > 0 and sigma_f > 0
        self.log_ell = math.log(ell)
        self.log_sigma_f = math.log(sigma_f)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X / ell, None if X2 is None else X2 / ell)
        return sf2 * np.exp(-0.5 * D2)

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_ell, self.log_sigma_f], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_ell, self.log_sigma_f = float(theta[0]), float(theta[1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X / ell)
        K = sf2 * np.exp(-0.5 * D2)
        grad_log_ell = K * (D2)
        grad_log_sigma_f = 2.0 * K
        return [grad_log_ell, grad_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_ell", "log_sigma_f"]


# --------------------------------------------------------------------------- #
#                      ARD RBF + Linear + Bias Kernel                         #
# --------------------------------------------------------------------------- #

class ARDRBFLinearBiasKernel(Kernel):
    """ARD RBF + linear + bias kernel.

    K = v0 * exp(-0.5 * sum_l w_l (x_l - x'_l)^2) + a0 + a1 * <x, x'>

    Hyperparameters (log-space): theta = [log_v0, log_w_1..d, log_a0, log_a1]
    """

    def __init__(self, v0: float = 1.0, w: Optional[np.ndarray] = None,
                 a0: float = 1e-6, a1: float = 1e-6):
        self.log_v0 = math.log(max(v0, 1e-12))
        if w is None:
            self.log_w = None
        else:
            w = np.asarray(w, dtype=float)
            assert np.all(w >= 0)
            self.log_w = np.log(np.maximum(w, 1e-12))
        self.log_a0 = math.log(max(a0, 1e-12))
        self.log_a1 = math.log(max(a1, 1e-12))

    def _ensure_log_w(self, X: np.ndarray):
        if self.log_w is None:
            self.log_w = np.zeros(X.shape[1], dtype=float)

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.atleast_2d(X)
        self._ensure_log_w(X)
        v0 = math.exp(self.log_v0)
        w = np.exp(self.log_w)
        a0 = math.exp(self.log_a0)
        a1 = math.exp(self.log_a1)
        if X2 is None:
            X2 = X
        Wsqrt = np.sqrt(w + 1e-32)
        Xw = X * Wsqrt
        X2w = X2 * Wsqrt
        D2 = _sqdist(Xw, X2w)
        K_rbf = v0 * np.exp(-0.5 * D2)
        K_lin = a1 * (X @ X2.T)
        K_bias = a0 * np.ones((X.shape[0], X2.shape[0]))
        return K_rbf + K_bias + K_lin

    def get_theta(self) -> np.ndarray:
        return np.concatenate([
            np.array([self.log_v0]),
            np.asarray(self.log_w),
            np.array([self.log_a0, self.log_a1]),
        ])

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_v0 = float(theta[0])
        d = len(theta) - 3
        self.log_w = np.array(theta[1:1+d], dtype=float)
        self.log_a0 = float(theta[-2])
        self.log_a1 = float(theta[-1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        X = np.atleast_2d(X)
        self._ensure_log_w(X)
        v0 = math.exp(self.log_v0)
        w = np.exp(self.log_w)
        a0 = math.exp(self.log_a0)
        a1 = math.exp(self.log_a1)
        Wsqrt = np.sqrt(w + 1e-32)
        Xw = X * Wsqrt
        D2 = _sqdist(Xw, None)
        K_rbf = v0 * np.exp(-0.5 * D2)
        g_log_v0 = K_rbf.copy()
        grads_w = []
        for l in range(X.shape[1]):
            dx_l = X[:, [l]] - X[:, [l]].T
            g = -0.5 * w[l] * (dx_l**2) * K_rbf
            grads_w.append(g)
        g_log_a0 = a0 * np.ones_like(K_rbf)
        g_log_a1 = a1 * (X @ X.T)
        return [g_log_v0, *grads_w, g_log_a0, g_log_a1]

    def hyperparameter_names(self) -> List[str]:
        names = ["log_v0"]
        if self.log_w is None:
            names += [f"log_w[{i}]" for i in range(0)]
        else:
            names += [f"log_w[{i}]" for i in range(len(self.log_w))]
        names += ["log_a0", "log_a1"]
        return names


# --------------------------------------------------------------------------- #
#                              Matern Kernel                                  #
# --------------------------------------------------------------------------- #

class MaternKernel(Kernel):
    """Matern kernel for nu in {0.5, 1.5, 2.5}.

    K = sigma_f^2 * f_nu(r / ell)

    Hyperparameters (log-space): theta = [log_ell, log_sigma_f]
    """

    def __init__(self, ell: float = 1.0, sigma_f: float = 1.0, nu: float = 1.5):
        assert ell > 0 and sigma_f > 0
        assert nu in (0.5, 1.5, 2.5), "nu must be 0.5, 1.5, or 2.5"
        self.log_ell = math.log(ell)
        self.log_sigma_f = math.log(sigma_f)
        self.nu = float(nu)

    def _form(self, r: np.ndarray, ell: float, sf2: float) -> np.ndarray:
        a = np.sqrt(2.0 * self.nu) * r / ell
        if self.nu == 0.5:
            return sf2 * np.exp(-a)
        elif self.nu == 1.5:
            return sf2 * (1.0 + a) * np.exp(-a)
        elif self.nu == 2.5:
            return sf2 * (1.0 + a + a*a/3.0) * np.exp(-a)
        else:
            raise NotImplementedError

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X, X2)
        r = np.sqrt(D2 + 1e-32)
        return self._form(r, ell, sf2)

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_ell, self.log_sigma_f], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_ell, self.log_sigma_f = float(theta[0]), float(theta[1])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        ell = math.exp(self.log_ell)
        sf2 = math.exp(2.0 * self.log_sigma_f)
        D2 = _sqdist(X, None)
        r = np.sqrt(D2 + 1e-32)
        a = np.sqrt(2.0 * self.nu) * r / ell
        if self.nu == 0.5:
            K = sf2 * np.exp(-a)
            g_log_ell = K * a
        elif self.nu == 1.5:
            K = sf2 * (1.0 + a) * np.exp(-a)
            g_log_ell = K * (a*a) / (1.0 + a)
        elif self.nu == 2.5:
            K = sf2 * (1.0 + a + a*a/3.0) * np.exp(-a)
            denom = (1.0 + a + a*a/3.0)
            g_log_ell = K * (a*a + a*a*a) / (3.0 * denom)
        else:
            raise NotImplementedError
        g_log_sigma_f = 2.0 * K
        return [g_log_ell, g_log_sigma_f]

    def hyperparameter_names(self) -> List[str]:
        return ["log_ell", "log_sigma_f"]


# --------------------------------------------------------------------------- #
#                       Exponential-Stable Kernel                             #
# --------------------------------------------------------------------------- #

class ExpStableKernel(Kernel):
    """Exponential BIBO-stable kernel on non-negative time.

    k(t1, t2) = H(t1) * H(t2) * exp(-omega * (t1 + t2)),  omega > 0

    Parameterised via log_omega for positivity.
    Expects 1-D inputs of shape (n,) or (n, 1).
    """

    def __init__(self, omega: float = 1.0):
        assert omega > 0, "omega must be > 0"
        self.log_omega = math.log(float(omega))

    @staticmethod
    def _as_1d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2 and X.shape[1] == 1:
            return X[:, 0]
        raise ValueError("ExpStableKernel expects 1D inputs (n,) or (n,1)")

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        t = self._as_1d(X)
        s = t if X2 is None else self._as_1d(X2)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        Hs = np.heaviside(s, 0.0)[None, :]
        T = t[:, None]
        S = s[None, :]
        K = Ht * Hs * np.exp(-omega * (T + S))
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_omega], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_omega = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        t = self._as_1d(X)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        T = t[:, None]
        # For gradient, compute training K(X, X)
        Hs = Ht.T
        S = T.T
        K = Ht * Hs * np.exp(-omega * (T + S))
        # dK/d(log_omega) = (dK/domega) * domega/d(log_omega) = -(T+S)*K * omega
        g_log_omega = -omega * (T + S) * K
        return [g_log_omega]

    def hyperparameter_names(self) -> List[str]:
        return ["log_omega"]


# --------------------------------------------------------------------------- #
#                       Tuned-Correlated (TC) Kernel                          #
# --------------------------------------------------------------------------- #

class TCKernel(Kernel):
    """Tuned-Correlated (TC) kernel on non-negative time.

    k(t1, t2) = H(t1) * H(t2) * exp(-omega * max(t1, t2)),  omega > 0

    Parameterised via log_omega for positivity.
    Expects 1-D inputs of shape (n,) or (n, 1).
    """

    def __init__(self, omega: float = 1.0):
        assert omega > 0, "omega must be > 0"
        self.log_omega = math.log(float(omega))

    @staticmethod
    def _as_1d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2 and X.shape[1] == 1:
            return X[:, 0]
        raise ValueError("TCKernel expects 1D inputs (n,) or (n,1)")

    def K(self, X: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        t = self._as_1d(X)
        s = t if X2 is None else self._as_1d(X2)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        Hs = np.heaviside(s, 0.0)[None, :]
        T = t[:, None]
        S = s[None, :]
        M = np.maximum(T, S)
        K = Ht * Hs * np.exp(-omega * M)
        return K

    def get_theta(self) -> np.ndarray:
        return np.array([self.log_omega], dtype=float)

    def set_theta(self, theta: np.ndarray) -> None:
        self.log_omega = float(theta[0])

    def grad_K_theta(self, X: np.ndarray) -> List[np.ndarray]:
        t = self._as_1d(X)
        omega = math.exp(self.log_omega)
        Ht = np.heaviside(t, 0.0)[:, None]
        T = t[:, None]
        Hs = Ht.T
        S = T.T
        M = np.maximum(T, S)
        K = Ht * Hs * np.exp(-omega * M)
        # dK/d(log_omega) = -(M * K) * omega
        g_log_omega = -omega * M * K
        return [g_log_omega]

    def hyperparameter_names(self) -> List[str]:
        return ["log_omega"]
